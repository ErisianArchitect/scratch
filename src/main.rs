#![allow(unused)]
use std::{borrow::Borrow, collections::{HashMap, VecDeque}, path::PathBuf, sync::{atomic::AtomicU64, Arc, Mutex}, time::{Duration, Instant}};

use glam::*;
use scratch::math::*;
use scratch::camera::*;
use scratch::perlin::perlin;
use rayon::prelude::*;
use noise::OpenSimplex;
use scratch::cubemap::*;
use image::{Rgb, RgbImage, Rgba, RgbaImage};
use noise::NoiseFn;
use rand::{rngs::StdRng, Rng, SeedableRng};
use scratch::{camera::Camera, chunk::{self, Chunk, RayHit}, cubemap::Cubemap, math::{self, Face}, open_simplex::open_simplex_2d, perlin::{make_seed, Permutation}, ray::Ray3};
use sha2::digest::typenum::Diff;

// do while loop, do[lo]op.
macro_rules! doop {
    ($($name:lifetime : )? $block:block while $condition:expr) => {
        $($name :)?loop {
            $block
            if !($condition) {
                break $($name)?;
            }
        }
    };
}

/// For toggling code for fast iteration.
/// Syntax:
/// ```rust, no_run
/// // run
/// code_toggle!([r] {
///     println!("This code will be run because of the 'r'.");
/// });
/// // no run
/// code_toggle!([] {
///     println!("There's no 'r' inside the brackets, so this code will not run.");
/// });
/// ```
macro_rules! code_toggle {
    ($([$($kw:ident)?] {$($tokens:tt)*})*) => {
        $(
            code_toggle!{@unwrap; [$($kw)?] { $($tokens)* }}
        )*
    };
    (@unwrap; [r] {$($tokens:tt)*}) => {
        $($tokens)*
    };
    (@unwrap; [] {$($tokens:tt)*}) => {};
}

macro_rules! timed {
    ($fmt:literal => $code:expr) => {
        let elapsed_time = timeit!{
            $code;
        };
        println!($fmt, time=elapsed_time);
    };
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Size {
    pub width: u32,
    pub height: u32,
}

impl std::fmt::Display for Size {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}x{}", self.width, self.height)
    }
}

impl Size {
    pub const fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
        }
    }

    #[inline]
    pub const fn index(self, x: u32, y: u32) -> u32 {
        (y * self.width) + x
    }

    /// Gets the position based on the index.
    #[inline]
    pub const fn inv_index(self, index: u32) -> (u32, u32) {
        (index % self.width, index / self.width)
    }

    #[inline]
    pub const fn iter_width(self) -> std::ops::Range<u32> {
        0..self.width
    }

    #[inline]
    pub const fn iter_height(self) -> std::ops::Range<u32> {
        0..self.height
    }
}

fn main() {
    let start = Instant::now();
    raycast_scene();
    let elapsed = start.elapsed();
    println!("Program finished in {elapsed:.3?}");
}

/// Used to calculate the ray direction towards -Z.
#[derive(Debug, Clone, Copy)]
pub struct RayCalc {
    mult: Vec2,
}

impl RayCalc {
    pub fn new(fov_rad: f32, screen_size: (u32, u32)) -> Self {
        let aspect_ratio = screen_size.0 as f32 / screen_size.1 as f32;
        let tan_fov_half = (fov_rad * 0.5).tan();
        let asp_fov = aspect_ratio * tan_fov_half;
        Self {
            mult: vec2(asp_fov, tan_fov_half),
        }
    }

    pub fn calc_ray_dir(self, ndc: Vec2) -> Vec3A {
        let m = ndc * self.mult;
        vec3a(m.x, m.y, -1.0).normalize()
    }
}

/// OpenSimplex color sampling.
struct PosColor {
    rsimp: OpenSimplex,
    gsimp: OpenSimplex,
    bsimp: OpenSimplex,
    scale: f64,
}

impl PosColor {
    pub fn get_with_scale(&self, pos: Vec3A, scale: f64) -> Vec3A {
        let pos_arr = [pos.x as f64 * scale, pos.y as f64 * scale, pos.z as f64 * scale];
        let r: f64 = self.rsimp.get(pos_arr) * 0.5 + 0.5;
        let g: f64 = self.gsimp.get(pos_arr) * 0.5 + 0.5;
        let b: f64 = self.bsimp.get(pos_arr) * 0.5 + 0.5;
        vec3a(r as f32, g as f32, b as f32)
    }

    pub fn get(&self, pos: Vec3A) -> Vec3A {
        let pos_arr = [pos.x as f64 * self.scale, pos.y as f64 * self.scale, pos.z as f64 * self.scale];
        let r: f64 = self.rsimp.get(pos_arr) * 0.5 + 0.5;
        let g: f64 = self.gsimp.get(pos_arr) * 0.5 + 0.5;
        let b: f64 = self.bsimp.get(pos_arr) * 0.5 + 0.5;
        vec3a(r as f32, g as f32, b as f32)
    }

}

fn norm_u8(norm: f32) -> u8 {
    (norm * 255.0) as u8
}

fn rgb(r: f32, g: f32, b: f32) -> Rgb<u8> {
    Rgb([
        norm_u8(r),
        norm_u8(g),
        norm_u8(b),
    ])
}

#[derive(Debug, Clone, Copy)]
pub struct DirectionalLight {
    pub direction: Vec3A,
    pub color: Vec3A,
    pub intensity: f32,
    pub pre_calc: Vec3A,
    pub inv_dir: Vec3A,
}

impl DirectionalLight {
    pub fn new(direction: Vec3A, color: Vec3A, intensity: f32) -> Self {
        Self {
            direction,
            color,
            intensity,
            pre_calc: color * intensity,
            inv_dir: -direction,
        }
    }
    #[inline(always)]
    pub fn apply(&self, color: Vec3A, normal: Vec3A) -> Vec3A {
        let dot = self.inv_dir.dot(normal);
        (color * self.pre_calc) * dot
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Shadow {
    pub factor: f32,
}

impl Shadow {
    #[inline(always)]
    pub fn apply(self, color: Vec3A) -> Vec3A {
        color * self.factor
    }
}

#[derive(Debug, Clone, Copy)]
pub struct AmbientLight {
    pub color: Vec3A,
    pub intensity: f32,
    pub pre_calc: Vec3A,
}

impl AmbientLight {
    pub fn new(color: Vec3A, intensity: f32) -> Self {
        Self {
            color,
            intensity,
            pre_calc: color * intensity,
        }
    }

    #[inline(always)]
    pub fn apply(self, color: Vec3A) -> Vec3A {
        self.pre_calc * color
    }
}

pub struct Lighting {
    directional: DirectionalLight,
    ambient: AmbientLight,
    shadow: Shadow,
}

impl Lighting {
    #[inline(always)]
    pub fn calculate(&self, color: Vec3A, normal: Vec3A, occluded: bool) -> Vec3A {
        let ambient = self.ambient.color;
        let directional = self.directional.inv_dir.dot(normal);
        let directional_color = self.directional.pre_calc * directional;
        let light = ((1.0 - directional) * self.ambient.intensity) * ambient + directional_color;
        let color = color * light;
        if occluded {
            self.shadow.apply(color)
        } else {
            color
        }
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct Reflection {
    reflectivity: f32,
}

/// Reflects a ray direction on a surface with the given normal.
#[inline(always)]
fn reflect(ray_dir: Vec3A, normal: Vec3A) -> Vec3A {
    ray_dir - 2.0 * (ray_dir * normal) * normal
}

/// Combine reflection color with diffuse based on reflectivity.
#[inline(always)]
fn combine_reflection(reflectivity: f32, diffuse: Vec3A, reflection: Vec3A) -> Vec3A {
    reflectivity * reflection + (1.0 - reflectivity) * diffuse
}

impl Reflection {
    /// Calculates the reflectivity based on the surface normal and view direction.
    #[inline(always)]
    pub fn calculate(
        self,
        surface_normal: Vec3A,
        view_dir: Vec3A,
    ) -> f32 {
        let dot = (-view_dir).dot(surface_normal).max(0.0);
        self.reflectivity + (1.0 - self.reflectivity) * (1.0 - dot).powf(5.0)
    }
}

/// Holds raytracer stuff.
pub struct TraceData {
    chunk: Chunk,
    skybox: Cubemap,
    pos_color: PosColor,
    lighting: Lighting,
    sky_color: Vec3A,
    reflection: Reflection,
    reflection_steps: u16,
}

struct ReflectionAccum {
    color_accum: [Vec3A; 8],
    reflectivity_accum: [f32; 8],
    index: usize,
}

// [diffuse7, diffuse6, diffuse5, diffuse4,
//  diffuse3, diffuse2, diffuse1, diffuse0]
// [
//  diffuse6 = combine(diffuse7, diffuse6, reflectivity6),
//  diffuse5 = combine(diffuse6, diffuse5, reflectivity5),
//  diffuse4 = combine(diffuse5, diffuse4, reflectivity4),
//  diffuse3 = combine(diffuse4, diffuse3, reflectivity3),
//  diffuse2 = combine(diffuse3, diffuse2, reflectivity2),
//  diffuse1 = combine(diffuse2, diffuse1, reflectivity1),
//  diffuse0 = combine(diffuse1, diffuse0, reflectivity0),
// ]

impl ReflectionAccum {
    pub const fn new() -> Self {
        Self {
            color_accum: [Vec3A::NAN; 8],
            reflectivity_accum: [f32::NAN; 8],
            index: 7,
        }
    }
}

impl TraceData {
    /// Calculates the color of a hit point before reflection calculation. This will also calculate shadow.
    #[inline(always)]
    pub fn calc_color_before_reflections(&self, hit_point: Vec3A, hit_normal: Vec3A) -> Vec3A {
        let color = self.pos_color.get(hit_point);
        let light_ray = Ray3::new(hit_point, self.lighting.directional.inv_dir);
        let light_hit = self.chunk.raycast(light_ray, 100.0);
        self.lighting.calculate(color, hit_normal, light_hit.is_some())
    }

    /// Calculates the grayscale color of a hit point before reflection calculation without a diffuse color. This will also calculate shadow.
    #[inline(always)]
    pub fn calc_color_before_reflections_no_diffuse(&self, hit_point: Vec3A, hit_normal: Vec3A) -> Vec3A {
        let color = Vec3A::ONE;
        let light_ray = Ray3::new(hit_point, self.lighting.directional.inv_dir);
        let light_hit = self.chunk.raycast(light_ray, 112.0);
        self.lighting.calculate(color, hit_normal, light_hit.is_some())
    }

    // pub fn trace_reflections_iterative(&self, ray: Ray3, near: f32, far: f32, steps: u16) -> Vec3A {
    //     let Some(hit) = self.chunk.raycast(ray, far) else {
    //         return self.sky_color(ray.dir);
    //     };
    //     let Some(face) = hit.face else {
    //         return Vec3A::X;
    //     };
    //     let hit_point = hit.get_hit_point(ray, face);
    //     let hit_normal = face.normal();
        
    //     for _step in 0..steps {

    //     }
    // }

    /// Trace reflections for exact number of steps and calculate the color.
    pub fn trace_reflections(&self, ray: Ray3, near: f32, far: f32, steps: u16) -> Vec3A {
        let Some(hit) = self.chunk.raycast(ray, far) else {
            // no hit, so return the sky color.
            return self.sky_color(ray.dir);
        };
        let Some(face) = hit.face else {
            // If there was no hit face, return Red so that it's clear that no hit was made.
            // this is useful for debugging since the camera will most likely be outside of the bounds of the scene, so there should be no red.
            // If there is red visible, that means that the ray is penetrating too far into the cube.
            return Vec3A::X;
        };
        // The ray is too close, so return Green. (for debugging purposes)
        if hit.distance < near {
            return Vec3A::Y;
        }

        let hit_point = hit.get_hit_point(ray, face);
        let hit_normal = face.normal();
        /// easy way to mix up colors is to create a checkerboard pattern.
        let checker = checkerboard(hit.coord.x, hit.coord.y, hit.coord.z);
        let checker_color = if checker {
            Vec3A::ONE
        } else {
            Vec3A::splat(0.3)
        };
        let mut diffuse = self.calc_color_before_reflections(hit_point, hit_normal);
        // apply the checkerboard pattern.
        diffuse = diffuse * checker_color;
        /// Edge detection.
        #[inline(always)]
        fn on_edge(x: f32, y: f32) -> bool {
            x < 0.05 || y < 0.05 || x >= 0.95 || y >= 0.95
        }
        /// Check if we need to trace more reflections. (this will test if the hit surface is reflective).
        if steps != 0 && self.chunk.get_reflection(hit.coord.x, hit.coord.y, hit.coord.z) {
            let reflect_dir = reflect(ray.dir, hit_normal);
            let reflect_ray = Ray3::new(hit_point, reflect_dir);
            let reflectivity = self.reflection.calculate(hit_normal, ray.dir);
            let reflection = self.trace_reflections(reflect_ray, 0.0, far - hit.distance, steps - 1);
            let final_color = combine_reflection(reflectivity, diffuse, reflection);
            final_color
        } else {
            // no reflection in this branch, so just return the diffuse color.
            // Edge detection to draw grid-lines.
            let hit_fract = hit_point.fract();
            match face {
                Face::PosX | Face::NegX if on_edge(hit_fract.y, hit_fract.z) => diffuse = diffuse * 0.1,
                Face::PosY | Face::NegY if on_edge(hit_fract.x, hit_fract.z) => diffuse = diffuse * 0.1,
                Face::PosZ | Face::NegZ if on_edge(hit_fract.x, hit_fract.y) => diffuse = diffuse * 0.1,
                _ => (),
            }
            diffuse
        }
    }

    /// This is the meat of the program. Just a simple `trace_color` method that returns the color that a ray "receives".
    /// This includes lighting and reflection calculations.
    #[inline(always)]
    pub fn trace_color(&self, ray: Ray3, near: f32, far: f32) -> Vec3A {
        self.trace_reflections(ray, near, far, self.reflection_steps)
    }

    /// Calculate the sky color given the ray direction. This will sample a cubemap.
    #[inline(always)]
    pub fn sky_color(&self, ray_dir: Vec3A) -> Vec3A {
        let rgb = self.skybox.sample_dir(ray_dir);
        let color = vec3a(
            math::byte_scalar(rgb.0[0]),
            math::byte_scalar(rgb.0[1]),
            math::byte_scalar(rgb.0[2]),
        );
        self.lighting.ambient.apply(color * self.sky_color)
    }
}

/// For creating a 3D checkerboard pattern.
#[inline(always)]
fn checkerboard(x: i32, y: i32, z: i32) -> bool {
    ((x & 1) ^ (y & 1) ^ (z & 1)) != 0
}

pub fn raycast_scene() {
    println!("Starting.");
    let start = Instant::now();
    let skybox = {
        let directory = PathBuf::from("./assets/textures/skybox_001/");
        let top_path = directory.join("purp_top.png");
        let bottom_path = directory.join("purp_bottom.png");
        let left_path = directory.join("purp_left.png");
        let right_path = directory.join("purp_right.png");
        let front_path = directory.join("purp_front.png");
        let back_path = directory.join("purp_back.png");
        let pos_x = read_texture(right_path).unwrap();
        let pos_y = read_texture(top_path).unwrap();
        let pos_z = read_texture(back_path).unwrap();
        let neg_x = read_texture(left_path).unwrap();
        let neg_y = read_texture(bottom_path).unwrap();
        let neg_z = read_texture(front_path).unwrap();
    
        Cubemap::new(pos_x, pos_y, pos_z, neg_x, neg_y, neg_z)
    };
    let elapsed = start.elapsed();
    println!("Loaded Cubemap in {elapsed:.3?}");
    /// The seed for PosColor.
    const SEED: u32 = 1205912;
    // let simplex = OpenSimplex::new(SEED);
    let rsimp = OpenSimplex::new(SEED + 0);
    let gsimp = OpenSimplex::new(SEED + 1);
    let bsimp = OpenSimplex::new(SEED + 2);

    let pos_color = PosColor {
        rsimp,
        gsimp,
        bsimp,
        scale: 1.0,
    };
    // Super Sampling Anti-aliasing.
    // Setting this to true means that the scene will render at twice the resolution then downsample to the target resolution.
    // Perhaps the more optimal way to do this would be to shoot for rays for each pixel instead of just one, but I haven't gotten that far yet.
    const SSAA: bool = false;
    // let perm = Permutation::from_seed(make_seed(SEED as u64));
    // I've left several grid sizes here to test out different resolutions.
    // let size = GridSize::new(512, 512);
    // let size = GridSize::new(2048, 2048);
    // let size = GridSize::new(640, 480);
    // let size = GridSize::new(1280, 720);
    // let size = GridSize::new(1920, 1080); // FHD
    let size = Size::new(1920*2, 1080*2); // 4K
    // let size = GridSize::new(1920*4, 1080*4); // 8K

    let size = if SSAA {
        Size::new(size.width * 2, size.height * 2)
    } else {
        size
    };
    // Different camera views for the scene.
    // favorite cam
    // let mut cam = Camera::from_look_at(vec3a(-24.0, 70.0-12.0, 48.0), vec3a(32., 32.-12., 32.), 45.0f32.to_radians(), 1.0, 100.0, (size.width, size.height));
    // let mut cam = Camera::from_look_at(vec3a(-24.0, 70.0-2.0, 48.0+20.0), vec3a(0., 42.5+10.0, 32.5+20.0), 45.0f32.to_radians(), 1.0, 100.0, (size.width, size.height));
    let mut cam = Camera::from_look_at(vec3a(-24.0, 70.0-12.0, 48.0+10.0), vec3a(0., 42.5, 32.5+10.0), 90.0f32.to_radians(), 1.0, 100.0, (size.width, size.height));
    // let mut cam = Camera::from_look_at(vec3a(-24.0, 70.0-12.0, 64.0+24.0), vec3a(32., 32.-12., 32.), 90.0f32.to_radians(), 1.0, 100.0, (size.width, size.height));
    // let mut cam = Camera::from_look_at(vec3a(42.0, 7.0, 42.0), vec3a(32., 5., 32.), 90.0f32.to_radians(), 1.0, 100.0, (size.width, size.height));
    // let mut cam = Camera::from_look_at(vec3a(24.0, 24.0, 16.0), vec3a(32.0, 8.0, 32.0), 90.0f32.to_radians(), 1.0, 100.0, (size.width, size.height));
    
    let mut chunk = Chunk::new();
    let mut trace = TraceData {
        chunk,
        skybox,
        pos_color,
        lighting: Lighting {
            directional: DirectionalLight::new(
                vec3a(0.5, -1.0, -1.0).normalize(),
                vec3a(1.0, 1.0, 1.0),
                1.0,
            ),
            ambient: AmbientLight::new(
                // color
                Vec3A::ONE,
                // intensity
                1.0,
            ),
            shadow: Shadow {
                factor: 0.2,
            },
        },
        // sky_color: Vec3A::splat(0.4),
        // sky_color: Vec3A::splat(0.0),
        sky_color: Vec3A::splat(2.0),
        reflection: Reflection { reflectivity: 0.5 },
        // reflection: Reflection { reflectivity: 1.0 },
        reflection_steps: 3,
    };

    let ray_calc = RayCalc::new(cam.fov, cam.screen_size);
    let cam_pos = cam.position;
    // by creating a reference, then moving it into the closure, we get better performance than if we didn't use a move closure and didn't use a reference.
    // I'm not sure why, but that's just how it is.
    let cam_ref = &cam;
    let cam_rot = cam.rotation_matrix();
    let start = Instant::now();
    let rays = (0..size.width*size.height).into_par_iter().map(move |i| {
        let (x, y) = size.inv_index(i);
        // NDC (normalized device coordinate) calculation.
        let xy = vec2(x as f32, (size.height - y) as f32);
        let wh = vec2(size.width as f32, size.height as f32);
        let hs = vec2(0.5, 0.5);
        let screen_pos = (xy / wh) - hs;
        let dir = ray_calc.calc_ray_dir(screen_pos);
        cam_rot * dir
    }).collect::<Vec<_>>();
    let elapsed = start.elapsed();
    println!("Calculated {} rays in {elapsed:.3?}", rays.len());

    // Much of the code after this point is toggleable code for fast iteration. If you want to get to the raytracing part, skip ahead.
    // I'll put up a big sign that says "RAYTRACING".

    /// This is for generating some simple terrain for testing purposes.
    code_toggle!([] {
        let start = Instant::now();
        for x in 0..64 {
            for z in 0..64 {
                let min_dist = x.min(z).min(63-x).min(63-z);
                let falloff = (min_dist as f32) / 32.0;
                let height = (perlin(&perm, x as f32 / 64.0, z as f32 / 64.0) * 0.5) + 0.5;
                let h = ((height * 64.0
                     * falloff
                    ) as i32).max(0);
                trace.chunk.fill_box((x, 0, z), (x+1, h, z+1), true);
            }
        }
        let elapsed = start.elapsed();
        println!("Generated terrain in {elapsed:.3?}");
    });
    fn box_at(start: (i32, i32, i32)) -> std::ops::Range<(i32, i32, i32)> {
        start..(start.0 + 6, start.1 + 6, start.2 + 6)
    }
    fn next_start(end: (i32, i32, i32)) -> (i32, i32, i32) {
        (end.0 - 3, end.1 - 3, end.2 - 3)
    }
    
    fn boxes(start: (i32, i32, i32), chunk: &mut Chunk) {
        let mut r = box_at(start);
        while r.end.0 <= 64 && r.end.1 <= 64 && r.end.2 <= 64 {
            if rand::random::<bool>()
            || true
            {
                chunk.draw_box(r.start, r.end, true);
                chunk.draw_box_reflection(r.start, r.end, true);
                if false
                || true
                {
                    let end = (r.end.0, r.start.1 + 1, r.end.2);
                    chunk.fill_box(r.start, end, true);
                    chunk.fill_box_reflection(r.start, end, true);
                    // let m = 1;
                    // let start = (
                    //     r.start.0 + m, r.start.1 + m, r.start.2 + m
                    // );
                    // let end = (
                    //     end.0 - m, end.1 + m, end.2 - m
                    // );
                    // chunk.fill_box(start, end, true);
                    // let m = 1;
                    // let start = (
                    //     start.0 + m, start.1 + m, start.2 + m
                    // );
                    // let end = (
                    //     end.0 - m, end.1 + m, end.2 - m
                    // );
                    // chunk.fill_box(start, end, true);
                    break;
                }
            }
            r = box_at(next_start(r.end));
        }
    }
    code_toggle!([r] {
        let start = Instant::now();
        for y in 0..64/10 {
            for z in 0..64/10 {
                for x in 0..64/10 {
                    let mut r = box_at((x * 10, y * 10, z * 10));
                    trace.chunk.draw_box(r.start, r.end, true);
                    let end = (r.end.0, r.start.1 + 1, r.end.2);
                    trace.chunk.fill_box(r.start, end, true);
                    if (x == 0 || z == 5) {
                        trace.chunk.fill_box_reflection(r.start, end, true);
                        // trace.chunk.draw_box_reflection(r.start, r.end, false);
                    }
                    // boxes((x * 10, y * 10, z * 10), &mut trace.chunk);
                }
            }
        }
        // trace.chunk.set(2, 41+10, 32+20, true);
        // trace.chunk.set_reflection(2, 41+10, 32+20, true);
        trace.chunk.set(2, 41, 32+10, true);
        // trace.chunk.set_reflection(2, 41, 32, true);
        let elapsed = start.elapsed();
        code_toggle!([r] {

        });
        trace.chunk.fill_box((0, 46, 0), (64, 64, 64), false);
        let st = 56/2 - 4;
        let en = st + 8;
        let b = 45;
        let t = b + 8;
        trace.chunk.draw_box((st, b, st), (en, t, en), true);
        trace.chunk.fill_box((0, 45, 0), (56, 46, 56), true);
        trace.chunk.fill_box_reflection((0, 45, 0), (56, 46, 56), true);
        println!("Placed blocks in {elapsed:.3?}");
    });
    // Apartments
    code_toggle!([] {
        for z in 0..16 {
            for x in 0..16 {
                for y in 0..16 {
                    let start = ivec3(x * 4, y * 4, z * 4);
                    let end = start + ivec3(4, 1, 4);
                    trace.chunk.fill_box((start.x, start.y, start.z), (end.x, end.y, end.z), true);
                    trace.chunk.fill_box_reflection((start.x+1, start.y, start.z+1), (end.x, end.y, end.z), true);
                    trace.chunk.fill_box((start.x, start.y + 1, start.z), (start.x+1, start.y + 4, start.z+1), true);
                    // trace.chunk.fill_box_reflection((start.x, start.y + 1, start.z), (start.x+1, start.y + 4, start.z+1), true);
                }
            }
        }
    });
    code_toggle!([r] {
        for z in 0..64 {
            for x in 0..64 {
                for y in 0..64 {
                    // let present = trace.chunk.get(x, y, z);
                    // if present && checkerboard(x, y, z) {
                    // }
                    trace.chunk.set_reflection(x, y, z, true);
                }
            }
        }
        let st = 56/2 - 4;
        let en = st + 8;
        let b = 46;
        let t = b + 7;
        trace.chunk.fill_box_reflection((st, b, st), (en, t, en), false);
        trace.chunk.set_reflection(2, 41, 32+10, false);
    });
    code_toggle!([] {
        const HELLO: [&'static str; 5] = [
            "1010111010001000111000101010111011001000110",
            "1010100010001000101000101010101010101000101",
            "1110111010001000101000101010101011001000101",
            "1010100010001000101000101010101010101000101",
            "1010111011101110111000111110111010101110110",
        ];
        trace.chunk.fill_box((0, 31, 0), (12, 31+12, 63), false);
        trace.chunk.fill_box((0, 30, 0), (63, 31, 63), true);
        for i in 0..HELLO.len() {
            for j in 0..HELLO[i].len() {
                let y = 36 - i as i32;
                let z = 21 + j as i32;
                if HELLO[i].as_bytes()[j] == b'1' {
                    trace.chunk.set(0, y, z, true);
                }
            }
        }
    });

    // Reflection scene
    code_toggle!([] {
        trace.chunk.fill_box((0, 0, 0), (64, 2, 64), true);
        trace.chunk.fill_box_reflection((26, 0, 26), (38, 2, 38), true);
        trace.chunk.fill_box((31, 4, 31), (33, 6, 33), true);
        trace.chunk.fill_box_reflection((31, 4, 31), (33, 6, 33), true);
    });

    code_toggle!([] {
        boxes((0, 0, 0), &mut trace.chunk);
        trace.chunk.fill_box((0, 0+16, 0), (64, 1+16, 64), true);
        trace.chunk.fill_box((31, 1+16, 31), (32, 16+16, 32), true);
    });

    code_toggle!([] {
        trace.chunk.fill_box((0, 0, 0), (64, 1, 64), true);
        for i in 0..4 {
            for j in 0..4 {
                for k in 0..4 {
                    let start = (i * 4 + 16, j * 4, k * 4 + 16);
                    let end = (start.0 + 5, start.1 + 5, start.2 + 5);
                    trace.chunk.draw_box(start, end, true);
                }
            }
        }
        trace.chunk.fill_box((17, 1, 17), (48, 31, 48), false);
    });

    // for x in 0..64 {
    //     for z in 0..64 {
    //         for y in 0..64 {
    //             let b = trace.chunk.get(x, y, z);
    //             let cb1 = checkerboard(x, y, z);
    //             let cb2 = checkerboard(x/2, y/2, z/2);
    //             let cb3 = checkerboard(x/4, y/4, z/4);
    //             trace.chunk.set(x, y, z, b && cb1 && cb2 && cb3);
    //         }
    //     }
    // }
    
    let mut img = RgbImage::new(size.width, size.height);

    let near = 0.1;
    let far = 250.0;
    // the following commented out line is for normalized depth calculation (0 for closest, 1 for furthest).
    // let depth_mul = 1.0 / (far - near);
    // #############################################
    // #                RAYTRACING!                #
    // #############################################
    let start = std::time::Instant::now();
    {
        let trace = &trace;
        let rays = rays.as_slice();
        img.par_pixels_mut().enumerate().for_each(move |(i, pixel)| {
            let ray = Ray3::new(cam_pos, rays[i]);
            let color = trace.trace_color(ray, near, far);
            *pixel = rgb(color.x, color.y, color.z);
            return;
        });
    }
    let elapsed = start.elapsed();
    println!("Rendered {size} image in {elapsed:.3?}");

    use image::*;
    if SSAA {
        let size = Size::new(size.width/2, size.height/2);
        println!("SSAA downscale to {size}");
        let img = DynamicImage::ImageRgb8(img);
        let resized = img.resize_exact(size.width, size.height, imageops::FilterType::Gaussian);
        resized.save("raycast.png").expect("Failed to save image.");
    } else {
        img.save("raycast.png").expect("Failed to save image.");
    }
}