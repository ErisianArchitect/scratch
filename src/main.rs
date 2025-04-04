#![allow(unused)]
use std::{borrow::Borrow, collections::{HashMap, VecDeque}, sync::{atomic::AtomicU64, Arc, Mutex}, time::{Duration, Instant}};

use glam::*;
use image::{Rgb, RgbImage, Rgba, RgbaImage};
use itertools::Itertools;
use noise::{NoiseFn, OpenSimplex};
use rand::{rngs::StdRng, Rng, SeedableRng};
use scratch::{camera::Camera, chunk::{self, Chunk, RayHit}, dag::*, diff_plot::DiffPlot, math::Face, open_simplex::open_simplex_2d, perlin::{make_seed, Permutation}, ray::Ray3, tracegrid::{GridSize, TraceGrid}};
use sha2::digest::typenum::Diff;

macro_rules! count_ids {
    () => { 0 };
    ($first:ident $(,$rest:ident)*$(,)?) => {
        (1 + count_ids!($($rest),*))
    };
}

// Goal:
// Have global registry of serializable types with IDs.

trait Fnord {
    fn foo(&self);
    fn bar(&self);
    fn baz(&self);
}

impl Fnord for i32 {
    fn foo(&self) {
        println!("i32::foo()");
    }

    fn bar(&self) {
        println!("i32::bar()");
    }

    fn baz(&self) {
        println!("i32::baz()");
    }
}

impl Fnord for &'static str {
    fn foo(&self) {
        println!("str::foo()");
    }

    fn bar(&self) {
        println!("str::bar()");
    }

    fn baz(&self) {
        println!("str::baz()");
    }
}

struct Fred {
    fnord: Box<dyn Fnord>,
}

impl Fred {
    pub fn new<T: Fnord + 'static>(value: T) -> Self {
        Self {
            fnord: Box::new(value),
        }
    }
}

impl std::ops::Deref for Fred {
    type Target = dyn Fnord;

    fn deref(&self) -> &Self::Target {
        self.fnord.as_ref()
    }
}

fn take_box<T: Into<Option<Box<u32>>>>(value: T) -> Option<Box<u32>> {
    value.into()
}

// fn expect<A, R: Try<Output>>(v: )

fn calculate_perf_tri() {
    let squares: [i64; 512] = std::array::from_fn(|i| (i as i64).pow(2));
    let squares_map: HashMap<i64, i64> = HashMap::from_iter(squares.iter().cloned().enumerate().map(|(i, sq)| { (sq, i as i64) }));
    for i in 1..512 {
        for j in 1..512 {
            let comb = squares[i] + squares[j];
            if let Some(&sq) = squares_map.get(&comb) {
                println!("{i} * {i} + {j} * {j} = {sq} * {sq}");
            }
        }
    }
}

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

fn plot_circle<F: FnMut(i32, i32)>(x: i32, y: i32, r: i32, mut f: F) {
    let mut px = -r;
    let mut py = 0;
    let mut err = 2-2*r;
    let mut r = r;
    doop!({
        f(x - px, y + py);
        f(x - py, y - px);
        f(x + px, y - py);
        f(x + py, y + px);
        r = err;
        if r <= py {
            py += 1;
            err += py * 2 + 1;
        }
        if r > px || err > py {
            px += 1;
            err += px * 2 + 1;
        }
    } while px < 0)
}

fn fill_circle<F: FnMut(i32, i32)>(x: i32, y: i32, r: i32, mut f: F) {
    let mut px = -r;
    let mut py = 0;
    let mut err = 2-2*r;
    let mut r = r;
    doop!({
        // f(x - px, y + py);
        // f(x + px, y - py);
        // f(x - py, y - px);
        // f(x + py, y + px);
        for i in (x + px)..=(x - px) {
            f(i, y + py);
            f(i, y - py);
        }
        r = err;
        if r <= py {
            py += 1;
            err += py * 2 + 1;
        }
        if r > px || err > py {
            px += 1;
            err += px * 2 + 1;
        }
    } while px < 0)
}

fn hor_line<F: FnMut(i32, i32)>(x_range: std::ops::Range<i32>, y: i32, mut f: F) {
    for x in x_range {
        f(x, y);
    }
}

// prototype!(
//     do!()
// );

fn gridsize_test() {
    let size = GridSize::new(2048, 2048);
    let mut indices = vec![(0u32, 0u32); (size.width*size.height) as usize];
    const ITERATIONS: usize = 1000;
    let mut timings = Vec::<Duration>::with_capacity(ITERATIONS);
    let mut work_proof = 0u32;
    for _ in 0..ITERATIONS {
        let start = Instant::now();
        for y in 0..size.height {
            for x in 0..size.width {
                let index = size.index(x, y);
                work_proof = work_proof.wrapping_add(index);
                indices[index as usize] = (x, y);
            }
        }
        work_proof = work_proof.wrapping_add(work_proof);
        let elapsed = start.elapsed();
        timings.push(elapsed);
    }
    println!("Work Proof: {work_proof}");
    let avg_count = timings.len() as u32;
    let avg_time = timings.into_iter().sum::<Duration>() / avg_count;
    println!("Iterations: {ITERATIONS}");
    println!("Average time: {avg_time:.3?}");
}

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

#[inline(always)]
pub const fn morton6(index: u32) -> u32 {
    let step1 = (index | (index << 6)) & 0b0000111000000111;
    let step2 = (step1 | (step1 << 2)) & 0b0011001000011001;
    let step3 = (step2 | (step2 << 2)) & 0b1001001001001001;
    return step3;
}

#[inline(always)]
pub fn fast_morton(index: u32, mask: u32) -> u32 {
    use std::arch::x86_64::_pdep_u32;
    unsafe {
        _pdep_u32(index, mask)
    }
}

#[inline(always)]
pub fn fast_morton6_3_x(index: u32) -> u32 {
    const SCATTER: u32 = 0b1001001001001001;
    fast_morton(index, SCATTER)
}

#[inline(always)]
pub fn fast_morton6_3_y(index: u32) -> u32 {
    const SCATTER: u32 = 0b10010010010010010;
    fast_morton(index, SCATTER)
}

#[inline(always)]
pub fn fast_morton6_3_z(index: u32) -> u32 {
    const SCATTER: u32 = 0b100100100100100100;
    fast_morton(index, SCATTER)
}

#[inline(always)]
pub fn morton6_3(x: u32, y: u32, z: u32) -> u32 {
    fast_morton6_3_x(x) | fast_morton6_3_y(y) | fast_morton6_3_z(z)
}

pub struct RaytraceChunk {
    blocks: Box<[u32]>,
}

impl RaytraceChunk {
    pub fn new() -> Self {
        Self {
            blocks: (0..64*64*64).map(|_| 0u32).collect(),
        }
    }

    pub fn get(&self, x: i32, y: i32, z: i32) -> u32 {
        let x = x as u32 & 0b111111;
        let y = y as u32 & 0b111111;
        let z = z as u32 & 0b111111;

        let index = ((y << 12) | (z << 6) | x) as usize;
        self.blocks[index]
    }

    pub fn get_morton(&self, x: i32, y: i32, z: i32) -> u32 {
        let index = morton6_3(x as u32, y as u32, z as u32);
        self.blocks[index as usize]
    }

    pub fn set(&mut self, x: i32, y: i32, z: i32, id: u32) {
        let x = x as u32 & 0b111111;
        let y = y as u32 & 0b111111;
        let z = z as u32 & 0b111111;

        let index = ((y << 12) | (z << 6) | x) as usize;
        self.blocks[index] = id;
    }

    pub fn set_morton(&mut self, x: i32, y: i32, z: i32, id: u32) {
        let index = morton6_3(x as u32, y as u32, z as u32);
        self.blocks[index as usize] = id;
    }
}

macro_rules! timeit {
    ($($tokens:tt)*) => {
        {
            let timeit_start = Instant::now();
            $($tokens)*
            timeit_start.elapsed()
        }
    };
}

fn main() {
    // gridsize_test();
    code_toggle!(
        [] {
            gridsize_test();
        }
        [r] {
            let start = Instant::now();
            raycast_scene();
            let elapsed = start.elapsed();
            println!("Program finished in {elapsed:.3?}");
        }
        [] {
            morton_test();
        }
        [] {
            glam_test();
        }
    );
    return;
}

// fn calc_ray(ndc: Vec2) -> Ray3 {

// }

// fn rot_dir(dir: Vec3, rot: Vec2) -> Vec3 {
//     // let rot = vec2(rot.x, rot.y);
//     let mut dir = dir;

//     let cy = rot.y.cos();
//     let sy = rot.y.sin();

//     let xay = dir.x * cy + dir.z * sy;
//     let zay = dir.x * sy - dir.z * cy;

//     dir.x = xay;
//     dir.z = zay;

//     let cx = rot.x.cos();
//     let sx = rot.x.sin();

//     let yax = dir.y * cx + dir.z * sx;
//     let zax = dir.y * sx - dir.z * cx;

//     dir.y = yax;
//     dir.z = zax;

//     dir
// }

#[inline(always)]
fn rot_dir(dir: Vec3, rot: Vec2) -> Vec3 {
    let rot = vec2(rot.x, rot.y);
    let mut dir = dir;
    let [x, y, z] = dir.to_array();
    let (sy, cy) = rot.y.sin_cos();
    let (sp, cp) = rot.x.sin_cos();

    let yz_reg = vec4(y, y, -z, z);
    let p_reg = vec4(cp, sp, sp, cp);
    let pm_res = yz_reg * p_reg;
    let yp_reg = vec2(pm_res.x, pm_res.y);
    let zp_reg = vec2(pm_res.z, pm_res.w);
    let yz = yp_reg + zp_reg;
    let xz_reg = vec4(x, -x, yz.y, yz.y);
    let y_reg = vec4(cy, sy, sy, cy);
    let ym_res = xz_reg * y_reg;
    let xp_reg = vec2(ym_res.x, ym_res.y);
    let zp_reg = vec2(ym_res.z, ym_res.w);
    let xz = xp_reg + zp_reg;

    vec3(xz.x, yz.y, xz.y)
    
    // let xp = x;
    // let yp = y * cp - z * sp;
    // let zp = y * sp + z * cp;

    // let x = xp * cy + zp * sy;
    // let y = yp;
    // let z = -xp * sy + zp * cy;
    // vec3(x, y, z)
}

#[test]
fn glam_test() {
    use glam::*;

    let x = -0.73;
    let y = -0.5;
    let fov = 45f32.to_radians();
    let aspect_ratio = 1920.0 / 1080.0;

    let tan_fov_half = (fov * 0.5).tan();
    let asp_fov = aspect_ratio * tan_fov_half;

    let nx = x * asp_fov;
    let ny = -y * tan_fov_half;
    // (forward + nx * right + ny * up).normalize()
    let ray_dir = vec3(nx, ny, -1.0).normalize();

    let cam = Camera::from_look_to(Vec3::ZERO, Vec3::NEG_Z, 45f32.to_radians(), 1.0, 100.0, (1920, 1080));

    let cam_dir = cam.normalized_screen_to_ray(vec2(x, y));

    let dot = ray_dir.dot(cam_dir.dir);
    println!("Dot: {dot:.5}");

    let angle = 90f32.to_radians();
    let s = angle.sin();
    let c = angle.cos();
    let unrot = vec2(0.0, 1.0);
    let rot = vec2(
        unrot.x * c - unrot.y * s,
        unrot.x * s + unrot.y * c,
    );
    let left = vec2(-1.0, 0.0);
    let dot = rot.dot(left);
    
    println!("{rot:?}\n{dot:.4?}");
    println!("################################");
    let rot = vec2(33.0f32.to_radians(), 12.0f32.to_radians());
    let quaty = Quat::from_rotation_y(rot.y);
    let quatx = Quat::from_rotation_x(rot.x);
    let quat = quatx * quaty;
    let quat = Quat::from_euler(EulerRot::YXZ, rot.y, rot.x, 0.0);
    let dir = Vec3::NEG_Z;
    let rot_dir1 = quat * dir;
    let rot_dir2 = rot_dir(dir, vec2(rot.x, rot.y));
    println!("Len1: {:.3}, Len2: {:.3}", rot_dir1.length(), rot_dir2.length());
    println!("Rot Dot: {}", rot_dir1.dot(rot_dir2));
}

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

    pub fn calc_ray_dir(self, ndc: Vec2) -> Vec3 {
        let m = ndc * self.mult;
        vec3(m.x, m.y, -1.0).normalize()
    }
}

macro_rules! timed {
    ($fmt:literal => $code:expr) => {
        let elapsed_time = timeit!{
            $code;
        };
        println!($fmt, time=elapsed_time);
    };
}

pub fn morton_test() {
    let mut chunk = RaytraceChunk::new();
    // let elapsed = timeit!{
        // };
        // println!("Set Morton: {elapsed:.3?}");
    timed!("Set: {time:.3?}" => {
        for x in 0..64 {
            for y in 0..64 {
                for z in 0..64 {
                    chunk.set(x, y, z, x as u32);
                }
            }
        }
    });
    let mut total = 0;
    timed!("Get: {time:.3?}" => {
        for x in 0..64 {
            for y in 0..64 {
                for z in 0..64 {
                    let id = chunk.get(x, y, z);
                    total += id;
                }
            }
        }
    });
    println!("Proof of work: {total}");
}

struct PosColor {
    rsimp: OpenSimplex,
    gsimp: OpenSimplex,
    bsimp: OpenSimplex,
    scale: f64,
}

impl PosColor {
    pub fn get_with_scale(&self, pos: Vec3, scale: f64) -> Vec3 {
        let pos_arr = [pos.x as f64 * scale, pos.y as f64 * scale, pos.z as f64 * scale];
        let r: f64 = self.rsimp.get(pos_arr) * 0.5 + 0.5;
        let g: f64 = self.gsimp.get(pos_arr) * 0.5 + 0.5;
        let b: f64 = self.bsimp.get(pos_arr) * 0.5 + 0.5;
        vec3(r as f32, g as f32, b as f32)
    }

    pub fn get(&self, pos: Vec3) -> Vec3 {
        // Vec3::ONE
        // Vec3::Y

        let pos_arr = [pos.x as f64 * self.scale, pos.y as f64 * self.scale, pos.z as f64 * self.scale];
        let r: f64 = self.rsimp.get(pos_arr) * 0.5 + 0.5;
        let g: f64 = self.gsimp.get(pos_arr) * 0.5 + 0.5;
        let b: f64 = self.bsimp.get(pos_arr) * 0.5 + 0.5;
        vec3(r as f32, g as f32, b as f32)
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
    pub direction: Vec3,
    pub color: Vec3,
    pub intensity: f32,
    pub pre_calc: Vec3,
    pub inv_dir: Vec3,
}

impl DirectionalLight {
    pub fn new(direction: Vec3, color: Vec3, intensity: f32) -> Self {
        Self {
            direction,
            color,
            intensity,
            pre_calc: color * intensity,
            inv_dir: -direction,
        }
    }
    #[inline(always)]
    pub fn apply(&self, color: Vec3, normal: Vec3) -> Vec3 {
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
    pub fn apply(self, color: Vec3) -> Vec3 {
        color * self.factor
    }
}

impl std::ops::Mul<Vec3> for Shadow {
    type Output = Vec3;
    #[inline(always)]
    fn mul(self, rhs: Vec3) -> Self::Output {
        self.apply(rhs)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct AmbientLight {
    pub color: Vec3,
    pub intensity: f32,
}

impl AmbientLight {
    #[inline(always)]
    pub fn apply(self, color: Vec3) -> Vec3 {
        self.color * color
    }
}

impl std::ops::Mul<Vec3> for AmbientLight {
    type Output = Vec3;
    #[inline(always)]
    fn mul(self, rhs: Vec3) -> Self::Output {
        self.apply(rhs)
    }
}

pub struct Lighting {
    directional: DirectionalLight,
    ambient: AmbientLight,
    shadow: Shadow,
}

impl Lighting {
    #[inline(always)]
    pub fn calculate(&self, color: Vec3, normal: Vec3, occluded: bool) -> Vec3 {
        let ambient = self.ambient.color;
        let directional = self.directional.inv_dir.dot(normal);
        let directional_color = self.directional.pre_calc * directional;
        let light = ((1.0 - directional) * self.ambient.intensity) * ambient + directional_color;
        let color = color * light;
        if occluded {
            self.shadow * color
        } else {
            color
        }
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct Reflection {
    reflectivity: f32,
}

#[inline(always)]
fn reflect(ray_dir: Vec3, normal: Vec3) -> Vec3 {
    ray_dir - 2.0 * (ray_dir * normal) * normal
}

#[inline(always)]
fn combine_reflection(reflectivity: f32, diffuse: Vec3, reflection: Vec3) -> Vec3 {
    reflectivity * reflection + (1.0 - reflectivity) * diffuse
}

impl Reflection {
    #[inline(always)]
    pub fn calculate(
        self,
        surface_normal: Vec3,
        view_dir: Vec3,
    ) -> f32 {
        let dot = (-view_dir).dot(surface_normal).max(0.0);
        self.reflectivity + (1.0 - self.reflectivity) * (1.0 - dot).powf(5.0)
    }
}

pub struct TraceData {
    chunk: Chunk,
    pos_color: PosColor,
    lighting: Lighting,
    sky_color: Vec3,
    reflection: Reflection,
    reflection_steps: u16,
}

impl TraceData {
    pub fn calc_color_before_reflections(&self, hit_point: Vec3, hit_normal: Vec3) -> Vec3 {
        let color = self.pos_color.get(hit_point);
        let light_ray = Ray3::new(hit_point, self.lighting.directional.inv_dir);
        let light_hit = self.chunk.raycast(light_ray, 100.0);
        self.lighting.calculate(color, hit_normal, light_hit.is_some())
    }

    pub fn trace_reflections(&self, ray: Ray3, near: f32, far: f32, steps: u16) -> Option<Vec3> {
        let Some(hit) = self.chunk.raycast(ray, far) else {
            return Some(self.sky_color(ray.dir));
        };
        let Some(face) = hit.face else {
            return Some(Vec3::X);
        };
        if hit.distance < near {
            return Some(Vec3::Y);
        }
        let hit_point = hit.get_hit_point(ray, face);
        let hit_normal = face.normal();
        let mut diffuse = self.calc_color_before_reflections(hit_point, hit_normal);
        fn on_edge(x: f32, y: f32) -> bool {
            x < 0.05 || y < 0.05 || x >= 0.95 || y >= 0.95
        }
        let hit_fract = hit_point.fract();
        match face {
            Face::PosX | Face::NegX if on_edge(hit_fract.y, hit_fract.z) => diffuse = diffuse * 0.1,
            Face::PosY | Face::NegY if on_edge(hit_fract.x, hit_fract.z) => diffuse = diffuse * 0.1,
            Face::PosZ | Face::NegZ if on_edge(hit_fract.x, hit_fract.y) => diffuse = diffuse * 0.1,
            _ => (),
        }
        if steps == 1 {
            return Some(diffuse);
        }
        let reflect_dir = reflect(ray.dir, hit_normal);
        let reflect_ray = Ray3::new(hit_point, reflect_dir);
        let reflectivity = self.reflection.calculate(hit_normal, ray.dir);
        let Some(reflection) = self.trace_reflections(reflect_ray, 0.0, far - hit.distance, steps - 1) else {
            // return Some(combine_reflection(reflectivity, diffuse, self.sky_color(reflect_dir)));
            return Some(diffuse);
        };
        let final_color = combine_reflection(reflectivity, diffuse, reflection);
        Some(final_color)
    }

    pub fn trace_color(&self, ray: Ray3, near: f32, far: f32) -> Vec3 {
        if let Some(color) = self.trace_reflections(ray, near, far, self.reflection_steps) {
            color
        } else {
            self.sky_color(ray.dir)
        }
    }

    pub fn sky_color(&self, ray_dir: Vec3) -> Vec3 {
        self.lighting.ambient.apply(self.pos_color.get(ray_dir) * self.sky_color)
    }
}

pub fn raycast_scene() {
    use glam::*;
    use scratch::math::*;
    use scratch::camera::*;
    use scratch::perlin::perlin;
    use rayon::prelude::*;
    use noise::OpenSimplex;
    const SSAA: bool = true;
    println!("Starting.");
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
    let perm = Permutation::from_seed(make_seed(SEED as u64));
    // let size = GridSize::new(1280, 720);
    // let size = GridSize::new(1920, 1080);
    let size = GridSize::new(1920*2, 1080*2);
    // let size = GridSize::new(1920*4, 1080*4);
    // let size = GridSize::new(2048, 2048);
    let size = if SSAA {
        GridSize::new(size.width * 2, size.height * 2)
    } else {
        size
    };
    // let same = 1024*16;
    // let size = GridSize::new(same, same/2);
    let mut cam = Camera::from_look_at(vec3(-24.0, 70.0-12.0, 48.0), vec3(32., 32.-12., 32.), 45.0f32.to_radians(), 1.0, 100.0, (size.width, size.height));
    // let mut cam = Camera::from_look_at(vec3(-24.0, 70.0-12.0, 64.0+24.0), vec3(32., 32.-12., 32.), 90.0f32.to_radians(), 1.0, 100.0, (size.width, size.height));
    // let mut cam = Camera::from_look_at(vec3(24.0, 24.0, 16.0), vec3(32.0, 8.0, 32.0), 90.0f32.to_radians(), 1.0, 100.0, (size.width, size.height));

    let mut last = IVec3::ZERO;
    let mut chunk = Chunk::new();
    let mut trace = TraceData {
        chunk,
        pos_color,
        lighting: Lighting {
            directional: DirectionalLight::new(
                vec3(0.5, -1.0, -1.0).normalize(),
                vec3(1.0, 1.0, 1.0),
                0.5,
            ),
            ambient: AmbientLight {
                color: vec3(1.0, 1.0, 1.0),
                intensity: 1.0,
            },
            shadow: Shadow {
                factor: 0.2,
            },
        },
        // sky_color: Vec3::splat(0.0),
        sky_color: Vec3::splat(1.0),
        reflection: Reflection { reflectivity: 0.5 },
        reflection_steps: 5,
    };
    fn checkerboard(x: i32, y: i32, z: i32) -> bool {
        ((x & 1) ^ (y & 1) ^ (z & 1)) != 0
    }
    let start = Instant::now();
    let ray_calc = RayCalc::new(cam.fov, cam.screen_size);
    let cam_pos = cam.position;
    let cam_ref = &cam;
    let cam_rot = cam.rotation_matrix();
    // let cam_rot = cam.rotation;
    let rays = (0..size.width*size.height).into_par_iter().map(move |i| {
        let (x, y) = size.inv_index(i);
        let xy = vec2(x as f32, (size.height - y) as f32);
        let wh = vec2(size.width as f32, size.height as f32);
        let hs = vec2(0.5, 0.5);
        let screen_pos = (xy / wh) - hs;
        // let screen_pos = vec2(x as f32 / size.width as f32 - 0.5, y as f32 / size.height as f32 - 0.5);
        // cam_ref.normalized_screen_to_ray(screen_pos).dir
        let dir = ray_calc.calc_ray_dir(screen_pos);
        cam_rot * dir
        // rot_dir(dir, cam_rot)
    }).collect::<Vec<_>>();
    let elapsed = start.elapsed();
    println!("Calculated {} rays in {elapsed:.3?}", rays.len());

    // for x in 0..64 {
    //     for z in 0..64 {
    //         for y in 0..64 {
    //             chunk.set(x, y, z, checkerboard(x, y, z));
    //         }
    //     }
    // }

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
    // for x in 0i32..64 {
    //     for y in 0i32..64 {
    //         for z in 0i32..64 {
    //             let ox = (32 - x).abs();
    //             let oy = (32 - y).abs();
    //             let oz = (32 - z).abs();
    //             let ods = ((ox + 7) * (ox + 7)) + ((oy + 7) * (oy + 7)) + oz * oz;
    //             if ods <= DS {
    //                 chunk.set(x as u32, y as u32, z as u32, true);
    //             }
    //             // let mut c = 0;
    //             // if x % 8 == 0 {
    //             //     c += 1;
    //             // }
    //             // if y % 8 == 0 {
    //             //     c += 1;
    //             // }
    //             // if z % 8 == 0 {
    //             //     c += 1;
    //             // }
    //             // if c > 1 {
    //             //     chunk.set(x, y, z, true);
    //             // }
    //         }
    //     }
    // }

    // for x in 0..64 {
    //     for y in 0..64 {
    //         for z in 0..64 {
    //             chunk.set(x, y, z, true);
    //         }
    //     }
    // }
    // let line_width = 4;
    // chunk.fill_box((16, 16, 16), (48, 48, 48), true);
    // chunk.fill_box((16, 16 + line_width, 16 + line_width), (48, 48 - line_width, 48 - line_width), false);
    // chunk.fill_box((16 + line_width, 16, 16 + line_width), (48 - line_width, 48, 48 - line_width), false);
    // chunk.fill_box((16 + line_width, 16 + line_width, 16), (48 - line_width, 48 - line_width, 48), false);

    // chunk.draw_box((16, 16, 16), (48, 48, 48), true);
    // chunk.draw_box((20, 20, 20), (44, 44, 44), true);
    // chunk.draw_box((24, 24, 24), (40, 40, 40), true);
    // chunk.draw_box((28, 28, 28), (36, 36, 36), true);
    // 10202010102020101020201
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
                if false
                || true
                {
                    let end = (r.end.0, r.start.1 + 1, r.end.2);
                    chunk.fill_box(r.start, end, true);
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
                    boxes((x * 10, y * 10, z * 10), &mut trace.chunk);
                }
            }
        }
        let elapsed = start.elapsed();
        println!("Placed blocks in {elapsed:.3?}");
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
    // let depth_mul = 1.0 / (far - near);
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
        let size = GridSize::new(size.width/2, size.height/2);
        println!("SSAA downscale to {size}");
        let img = DynamicImage::ImageRgb8(img);
        let resized = img.resize_exact(size.width, size.height, imageops::FilterType::Gaussian);
        resized.save("raycast.png").expect("Failed to save image.");
    } else {
        img.save("raycast.png").expect("Failed to save image.");
    }
}

macro_rules! grave {($($_:tt)*) => {};}

grave!{
    fn in_bounds(x: i32, y: i32) -> bool {
        x >= 0 && y >= 0 && x < 1024 && y < 1024
    }

    fn pix(step_count: i32, level: i32) -> Rgb<u8> {
        let level = level as f32;
        let t = level / step_count as f32;
        let gray = (255.0 * t) as u8;
        Rgb([gray; 3])
    }

    struct Algo {
        rng: StdRng,
        plot: DiffPlot,
        img: RgbImage,
        step_count: i32,
        queue: VecDeque<(i32, i32, i32)>,
    }

    let mut algo = Algo {
        rng: StdRng::from_seed(make_seed(13512512)),
        plot: DiffPlot::new(1024, 1024),
        img: RgbImage::new(1024, 1024),
        step_count: 20,
        queue: VecDeque::from([(512, 512, 20)]),
    };

    const DISP_R: &[i32] = &[0, 1, 3, 5, 7, 11, -1, -3, -5, -7, -11];

    fn algo_step(algo: &mut Algo, x: i32, y: i32, step: i32) {
        if !in_bounds(x, y) {
            return;
        }
        let disp_r: usize = algo.rng.random_range(0..DISP_R.len());
        let disp_r = DISP_R[disp_r];
        plot_circle(x, y, 12 + disp_r, |x, y| {
            if !in_bounds(x, y) {
                return;
            }
            if algo.plot.set(x as u32, y as u32, true) {
                return;
            }
            algo.img.put_pixel(x as u32, y as u32, pix(algo.step_count, step));
            if step != 0 {
                algo.queue.push_back((x, y, step - 1));
            }
        });
        fill_circle(x, y, 12 + disp_r, |x, y| {
            if !in_bounds(x, y) {
                return;
            }
            if algo.plot.set(x as u32, y as u32, true) {
                return;
            }
            algo.img.put_pixel(x as u32, y as u32, pix(algo.step_count, step));
        });
    }
    while let Some((x, y, step)) = algo.queue.pop_front() {
        println!("{step} {:?}", pix(10, step));
        algo_step(&mut algo, x, y, step);
    }

    // fill_circle(128, 128, 16, |x, y| {
    //     if x < 0 || y < 0 || x >= 256 || y >= 256 {
    //         return;
    //     }
    //     let x = x as u32;
    //     let y = y as u32;
    //     // img.put_pixel(x, y, Rgb([255, 0, 0]));
    // });
    // plot_circle(128, 128, 16, |x, y| {
    //     if x < 0 || y < 0 || x >= 256 || y >= 256 {
    //         return;
    //     }
    //     let x = x as u32;
    //     let y = y as u32;
    //     // img.put_pixel(x, y, Rgb([0, 255, 0]));
    // });
    algo.img.save("output.png").expect("Failed to save image.");
}

grave!{
    fn branch_experiment() {
        pub struct Branch<I, F, T, R> {
            false_: F,
            true_: T,
            _phantom: std::marker::PhantomData<(I, R)>
        }
        
        impl<I, F, T, R> Branch<I, F, T, R>
        where
            F: FnMut(I) -> R,
            T: FnMut(I) -> R,
        {
            pub const fn new(false_: F, true_: T) -> Self {
                Self {
                    false_,
                    true_,
                    _phantom: std::marker::PhantomData,
                }
            }
        
            pub fn branch(&mut self, branch: bool, input: I) -> R {
                match branch 
                {
                    false => (self.false_)(input),
                    true => (self.true_)(input),
                }
            }
        }
        
        let mut branch = Branch::new(
            |i: i32| i - 1,
            |i: i32| i + 1,
        );
        println!("{}", branch.branch(false, 3));
        println!("{}", branch.branch(true, 3));
    }
}

macro_rules! prototype { ($($_:tt)*) => {} }

prototype!(
    struct Foo {
        name: &'static str,
    }

    impl Foo {
        pub fn new(name: &'static str) -> Self {
            Self {
                name,
            }
        }
        
        pub macro bar(self) {
            () => {};
            ($($name:ident),*) => {
                $(
                    self.$name();
                )*
            };
        }
        
        fn fnord(&self) {
            println!("fnord({})", self.name);
        }
        
        fn fred(&self) {
            println!("fred({})", self.name);
        }
        
        fn baz(&self) {
            println!("baz({})", self.name);
        }
    }
    
    
    
    fn main() {
        let foo = Foo::new("Doobie");
        foo.bar(fnord, fred, baz);
    }
);
