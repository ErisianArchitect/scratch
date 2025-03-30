#![allow(unused)]
use std::{borrow::Borrow, collections::{HashMap, VecDeque}, sync::{Arc, Mutex}, time::{Duration, Instant}};

use image::{Rgb, RgbImage, Rgba, RgbaImage};
use itertools::Itertools;
use rand::{rngs::StdRng, Rng, SeedableRng};
use scratch::{chunk::Chunk, dag::*, diff_plot::DiffPlot, open_simplex::open_simplex_2d, perlin::{make_seed, Permutation}, ray::Ray3};
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

fn main() {
    use glam::*;
    use scratch::math::*;
    use scratch::camera::*;
    use scratch::perlin::perlin;
    let perm = Permutation::from_seed(make_seed(1243));
    let img_size = (1920*2, 1080*2);
    // let img_size = (256, 256);
    let mut cam = Camera::from_look_at(vec3(-48.0, 64.0, -64.0), vec3(32., 8., 32.), 45.0, 1.0, 100.0, img_size);

    let mut last = IVec3::ZERO;
    let mut chunk = Chunk::new();
    let line_width = 8;
    fn checkerboard(x: u32, y: u32, z: u32) -> bool {
        ((x & 1) ^ (y & 1) ^ (z & 1)) != 0
    }

    // for x in 0..64 {
    //     for z in 0..64 {
    //         for y in 0..64 {
    //             chunk.set(x, y, z, checkerboard(x, y, z));
    //         }
    //     }
    // }

    for x in 0..64 {
        for z in 0..64 {
            let min_dist = x.min(z).min(63-x).min(63-z);
            let falloff = (min_dist as f32) / 32.0;
            let height = (perlin(&perm, x as f32 / 64.0, z as f32 / 64.0) * 0.5) + 0.5;
            let h = ((height * 64.0 * falloff) as u32).max(1);
            chunk.fill_box((x, 0, z), (x+1, h, z+1), true);
        }
    }

    // chunk.fill_box((16, 16, 16), (48, 48, 48), true);
    // chunk.fill_box((16, 16 + line_width, 16 + line_width), (48, 48 - line_width, 48 - line_width), false);
    // chunk.fill_box((16 + line_width, 16, 16 + line_width), (48 - line_width, 48, 48 - line_width), false);
    // chunk.fill_box((16 + line_width, 16 + line_width, 16), (48 - line_width, 48 - line_width, 48), false);

    for x in 0..64 {
        for z in 0..64 {
            for y in 0..64 {
                let b = chunk.get(x, y, z);
                let cb1 = checkerboard(x, y, z);
                let cb2 = checkerboard(x/2, y/2, z/2) || true;
                let cb3 = checkerboard(x/4, y/4, z/4) || true;
                chunk.set(x, y, z, b && cb1 && cb2 && cb3);
            }
        }
    }


    // let mut plot = DiffPlot::new(1024, 1024);
    
    let mut img = RgbImage::new(img_size.0, img_size.1);

    let near = 0.1;
    let far = 500.0;
    let depth_mul = 1.0 / (far - near);
    let start = std::time::Instant::now();
    for y in 0..img_size.1 {
        for x in 0..img_size.0 {
    // for y in 2048..2049 {
    //     for x in 2048..2049 {
            let screen_pos = vec2(x as f32 / img_size.0 as f32 - 0.5, y as f32 / img_size.1 as f32 - 0.5);
            let ray = cam.normalized_screen_to_ray(screen_pos);
            {
                let last = &mut last;
                let img = &mut img;
                let chunk = &chunk;
                let mut steps = 0;
                let max_dist = far;
                if let Some(hit) = chunk.raycast(ray, max_dist) {
                    if hit.distance >= near && hit.distance < far {
                        let dnorm = (hit.distance - near) * depth_mul;
                        let pix = (dnorm * 255.0) as u8;
                        let rgb = match hit.face.map(|face| face.axis()) {
                            Some(Axis::X) => Rgb([pix, 0, 0]),
                            Some(Axis::Y) => Rgb([0, pix, 0]),
                            Some(Axis::Z) => Rgb([0, 0, pix]),
                            _ => Rgb([255, 255, 255]),
                        };
                        img.put_pixel(x, y, rgb);
                    }
                }
                // raycast(ray, Vec3::ONE, Vec3::ONE, move |p, f, d| {
                //     // steps += 1;
                //     // last.clone_from(&p);
                //     if p.x < 0 || p.y < 0 || p.z < 0
                //     || p.x >= 64 || p.y >= 64 || p.z >= 64 {
                //         return d > max_dist;
                //     }
                //     if chunk.get(p.x as u32, p.y as u32, p.z as u32) {
                //         // img.set(x, y, true);
                //         if d >= near && d < far {
                //             let dnorm = (d - near) * depth_mul;
                //             let pix = (dnorm * 255.0) as u8;
                //             let rgb = match f.axis() {
                //                 Axis::X => Rgb([pix, 0, 0]),
                //                 Axis::Y => Rgb([0, pix, 0]),
                //                 Axis::Z => Rgb([0, 0, pix]),
                //             };
                //             img.put_pixel(x, y, rgb);
                //             return true;
                //         }
                //         return true;
                //     }
                //     d > max_dist
                // });
            }
        }
    }
    let elapsed = start.elapsed();
    println!("Completed in {elapsed:?}");

    // for y in 0..1024 {
    //     for x in 0..1024 {
    //         if plot.get(x, y) {
    //             img.put_pixel(x, y, Rgb([255, 255, 255]));
    //         } else {
    //             img.put_pixel(x, y, Rgb([0, 0, 0]));
    //         }
    //     }
    // }

    img.save("raycast.png").expect("Failed to save image.");

    println!("Last hit point: {}", last);

    return;
    let point = vec3(0.5, 0.5, 0.5);
    let direction = vec3(14.0, 2.0, 1.0).normalize();
    let cell_size = Vec3::ONE;
    let cell_offset = Vec3::ZERO;
    let mut counter = 0usize;
    let mut total = 0usize;
    let start = std::time::Instant::now();
    let mut last = IVec3::ZERO;
    let mut last_d = 0.0;
    for _ in 0..(16*16) {
        counter = 0;
        {
            let counter = &mut counter;
            let last = &mut last;
            let last_d = &mut last_d;
            #[no_mangle]
            raycast(Ray3::new(point, direction), cell_size, cell_offset, move |p, f, d| {
                last.clone_from(p);
                *last_d = d;
                *counter += 1;
                // d < 1000.0
                *counter >= 1000
            });
        }
        total += counter;
    }
    println!("Elapsed: {:?}", start.elapsed());
    println!("Last: {last:?}");
    println!("Hit Point: {}", point + direction * last_d);
    println!("Counter: {total}");
    return;
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
    return;
    let start = Instant::now();
    let dur = Duration::from_secs(1);
    let end = start + dur;
}

macro_rules! grave {($($_:tt)*) => {};}

grave!{

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
