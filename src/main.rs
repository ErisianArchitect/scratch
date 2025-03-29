#![allow(unused)]
use std::{borrow::Borrow, collections::{HashMap, VecDeque}, sync::{Arc, Mutex}, time::{Duration, Instant}};

use image::{Rgb, RgbImage, Rgba, RgbaImage};
use itertools::Itertools;
use rand::{rngs::StdRng, Rng, SeedableRng};
use scratch::{dag::*, diff_plot::DiffPlot, perlin::make_seed};
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
    let point = vec3(0.5, 0.5, 0.5);
    let direction = vec3(1.0, 2.0, 0.0).normalize();
    let cell_size = Vec3::ONE;
    let cell_offset = Vec3::ZERO;
    let mut counter = 0usize;
    let start = std::time::Instant::now();
    let mut last = IVec3::ZERO;
    let mut last_i = 0;
    let mut last_d = 0.0;
    for _ in 0..(640*360) {
        #[no_mangle]
        raycast(point, direction, cell_size, cell_offset, 100, |p, i, d| {
            // println!("{p} {i}");
            counter += 1;
            last = std::hint::black_box(p);
            last_i = std::hint::black_box(i);
            last_d = std::hint::black_box(d);
            true
        });
    }
    println!("Elapsed: {:?}", start.elapsed());
    println!("Last: {last:?}");
    println!("Last Index: {last_i}");
    println!("Hit Point: {}", point + direction * last_d);
    println!("Counter: {counter}");
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
