#![allow(unused)]
use rand::prelude::*;
use rand::seq::SliceRandom;
use sha2::Digest;
use sha2::Sha256;

pub fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

pub fn make_seed(value: u64) -> [u8; 32] {
    use std::io::Write;
    let mut hasher = Sha256::default();
    hasher.write_all(&value.to_le_bytes()).expect("Failed to write.");
    hasher.finalize().into()
}

pub struct Permutation {
    // Duplicated for cache locality when overflow happens. Removes the need for bitwise instructions.
    array: Box<[u8; 512]>
}

impl Permutation {
    pub fn get(&self, index: usize) -> usize {
        self.array[index] as usize
    }

    pub fn from_seed(seed: [u8; 32]) -> Self {
        let mut rng = StdRng::from_seed(seed);
        let mut array: [u8; 256] = std::array::from_fn(|i| i as u8);
        array.shuffle(&mut rng);
        let array: Box<[u8; 512]> = Box::new(std::array::from_fn(move |i| array[i & 255]));
        Self {
            array
        }

    }
}

fn fade(t: f32) -> f32 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

fn grad(hash: usize, x: f32, y: f32) -> f32 {
    let h = hash & 7;
    let u = if h < 4 { x } else { y };
    let v = if h < 4 { y } else { x };
    (if (h & 1) == 0 { u } else { -u }) + (if (h & 2) == 0 { v } else { -v })
}

fn perlin(permutation: &Permutation, x: f32, y: f32) -> f32 {
    let x0 = x.floor() as usize & 255;
    let y0 = y.floor() as usize & 255;
    let x1 = x0 + 1;
    let y1 = y0 + 1;

    let xf = x.fract();
    let yf = y.fract();

    let u = fade(xf);
    let v = fade(yf);

    let a = permutation.get(x0) + y0;
    let b = permutation.get(x1) + y0;
    let c = permutation.get(x0) + y1;
    let d = permutation.get(x1) + y1;

    let x1_interp = lerp(grad(permutation.get(a), xf, yf), grad(permutation.get(b), xf - 1.0, yf), u);
    let x2_interp = lerp(grad(permutation.get(c), xf, yf - 1.0), grad(permutation.get(d), xf - 1.0, yf - 1.0), u);

    lerp(x1_interp, x2_interp, v)
}

fn make_uv(x: u32, y: u32) -> (f32, f32) {
    (x as f32 / 511.0, y as f32 / 511.0)
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use image::Rgba;

    use super::*;
    #[test]
    fn perlin_test() {
        let perm = Permutation::from_seed(make_seed(1531096152109));
        let mut img = image::RgbaImage::new(512, 512);

        for y in 0..512 {
            for x in 0..512 {
                // let (xf, yf) = make_uv(x, y);
                // let noise = perlin(&perm, xf * 10.0, yf * 10.0);
                let noise = perlin(&perm, (x as f32 + 0.1) * 0.03, (y as f32 + 0.1) * 0.03);
                let norm = (noise + 1.0) / 2.0;
                let gray = (norm * 255.0) as u8;
                img.put_pixel(x, y, Rgba([gray, gray, gray, 255]));
            }
        }

        img.save("noise.png").expect("Failed to save image.");
    }
}