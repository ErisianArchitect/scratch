#![allow(unused)]

use super::perlin::{Permutation, lerp, make_seed};

#[rustfmt::skip]
const GRADIENTS_2D: [(f32, f32); 8] = [
    (5.0, 2.0), (2.0, 5.0), (-5.0, 2.0), (-2.0, 5.0),
    (5.0, -2.0), (2.0, -5.0), (-5.0, -2.0), (-2.0, -5.0),
];

const SQRT3: f32 = 1.73205080757;
const F2: f32 = 0.5 * (SQRT3 - 1.0);
const G2: f32 = (3.0 - SQRT3) / 6.0;

pub fn open_simplex_2d(permutation: &Permutation, x: f32, y: f32) -> f32 {
    let s = (x + y) * F2;
    let xs = x + s;
    let ys = y + s;

    let i = xs.floor() as isize;
    let j = ys.floor() as isize;

    // Unskew to get original coordinates
    let t = (i + j) as f32 * G2;
    let x0 = x - (i as f32 - t);
    let y0 = y - (j as f32 - t);

    // Determine simplex corner order
    let (i1, j1) = if x0 > y0 { (1, 0) } else { (0, 1) };

    // Second corner coordinates
    let x1 = x0 - i1 as f32 + G2;
    let y1 = y0 - j1 as f32 + G2;

    // Third corner coordinates
    let x2 = x0 - 1.0 + 2.0 * G2;
    let y2 = y0 - 1.0 + 2.0 * G2;

    // Hash gradient indices
    let ii = (i & 255) as usize;
    let jj = (j & 255) as usize;
    let gi0 = permutation.get(ii + permutation.get(jj)) & 7;
    let gi1 = permutation.get(ii + i1 + permutation.get(jj + j1)) & 7;
    let gi2 = permutation.get(ii + 1 + permutation.get(jj + 1)) & 7;

    // Computate dot products
    let mut n0 = 0.0;
    let mut n1 = 0.0;
    let mut n2 = 0.0;

    let t0 = 0.5 - x0 * x0 - y0 * y0;
    if t0 > 0.0 {
        let g = GRADIENTS_2D[gi0];
        n0 = (t0 * t0) * (t0 * t0) * (g.0 * x0 + g.1 * y0);
    }

    let t1 = 0.5 - x1 * x1 - y1 * y1;
    if t1 > 0.0 {
        let g = GRADIENTS_2D[gi1];
        n1 = (t1 * t1) * (t1 * t1) * (g.0 * x1 + g.1 * y1);
    }

    let t2 = 0.5 - x2 * x2 - y2 * y2;
    if t2 > 0.0 {
        let g = GRADIENTS_2D[gi2];
        n2 = (t2 * t2) * (t2 * t2) * (g.0 * x2 + g.1 * y2);
    }

    // Sum contributions and scale
    // 18.541 was the value I found to scale the best. It's not perfect, but there is no perfect value. It's close enough.
    18.541 * (n0 + n1 + n2)
}

#[cfg(test)]
mod tests {
    use image::Rgba;

    use super::*;
    #[test]
    fn open_simplex_test() {
        let permutation = Permutation::from_seed(make_seed(09512));

        let mut img = image::RgbaImage::new(512, 512);
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        for y in 0..512 {
            for x in 0..512 {
                // let (xf, yf) = make_uv(x, y);
                // let noise = perlin(&perm, xf * 10.0, yf * 10.0);
                let noise = open_simplex_2d(&permutation, (x as f32 + 0.0) * 0.01, (y as f32 + 0.0) * 0.01);
                if noise < min {
                    min = noise;
                }
                if noise > max {
                    max = noise;
                }
                let norm = ((noise + 1.0) / 2.0).clamp(0.0, 1.0);
                let gray = (norm * 255.0) as u8;
                img.put_pixel(x, y, Rgba([gray, gray, gray, 255]));
            }
        }
        println!("Min: {min:.7}\nMax: {max:.7}");
        img.save("noise.png").expect("Failed to save image.");
    }
}