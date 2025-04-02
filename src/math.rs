use core::f32;

use glam::*;

use crate::ray::Ray3;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Axis {
    X,
    Y,
    Z,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Face {
    PosX,
    PosY,
    PosZ,
    NegX,
    NegY,
    NegZ,
}

impl Face {
    pub fn axis(self) -> Axis {
        match self {
            Face::PosX | Face::NegX => Axis::X,
            Face::PosY | Face::NegY => Axis::Y,
            Face::PosZ | Face::NegZ => Axis::Z,
        }
    }

    pub fn normal(self) -> Vec3 {
        match self {
            Face::PosX => Vec3::X,
            Face::PosY => Vec3::Y,
            Face::PosZ => Vec3::Z,
            Face::NegX => Vec3::NEG_X,
            Face::NegY => Vec3::NEG_Y,
            Face::NegZ => Vec3::NEG_Z,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Sign {
    Neg = -1,
    Zero = 0,
    Pos = 1,
}

#[inline]
pub const fn scrunch(value: f32) -> f32 {
    1.0 / value
}

#[inline]
pub const fn scrunch_factor(cell_size: Vec3) -> Vec3 {
    vec3(
        scrunch(cell_size.x),
        scrunch(cell_size.y),
        scrunch(cell_size.z),
    )
}

pub fn scrunch_vector(input: Vec3, cell_size: Vec3) -> Vec3 {
    // So the idea is that you have a cell_size, and you have a direction/position, and you want
    // to emulate a cell_size of (1.0, 1.0, 1.0), so you "scrunch" the direction/position
    // So that it's the equivalent of a cell_size of (1.0, 1.0, 1.0).
    input * scrunch_factor(cell_size)
}

pub fn next_face(point: Vec3, direction: Vec3, cell_size: Vec3, cell_offset: Vec3) -> (Face, f32) {
    let offset_point = point - cell_offset;
    let cell_offset = vec3(
        offset_point.x.rem_euclid(cell_size.x),
        offset_point.y.rem_euclid(cell_size.y),
        offset_point.z.rem_euclid(cell_size.z),
    );
    let x_distance = if direction.x > 0.0 {
        let grid_remainder = cell_size.x - cell_offset.x;
        grid_remainder / direction.x
    } else if direction.x < 0.0 {
        if cell_offset.x == 0.0 {
            cell_size.x / -direction.x
        } else {
            cell_offset.x / -direction.x
        }
    } else {
        f32::INFINITY
    };
    let y_distance = if direction.y > 0.0 {
        let grid_remainder = cell_size.y - cell_offset.y;
        grid_remainder / direction.y
    } else if direction.y < 0.0 {
        if cell_offset.y == 0.0 {
            cell_size.y / -direction.y
        } else {
            cell_offset.y / -direction.y
        }
    } else {
        f32::INFINITY
    };
    let z_distance = if direction.z > 0.0 {
        let grid_remainder = cell_size.z - cell_offset.z;
        grid_remainder / direction.z
    } else if direction.z < 0.0 {
        if cell_offset.z == 0.0 {
            cell_size.z / -direction.z
        } else {
            cell_offset.z / -direction.z
        }
    } else {
        f32::INFINITY
    };

    if x_distance < y_distance {
        if x_distance < z_distance {
            (if direction.x > 0.0 { Face::PosX } else { Face::NegX }, x_distance)
        } else {
            (if direction.z > 0.0 { Face::PosZ } else { Face::NegZ }, z_distance)
        }
    } else {
        if y_distance < z_distance {
            (if direction.y > 0.0 { Face::PosY } else { Face::NegY }, y_distance)
        } else {
            (if direction.z > 0.0 { Face::PosZ } else { Face::NegZ }, z_distance)
        }
    }
}

// pub fn raycast<F: FnMut(Face, Vec3)>(point: Vec3, direction: Vec3, cell_size: Vec3, cell_offset: Vec3, step_count: usize, mut f: F) {
//     let mut dist_accum = 0.0;
//     let mut steps = 0;
//     let mut p = point;
//     while steps < step_count {
//         let (axis, dist) = next_face(p, direction, cell_size, cell_offset);
//         dist_accum += dist;
//         p = point + direction * dist_accum;
//         f(axis, p);
//         steps += 1;
//     }
// }

pub fn raycast<F: FnMut(&IVec3, Face, f32) -> bool>(
    ray: Ray3,
    cell_size: Vec3,
    cell_offset: Vec3,
    mut callback: F,
) {
    fn calc_step(cell_size: f32, magnitude: f32) -> f32 {
        cell_size / magnitude.abs().max(<f32>::MIN_POSITIVE)
    }
    let delta = vec3(
        calc_step(cell_size.x, ray.dir.x),
        calc_step(cell_size.y, ray.dir.y),
        calc_step(cell_size.z, ray.dir.z),
    );

    let sign = ray.dir.signum();
    let step = sign.as_ivec3();
    let face = (
        if step.x >= 0 {
            Face::NegX
        } else {
            Face::PosX
        },
        if step.y >= 0 {
            Face::NegY
        } else {
            Face::PosY
        },
        if step.z >= 0 {
            Face::NegZ
        } else {
            Face::PosZ
        },
    );

    let origin = ray.pos - cell_offset;
    let inner = origin.rem_euclid(cell_size);

    fn calc_t_max(step: i32, cell_size: f32, p: f32, magnitude: f32) -> f32 {
        if step > 0 {
            (cell_size - p) / magnitude.abs().max(<f32>::MIN_POSITIVE)
        } else if step < 0 {
            p / magnitude.abs().max(<f32>::MIN_POSITIVE)
        } else {
            f32::INFINITY
        }
    }
    let mut t_max = vec3(
        calc_t_max(step.x, cell_size.x, inner.x, ray.dir.x),
        calc_t_max(step.y, cell_size.y, inner.y, ray.dir.y),
        calc_t_max(step.z, cell_size.z, inner.z, ray.dir.z),
    );
 
    let mut cell = (origin / cell_size).floor().as_ivec3();
    callback(&cell, Face::PosY, 0.0);
    loop {
        if t_max.x <= t_max.y {
            if t_max.x <= t_max.z {
                cell.x += step.x;
                if callback(&cell, face.0, t_max.x) {
                    return;
                }
                t_max.x += delta.x;
            } else {
                cell.z += step.z;
                if callback(&cell, face.2, t_max.z) {
                    return;
                }
                t_max.z += delta.z;
            }
        } else {
            if t_max.y <= t_max.z {
                cell.y += step.y;
                if callback(&cell, face.1, t_max.y) {
                    return;
                }
                t_max.y += delta.y;
            } else {
                cell.z += step.z;
                if callback(&cell, face.2, t_max.z) {
                    return;
                }
                t_max.z += delta.z;
            }
        }
    }
}

// fn part1by1(n: u32) -> u32 {
//     var n = n & 0x0000ffff;
//     n = (n | (n << 8)) & 0x00ff00ff;
//     n = (n | (n << 4)) & 0x0f0f0f0f;
//     n = (n | (n << 2)) & 0x33333333;
//     n = (n | (n << 1)) & 0x55555555;
//     return n;
// }

// fn morton_encode(x: u32, y: u32) -> u32 {
//     return (part1by1(y) << 1) | part1by1(x);
// }

// fn chunk_index_flat(x: usize, y: usize, z: usize) -> usize {
//     x | (y << 4) | (z << 8)
// }

// Morton index for 2048x2048

// fn chunk_index_morton(x: usize, y: usize, z: usize) -> usize {
//     #[inline(always)]
//     fn shutter(n: usize) -> usize {
//         //       0b1111
//         //       0b1010101
//         //       0b001001001001
//         let step1 = (n | (n << 4)) &    0b000011000011000011000011000011000011;
//         return (step1 | (step1 << 2)) & 0b001001001001001001001001001001001001;
//     }
//     shutter(x) | (shutter(y) << 1) | (shutter(z) << 2)
// }

// fn morton2_2048(x: u32, y: u32) -> u32 {
//     #[inline(always)]
//     fn shutter(n: u32) -> u32 {
//         // 0b    11  111  111  111
//         // 0b101010101010101010101
//         let step1 = (n | (n << 10)) & 0b0000000000011111111111;
//     }
// }

// fn chunk_index_morton(x: usize, y: usize, z: usize) -> usize {
//     #[inline(always)]
//     fn shutter(n: usize) -> usize {
//         // 0b1111100000000000111111
//         // 0b1100000011100000000000111000111
//         // 0b1100000011100000000000111000000111
//         let step1 = (n | (n << 11)) & 0b1111100000000000111111;
//         let step2 = (n | (n << 6)) & 0b1100000011100000000000111000000111;
//         (step1 | (step1 << 3)) & 0b001001001001001001001001001001001001001001001001001001001001001
//     }
//     shutter(x) | (shutter(y) << 1) | (shutter(z) << 2)
// }

// #[test]
// fn morton_test() {
//     let morton = chunk_index_morton(8, 31, 31);
//     println!("Morton: {morton}");
// }

#[test]
fn to_i32_test() {
    let v = f32::NEG_INFINITY;
    println!("{}", v.signum() as i32);
}

#[test]
fn scrunch_test() {
    let size = vec3(3.0, 2.0, 1.0);
    let p = vec3(1.0, 1.0, 1.0);
    let s = scrunch_vector(p, size);
    println!("{s}");
    println!("{}", s * size);
}

// #[test]
// fn raycast_test() {
//     let point = vec3(0.5, 0.5, 0.5);
//     let direction = vec3(1.0, 0.0, 0.0).normalize();
//     let cell_size = Vec3::ONE;
//     let cell_offset = Vec3::ZERO;
//     // let mut counter = 0usize;
//     let start = std::time::Instant::now();
//     // raycast(Ray3::new(point, direction), cell_size, cell_offset, |p, d| {
//     //     println!("{p:?}, {d}");
//     //     let loc = point + direction * d;
//     //     println!("Location: {loc:?}");
//     //     d < 100.0
//     // });
//     // for _ in 0..256 {
//     //     for _ in 0..256 {
//     //         raycast(point, direction, cell_size, cell_offset, 10, |_, _, _| {
//     //             // println!("{p} {i}");
//     //             counter += 1;
//     //             true
//     //         });
//     //     }
//     // }
//     println!("Elapsed: {:?}", start.elapsed());
//     // println!("Counter: {counter}");
// }