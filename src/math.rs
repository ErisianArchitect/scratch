use core::f32;

use glam::*;

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

pub fn raycast<F: FnMut(&IVec3, f32) -> bool>(
    ray_origin: Vec3,
    ray_direction: Vec3,
    cell_size: Vec3,
    cell_offset: Vec3,
    mut callback: F,
) {
    fn calc_step(cell_size: f32, magnitude: f32) -> f32 {
        cell_size / magnitude.abs().max(<f32>::MIN_POSITIVE)
    }
    let delta = vec3(
        calc_step(cell_size.x, ray_direction.x),
        calc_step(cell_size.y, ray_direction.y),
        calc_step(cell_size.z, ray_direction.z),
    );

    let sign = ray_direction.signum();
    let step = sign.as_ivec3();

    let origin = ray_origin - cell_offset;
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
        calc_t_max(step.x, cell_size.x, inner.x, ray_direction.x),
        calc_t_max(step.y, cell_size.y, inner.y, ray_direction.y),
        calc_t_max(step.z, cell_size.z, inner.z, ray_direction.z),
    );
 
    let mut cell = (origin / cell_size).floor().as_ivec3();
    callback(&cell, 0.0);
    loop {
        if t_max.x <= t_max.y {
            if t_max.x <= t_max.z {
                cell.x += step.x;
                if callback(&cell, t_max.x) {
                    return;
                }
                t_max.x += delta.x;
            } else {
                cell.z += step.z;
                if callback(&cell, t_max.z) {
                    return;
                }
                t_max.z += delta.z;
            }
        } else {
            if t_max.y <= t_max.z {
                cell.y += step.y;
                if callback(&cell, t_max.y) {
                    return;
                }
                t_max.y += delta.y;
            } else {
                cell.z += step.z;
                if callback(&cell, t_max.z) {
                    return;
                }
                t_max.z += delta.z;
            }
        }
    }
}

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

#[test]
fn raycast_test() {
    let point = vec3(0.5, 0.5, 0.5);
    let direction = vec3(1.0, 0.0, 0.0).normalize();
    let cell_size = Vec3::ONE;
    let cell_offset = Vec3::ZERO;
    // let mut counter = 0usize;
    let start = std::time::Instant::now();
    raycast(point, direction, cell_size, cell_offset, |p, d| {
        println!("{p:?}, {d}");
        let loc = point + direction * d;
        println!("Location: {loc:?}");
        d < 100.0
    });
    // for _ in 0..256 {
    //     for _ in 0..256 {
    //         raycast(point, direction, cell_size, cell_offset, 10, |_, _, _| {
    //             // println!("{p} {i}");
    //             counter += 1;
    //             true
    //         });
    //     }
    // }
    println!("Elapsed: {:?}", start.elapsed());
    // println!("Counter: {counter}");
}