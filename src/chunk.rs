use std::f32::MIN_POSITIVE;

use crate::{math::Face, ray::Ray3};
use glam::*;


// 64x64x64 chunk.
pub struct Chunk {
    cols: Box<[u64]>,
}

impl Chunk {
    pub fn new() -> Self {
        Self {
            cols: (0..4096).map(|_| 0u64).collect::<Box<_>>(),
        }
    }

    pub fn col_index(x: u32, z: u32) -> u32 {
        x | (z << 6)
    }

    pub fn get(&self, x: u32, y: u32, z: u32) -> bool {
        if (x | y | z) >= 64 {
            return false;
        }
        let index = Self::col_index(x, z);
        let col = self.cols[index as usize];
        (col & (1 << y)) != 0
    }

    pub fn set(&mut self, x: u32, y: u32, z: u32, on: bool) {
        let index = Self::col_index(x, z);
        let mut col = self.cols[index as usize];
        if on {
            col = col | (1 << y);
        } else {
            col = col & !(1 << y);
        }
        self.cols[index as usize] = col;
    }

    pub fn fill_box(&mut self, start: (u32, u32, u32), end: (u32, u32, u32), on: bool) {
        let mask = (1u64 << end.1) - 1;
        let mask = mask & !((1 << start.1) - 1);
        if on {
            for z in start.2..end.2 {
                for x in start.0..end.0 {
                    let index = Self::col_index(x, z);
                    let col = self.cols[index as usize];
                    self.cols[index as usize] = col | mask;
                }
            }
        } else {
            let mask = !mask;
            for z in start.2..end.2 {
                for x in start.0..end.0 {
                    let index = Self::col_index(x, z);
                    let col = self.cols[index as usize];
                    self.cols[index as usize] = col & mask;
                }
            }

        }
    }

    pub fn raycast(&self, ray: Ray3, max_distance: f32) -> Option<RayHit> {
        let mut ray = ray;
        let (step, delta_max, delta_add) = if ray.pos.x < 0.0 || ray.pos.y < 0.0 || ray.pos.z < 0.0
        || ray.pos.x >= 64.0 || ray.pos.y >= 64.0 || ray.pos.z >= 64.0 {
            // Calculate entry point or early return.
            if (ray.pos.x < 0.0 && ray.dir.x <= 0.0)
            || (ray.pos.x >= 64.0 && ray.dir.x >= 0.0)
            || (ray.pos.y < 0.0 && ray.dir.y <= 0.0)
            || (ray.pos.y >= 64.0 && ray.dir.y >= 0.0)
            || (ray.pos.z < 0.0 && ray.dir.z <= 0.0)
            || (ray.pos.z >= 64.0 && ray.dir.z >= 0.0) {
                return None;
            }
            // calculate distance to cross each plane
            let sign = ray.dir.signum();
            let step = sign.as_ivec3();
            let (dx_min, dx_max) = if step.x > 0 {
                (
                    -ray.pos.x / ray.dir.x,
                    (64.0 - ray.pos.x) / ray.dir.x,
                )
            } else if step.x < 0 {
                (
                    (ray.pos.x - 64.0) / -ray.dir.x,
                    ray.pos.x / -ray.dir.x,
                )
            } else {
                (<f32>::NEG_INFINITY, <f32>::INFINITY)
            };
            let (dy_min, dy_max) = if step.y > 0 {
                (
                    -ray.pos.y / ray.dir.y,
                    (64.0 - ray.pos.y) / ray.dir.y,
                )
            } else if step.y < 0 {
                (
                    (ray.pos.y - 64.0) / -ray.dir.y,
                    ray.pos.y / -ray.dir.y,
                )
            } else {
                (<f32>::NEG_INFINITY, <f32>::INFINITY)
            };
            let (dz_min, dz_max) = if step.z > 0 {
                (
                    -ray.pos.z / ray.dir.z,
                    (64.0 - ray.pos.z) / ray.dir.z,
                )
            } else if step.z < 0 {
                (
                    (ray.pos.z - 64.0) / -ray.dir.z,
                    ray.pos.z / -ray.dir.z,
                )
            } else {
                (<f32>::NEG_INFINITY, <f32>::INFINITY)
            };

            let max_min = dx_min.max(dy_min.max(dz_min));
            let min_max = dx_max.min(dy_max.min(dz_max));
            if max_min >= min_max {
                return None;
            }
            // let pos = ray.pos;
            ray.pos = ray.pos + ray.dir * (max_min + 1e-3);
            // println!(
            //     "{:?} {:?} {:?} ({}, {}, {})",
            //     pos,
            //     ray.dir,
            //     (ray.pos.x.floor(), ray.pos.y.floor(), ray.pos.z.floor()),
            //     dx_min, dy_min, dz_min,
            // );
            // return None;
            (
                step,
                vec3(dx_max, dy_max, dz_max),
                max_min,
            )
        } else {
            let sign = ray.dir.signum();
            let step = sign.as_ivec3();
            let dx_max = if step.x > 0 {
                (64.0 - ray.pos.x) / ray.dir.x
            } else if step.x < 0 {
                ray.pos.x / -ray.dir.x
            } else {
                <f32>::INFINITY
            };
            let dy_max = if step.y > 0 {
                (64.0 - ray.pos.y) / ray.dir.y
            } else if step.y < 0 {
                ray.pos.y / -ray.dir.y
            } else {
                <f32>::INFINITY
            };
            let dz_max = if step.z > 0 {
                (64.0 - ray.pos.z) / ray.dir.z
            } else if step.z < 0 {
                ray.pos.z / -ray.dir.z
            } else {
                <f32>::INFINITY
            };
            (
                step,
                vec3(dx_max, dy_max, dz_max),
                0.0,
            )
        };
        fn calc_delta(mag: f32) -> f32 {
            1.0 / mag.abs().max(<f32>::MIN_POSITIVE)
        }
        let delta = vec3(
            calc_delta(ray.dir.x),
            calc_delta(ray.dir.y),
            calc_delta(ray.dir.z),
        );

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

        let fract = ray.pos.fract();

        fn calc_t_max(step: i32, fract: f32, mag: f32) -> f32 {
            if step > 0 {
                (1.0 - fract) / mag.abs().max(<f32>::MIN_POSITIVE)
            } else if step < 0 {
                fract / mag.abs().max(<f32>::MIN_POSITIVE)
            } else {
                <f32>::INFINITY
            }
        }
        let mut t_max = vec3(
            calc_t_max(step.x, fract.x, ray.dir.x) + delta_add,
            calc_t_max(step.y, fract.y, ray.dir.y) + delta_add,
            calc_t_max(step.z, fract.z, ray.dir.z) + delta_add,
        );
        const SCALE_FACTOR: f32 = 10000.0;
        const UNSCALE_FACTOR: f32 = 1.0 / SCALE_FACTOR;
        fn scale(value: f32) -> i32 {
            (value * SCALE_FACTOR) as i32
        }
        fn unscale(value: i32) -> f32 {
            (value as f32) * UNSCALE_FACTOR
        }
        let mut i_tmax = ivec3(
            scale(t_max.x),
            scale(t_max.y),
            scale(t_max.z),
        );
        let idelta = ivec3(
            scale(delta.x),
            scale(delta.y),
            scale(delta.z),
        );
        let idelta_max = ivec3(
            scale(delta_max.x),
            scale(delta_max.y),
            scale(delta_max.z),
        );
        let i_max_dist = scale(max_distance);

        let mut cell = ray.pos.floor().as_ivec3();
        let coord = (
            cell.x as u32,
            cell.y as u32,
            cell.z as u32,
        );
        if self.get(coord.0, coord.1, coord.2) {
            return Some(RayHit {
                face: None,
                coord: cell,
                distance: 0.0,
            });
        }
        loop {
            if i_tmax.x <= i_tmax.y {
                if i_tmax.x <= i_tmax.z {
                    if i_tmax.x >= idelta_max.x || i_tmax.x >= i_max_dist {
                        return None;
                    }
                    cell.x += step.x;
                    let coord = (cell.x as u32, cell.y as u32, cell.z as u32);
                    if self.get(coord.0, coord.1, coord.2) {
                        return Some(RayHit::hit_face(face.0, cell, unscale(i_tmax.x)));
                    }
                    i_tmax.x += idelta.x;
                } else {
                    if i_tmax.z >= idelta_max.z || i_tmax.z >= i_max_dist {
                        return None;
                    }
                    cell.z += step.z;
                    let coord = (cell.x as u32, cell.y as u32, cell.z as u32);
                    if self.get(coord.0, coord.1, coord.2) {
                        return Some(RayHit::hit_face(face.2, cell, unscale(i_tmax.z)));
                    }
                    i_tmax.z += idelta.z;
                }
            } else {
                if i_tmax.y <= i_tmax.z {
                    if i_tmax.y >= idelta_max.y || i_tmax.y >= i_max_dist {
                        return None;
                    }
                    cell.y += step.y;
                    let coord = (cell.x as u32, cell.y as u32, cell.z as u32);
                    if self.get(coord.0, coord.1, coord.2) {
                        return Some(RayHit::hit_face(face.1, cell, unscale(i_tmax.y)));
                    }
                    i_tmax.y += idelta.y;
                } else {
                    if i_tmax.z >= idelta_max.z || i_tmax.z >= i_max_dist {
                        return None;
                    }
                    cell.z += step.z;
                    let coord = (cell.x as u32, cell.y as u32, cell.z as u32);
                    if self.get(coord.0, coord.1, coord.2) {
                        return Some(RayHit::hit_face(face.2, cell, unscale(i_tmax.z)));
                    }
                    i_tmax.z += idelta.z;
                }
            }
        }
    }
}

pub struct RayHit {
    pub face: Option<Face>,
    pub coord: IVec3,
    pub distance: f32,
}

impl RayHit {
    pub fn hit_face(face: Face, coord: IVec3, distance: f32) -> Self {
        Self {
            face: Some(face),
            coord,
            distance,
        }
    }

    pub fn hit_cell(coord: IVec3, distance: f32) -> Self {
        Self {
            face: None,
            coord,
            distance,
        }
    }
}

#[test]
fn size_test() {
    println!("RayHit Size: {}", std::mem::size_of::<RayHit>());
    println!("Option<RayHit> Size: {}", std::mem::size_of::<Option<RayHit>>());
}