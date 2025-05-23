use crate::{math::Face, ray::Ray3};
use glam::*;


// 64x64x64 chunk.
pub struct Chunk {
    cols: Box<[u64]>,
    reflection_cols: Box<[u64]>,
}

impl Chunk {
    pub fn new() -> Self {
        Self {
            cols: (0..4096).map(|_| 0u64).collect::<Box<_>>(),
            reflection_cols: (0..4096).map(|_| 0u64).collect::<Box<_>>(),
        }
    }

    #[inline(always)]
    pub fn col_index(x: i32, z: i32) -> i32 {
        x | z << 6
    }

    #[inline(always)]
    pub fn get(&self, x: i32, y: i32, z: i32) -> bool {
        // Before
        // if x < 0 || y < 0 || z < 0 || x >= 64 || y >= 64 || z >= 64 {
        //     return false;
        // }
        // After
        // This is a neat bit level trick to bounds check x, y, and z
        // simultaneously. This works because of Two's Complement, which
        // causes negative numbers to become exceedingly high positive
        // numbers when cast to an unsigned type.
        // This reduces the bounds check from 11 instructions down to just 4.
        let xyz = x | y | z;
        if (xyz as u32) >= 64 { // this works because of twos-complement.
            return false;
        }

        let index = Self::col_index(x, z);
        let col = self.cols[index as usize];
        (col & (1 << y)) != 0
    }

    #[inline(always)]
    pub fn get_reflection(&self, x: i32, y: i32, z: i32) -> bool {
        // Before
        // if x < 0 || y < 0 || z < 0 || x >= 64 || y >= 64 || z >= 64 {
        //     return false;
        // }
        // After
        let xyz = x | y | z;
        if (xyz as u32) >= 64 {
            return false;
        }
        let index = Self::col_index(x, z);
        let col = self.reflection_cols[index as usize];
        (col & (1 << y)) != 0
    }

    pub fn set(&mut self, x: i32, y: i32, z: i32, on: bool) {
        let xyz = x | y | z;
        if (xyz as u32) >= 64 {
            return;
        }
        let index = Self::col_index(x, z);
        let mut col = self.cols[index as usize];
        if on {
            col = col | (1 << y);
        } else {
            col = col & !(1 << y);
        }
        self.cols[index as usize] = col;
    }

    pub fn set_reflection(&mut self, x: i32, y: i32, z: i32, on: bool) {
        let xyz = x | y | z;
        if (xyz as u32) >= 64 {
            return;
        }
        let index = Self::col_index(x, z);
        let mut col = self.reflection_cols[index as usize];
        if on {
            col = col | (1 << y);
        } else {
            col = col & !(1 << y);
        }
        self.reflection_cols[index as usize] = col;
    }

    pub fn fill_box(&mut self, start: (i32, i32, i32), end: (i32, i32, i32), on: bool) {
        let end = (end.0.clamp(0, 64), end.1.clamp(0, 64), end.2.clamp(0, 64));
        let start = (
            start.0.clamp(0, end.0),
            start.1.clamp(0, end.1),
            start.2.clamp(0, end.2),
        );
        for x in start.0..end.0 {
            for z in start.2..end.2 {
                for y in start.1..end.1 {
                    self.set(x, y, z, on);
                }
            }
        }
    }

    pub fn fill_box_reflection(&mut self, start: (i32, i32, i32), end: (i32, i32, i32), on: bool) {
        let end = (end.0.clamp(0, 64), end.1.clamp(0, 64), end.2.clamp(0, 64));
        let start = (
            start.0.clamp(0, end.0),
            start.1.clamp(0, end.1),
            start.2.clamp(0, end.2),
        );
        for x in start.0..end.0 {
            for z in start.2..end.2 {
                for y in start.1..end.1 {
                    self.set_reflection(x, y, z, on);
                }
            }
        }
    }

    pub fn set_with<F: Fn(bool) -> bool>(&mut self, x: i32, y: i32, z: i32, f: F) {
        let cur = self.get(x, y, z);
        let new = f(cur);
        self.set(x, y, z, new);
    }

    pub fn set_with_reflection<F: Fn(bool) -> bool>(&mut self, x: i32, y: i32, z: i32, f: F) {
        let cur = self.get_reflection(x, y, z);
        let new = f(cur);
        self.set_reflection(x, y, z, new);
    }

    pub fn draw_box_with<F: Fn(bool) -> bool>(&mut self, start: (i32, i32, i32), end: (i32, i32, i32), f: F) {
        for y in start.1..end.1 {
            self.set_with(start.0, y, start.2, |b| f(b));
            self.set_with(end.0 - 1, y, start.2, |b| f(b));
            self.set_with(start.0, y, end.2 - 1, |b| f(b));
            self.set_with(end.0 - 1, y, end.2 - 1, |b| f(b));
        }
        for z in start.2+1..end.2-1 {
            self.set_with(start.0, start.1, z, |b| f(b));
            self.set_with(end.0-1, start.1, z, |b| f(b));
            self.set_with(start.0, end.1-1, z, |b| f(b));
            self.set_with(end.0-1, end.1-1, z, |b| f(b));
        }
        for x in start.0+1..end.0-1 {
            self.set_with(x, start.1, start.2, |b| f(b));
            self.set_with(x, end.1-1, start.2, |b| f(b));
            self.set_with(x, start.1, end.2-1, |b| f(b));
            self.set_with(x, end.1-1, end.2-1, |b| f(b));
        }
    }

    pub fn draw_box_with_reflection<F: Fn(bool) -> bool>(&mut self, start: (i32, i32, i32), end: (i32, i32, i32), f: F) {
        for y in start.1..end.1 {
            self.set_with_reflection(start.0, y, start.2, |b| f(b));
            self.set_with_reflection(end.0 - 1, y, start.2, |b| f(b));
            self.set_with_reflection(start.0, y, end.2 - 1, |b| f(b));
            self.set_with_reflection(end.0 - 1, y, end.2 - 1, |b| f(b));
        }
        for z in start.2+1..end.2-1 {
            self.set_with_reflection(start.0, start.1, z, |b| f(b));
            self.set_with_reflection(end.0-1, start.1, z, |b| f(b));
            self.set_with_reflection(start.0, end.1-1, z, |b| f(b));
            self.set_with_reflection(end.0-1, end.1-1, z, |b| f(b));
        }
        for x in start.0+1..end.0-1 {
            self.set_with_reflection(x, start.1, start.2, |b| f(b));
            self.set_with_reflection(x, end.1-1, start.2, |b| f(b));
            self.set_with_reflection(x, start.1, end.2-1, |b| f(b));
            self.set_with_reflection(x, end.1-1, end.2-1, |b| f(b));
        }
    }

    pub fn draw_box(&mut self, start: (i32, i32, i32), end: (i32, i32, i32), on: bool) {
        self.draw_box_with(start, end, |_| on);
    }

    pub fn draw_box_reflection(&mut self, start: (i32, i32, i32), end: (i32, i32, i32), on: bool) {
        self.draw_box_with_reflection(start, end, |_| on);
    }

    pub fn raycast(&self, ray: Ray3, max_distance: f32) -> Option<RayHit> {
        let mut ray = ray;
        let lt = ray.pos.cmplt(Vec3A::ZERO);
        const SIXTY_FOUR: Vec3A = Vec3A::splat(64.0);
        let ge = ray.pos.cmpge(SIXTY_FOUR);
        let outside = lt | ge;
        let (step, delta_min, delta_max, delta_add) = if outside.any() {
            // Calculate entry point (if there is one).
            // calculate distance to cross each plane
            let sign = ray.dir.signum();
            let step = sign.as_ivec3();

            let neg_sign = sign.cmplt(Vec3A::ZERO);
            let pos_sign = sign.cmpgt(Vec3A::ZERO);

            if ((lt & neg_sign) | (ge & pos_sign)).any() {
                return None;
            }
            // if lt.test(0) && step.x < 0 // 4
            // || lt.test(1) && step.y < 0 // 5 9
            // || lt.test(2) && step.z < 0 // 5 14
            // || ge.test(0) && step.x > 0 // 5 19
            // || ge.test(1) && step.y > 0 // 5 24
            // || ge.test(2) && step.z > 0 {// 5 29
            //     return None;
            // }
            let (dx_min, dx_max) = match step.x + 1 {
                0 => {
                    (
                        (ray.pos.x - 64.0) / -ray.dir.x,
                        ray.pos.x / -ray.dir.x,
                    )
                }
                1 => {
                    (<f32>::NEG_INFINITY, <f32>::INFINITY)
                }
                2 => {
                    (
                        -ray.pos.x / ray.dir.x,
                        (64.0 - ray.pos.x) / ray.dir.x,
                    )
                }
                _ => unreachable!(),
            };
            let (dy_min, dy_max) = match step.y + 1 {
                0 => {
                    (
                        (ray.pos.y - 64.0) / -ray.dir.y,
                        ray.pos.y / -ray.dir.y,
                    )
                }
                1 => {
                    (<f32>::NEG_INFINITY, <f32>::INFINITY)
                }
                2 => {
                    (
                        -ray.pos.y / ray.dir.y,
                        (64.0 - ray.pos.y) / ray.dir.y,
                    )
                }
                _ => unreachable!()
            };
            let (dz_min, dz_max) = match step.z + 1 {
                0 => {
                    (
                        (ray.pos.z - 64.0) / -ray.dir.z,
                        ray.pos.z / -ray.dir.z,
                    )
                }
                1 => {
                    (<f32>::NEG_INFINITY, <f32>::INFINITY)
                }
                2 => {
                    (
                        -ray.pos.z / ray.dir.z,
                        (64.0 - ray.pos.z) / ray.dir.z,
                    )
                }
                _ => unreachable!()
            };
            let max_min = dx_min.max(dy_min.max(dz_min));
            let min_max = dx_max.min(dy_max.min(dz_max));
            // Early return, the ray does not hit the volume.
            if max_min >= min_max {
                return None;
            }
            // This is needed to penetrate the ray into the bounding box.
            // Otherwise you'll get weird circles from the rays popping
            // in and out of the next cell. This ensures that the ray
            // will be inside the bounding box.
            const RAY_PENETRATION: f32 = 1e-5;
            let delta_add = max_min + RAY_PENETRATION;
            if delta_add >= max_distance {
                return None;
            }
            ray.pos = ray.pos + ray.dir * delta_add;
            (
                step,
                Some(vec3(dx_min, dy_min, dz_min)),
                vec3(dx_max, dy_max, dz_max),
                delta_add,
            )
        } else {
            let sign = ray.dir.signum();
            let step = sign.as_ivec3();
            let dx_max = match step.x + 1 {
                0 => {
                    ray.pos.x / -ray.dir.x
                }
                1 => {
                    <f32>::INFINITY
                }
                2 => {
                    (64.0 - ray.pos.x) / ray.dir.x
                }
                _ => unreachable!()
            };
            let dy_max = match step.y + 1 {
                0 => {
                    ray.pos.y / -ray.dir.y
                }
                1 => {
                    <f32>::INFINITY
                }
                2 => {
                    (64.0 - ray.pos.y) / ray.dir.y
                }
                _ => unreachable!()
            };
            let dz_max = match step.z + 1{
                0 => {
                    ray.pos.z / -ray.dir.z
                }
                1 => {
                    <f32>::INFINITY
                }
                2 => {
                    (64.0 - ray.pos.z) / ray.dir.z
                }
                _ => unreachable!()
            };
            (
                step,
                None,
                vec3(dx_max, dy_max, dz_max),
                0.0,
            )
        };
        #[inline(always)]
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

        #[inline(always)]
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

        let mut cell = ray.pos.floor().as_ivec3();
        if self.get(cell.x, cell.y, cell.z) {
            return Some(RayHit {
                face: delta_min.map(|min| {
                    if min.x >= min.y {
                        if min.x >= min.z {
                            face.0
                        } else {
                            face.2
                        }
                    } else {
                        if min.y >= min.z {
                            face.1
                        } else {
                            face.2
                        }
                    }
                }),
                coord: cell,
                distance: delta_add,
            });
        }
        let max_d = vec3a(
            delta_max.x.min(max_distance),
            delta_max.y.min(max_distance),
            delta_max.z.min(max_distance),
        );
        loop {

            if t_max.x <= t_max.y {
                if t_max.x <= t_max.z {
                    if t_max.x >= max_d.x {
                        return None;
                    }
                    cell.x += step.x;
                    if self.get(cell.x, cell.y, cell.z) {
                        return Some(RayHit::hit_face(face.0, cell, t_max.x));
                    }
                    t_max.x += delta.x;
                } else {
                    if t_max.z >= max_d.z {
                        return None;
                    }
                    cell.z += step.z;
                    if self.get(cell.x, cell.y, cell.z) {
                        return Some(RayHit::hit_face(face.2, cell, t_max.z));
                    }
                    t_max.z += delta.z;
                }
            } else {
                if t_max.y <= t_max.z {
                    if t_max.y >= max_d.y {
                        return None;
                    }
                    cell.y += step.y;
                    if self.get(cell.x, cell.y, cell.z) {
                        return Some(RayHit::hit_face(face.1, cell, t_max.y));
                    }
                    t_max.y += delta.y;
                } else {
                    if t_max.z >= max_d.z {
                        return None;
                    }
                    cell.z += step.z;
                    if self.get(cell.x, cell.y, cell.z) {
                        return Some(RayHit::hit_face(face.2, cell, t_max.z));
                    }
                    t_max.z += delta.z;
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
    #[inline(always)]
    pub fn hit_face(face: Face, coord: IVec3, distance: f32) -> Self {
        Self {
            face: Some(face),
            coord,
            distance,
        }
    }

    #[inline(always)]
    pub fn hit_cell(coord: IVec3, distance: f32) -> Self {
        Self {
            face: None,
            coord,
            distance,
        }
    }

    #[inline(always)]
    pub fn get_hit_point(&self, ray: Ray3, face: Face) -> Vec3A {
        let point = ray.point_on_ray(self.distance);
        let pre_hit = match face {
            Face::PosX => ivec3(self.coord.x + 1, self.coord.y, self.coord.z),
            Face::PosY => ivec3(self.coord.x, self.coord.y + 1, self.coord.z),
            Face::PosZ => ivec3(self.coord.x, self.coord.y, self.coord.z + 1),
            Face::NegX => ivec3(self.coord.x - 1, self.coord.y, self.coord.z),
            Face::NegY => ivec3(self.coord.x, self.coord.y - 1, self.coord.z),
            Face::NegZ => ivec3(self.coord.x, self.coord.y, self.coord.z - 1),
        };
        let pre_hit = pre_hit.as_vec3a();
        const SMIDGEN: Vec3A = Vec3A::splat(1e-3);
        const UNSMIDGEN: Vec3A = Vec3A::splat(1.0-1e-3);
        // sometimes the hit-point is in the wrong cell (if it goes too far)
        // so you want to bring it back into the correct cell.
        let min = pre_hit + SMIDGEN;
        let max = pre_hit + UNSMIDGEN;
        // point.max(min).min(max)
        point.clamp(min, max)
    }
}