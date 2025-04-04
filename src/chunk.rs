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

    pub fn col_index(x: i32, z: i32) -> i32 {
        (x & 0b111111) | ((z & 0b111111) << 6)
    }

    pub fn get(&self, x: i32, y: i32, z: i32) -> bool {
        let xyz = x | y | z;
        if xyz >= 64 || xyz < 0 {
            return false;
        }
        let index = Self::col_index(x, z);
        let col = self.cols[index as usize];
        (col & (1 << y)) != 0
    }

    pub fn get_reflection(&self, x: i32, y: i32, z: i32) -> bool {
        let xyz = x | y | z;
        if xyz >= 64 || xyz < 0 {
            return false;
        }
        let index = Self::col_index(x, z);
        let col = self.reflection_cols[index as usize];
        (col & (1 << y)) != 0
    }

    pub fn set(&mut self, x: i32, y: i32, z: i32, on: bool) {
        let xyz = x | y | z;
        if xyz >= 64 || xyz < 0 {
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
        if xyz >= 64 || xyz < 0 {
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
        let gt = ray.pos.cmpge(SIXTY_FOUR);
        let outside = lt | gt;
        let (step, delta_min, delta_max, delta_add) = if outside.any() {
            // Calculate entry point (if there is one).
            // calculate distance to cross each plane
            let sign = ray.dir.signum();
            let step = sign.as_ivec3();
            if step.x < 0 && lt.test(0)
            || step.x > 0 && gt.test(0)
            || step.y < 0 && lt.test(1)
            || step.y > 0 && gt.test(1)
            || step.z < 0 && lt.test(2)
            || step.z > 0 && gt.test(2) {
                return None;
            }
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
            ray.pos = ray.pos + ray.dir * (max_min + 1e-5);
            (
                step,
                Some(vec3(dx_min, dy_min, dz_min)),
                vec3(dx_max, dy_max, dz_max),
                max_min + 1e-5,
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
        if delta_add >= max_distance {
            return None;
        }
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
        loop {
            if t_max.x <= t_max.y {
                if t_max.x <= t_max.z {
                    if t_max.x >= delta_max.x || t_max.x >= max_distance {
                        return None;
                    }
                    cell.x += step.x;
                    if self.get(cell.x, cell.y, cell.z) {
                        return Some(RayHit::hit_face(face.0, cell, t_max.x));
                    }
                    t_max.x += delta.x;
                } else {
                    if t_max.z >= delta_max.z || t_max.z >= max_distance {
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
                    if t_max.y >= delta_max.y || t_max.y >= max_distance {
                        return None;
                    }
                    cell.y += step.y;
                    if self.get(cell.x, cell.y, cell.z) {
                        return Some(RayHit::hit_face(face.1, cell, t_max.y));
                    }
                    t_max.y += delta.y;
                } else {
                    if t_max.z >= delta_max.z || t_max.z >= max_distance {
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

    #[inline(always)]
    pub fn get_hit_point(&self, ray: Ray3, face: Face) -> Vec3A {
        let point = ray.point_on_ray(self.distance);
        let pre_hit = self.coord + match face {
            Face::PosX => ivec3(1, 0, 0),
            Face::PosY => ivec3(0, 1, 0),
            Face::PosZ => ivec3(0, 0, 1),
            Face::NegX => ivec3(-1, 0, 0),
            Face::NegY => ivec3(0, -1, 0),
            Face::NegZ => ivec3(0, 0, -1),
        };
        let pre_hit = pre_hit.as_vec3a();
        const SMIDGEN: Vec3A = Vec3A::splat(1e-3);
        const UNSMIDGEN: Vec3A = Vec3A::splat(1.0-1e-3);
        let min = pre_hit + SMIDGEN;
        let max = pre_hit + UNSMIDGEN;
        point.clamp(min, max)
    }
}

#[test]
fn size_test() {
    println!("RayHit Size: {}", std::mem::size_of::<RayHit>());
    println!("Option<RayHit> Size: {}", std::mem::size_of::<Option<RayHit>>());
}

#[cfg(test)]
mod tests {
    use crate::tracegrid::GridSize;

    use super::*;
    #[test]
    fn next_pow2() {
        let size = GridSize::new(1920/2, 1080/2);
        let w = size.width.next_power_of_two() - size.width;
        let h = size.height.next_power_of_two() - size.height;
        if w <= h {
            println!("Width: {}\nNext Pow2: {}\nDifference: {}", size.width, size.width.next_power_of_two(), w);
        } else {
            println!("Height: {}\nNext Pow2: {}\nDifference: {}", size.height, size.height.next_power_of_two(), h);
        }
        let a = 1512;
        let b = 5125;
        println!("{}", (a | b) < 0);
    }
}