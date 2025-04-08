use glam::*;

#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct Ray3 {
    pub pos: Vec3A,
    pub dir: Vec3A,
}

impl Ray3 {
    /// Creates a new [Ray3].
    /// 
    /// This does not normalize the direction, so make sure you normalize
    /// that first.
    #[inline(always)]
    pub fn new(pos: Vec3A, dir: Vec3A) -> Self {
        Self {
            pos,
            dir,
        }
    }

    #[inline(always)]
    pub fn from_target(pos: Vec3A, target: Vec3A) -> Self {
        Self {
            pos,
            dir: (target - pos).normalize(),
        }
    }

    #[inline(always)]
    pub fn invert_dir(self) -> Self {
        Self {
            pos: self.pos,
            dir: -self.dir,
        }
    }

    #[inline(always)]
    pub fn point_on_ray(&self, distance: f32) -> Vec3A {
        (self.dir * distance) + self.pos
    }
}