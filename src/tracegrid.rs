#![allow(unused)]
use glam::IVec3;

use crate::{chunk::RayHit, math::Face, ray::Ray3};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GridSizeLookup {
    WidthPow2(u32),
    HeightPow2(u32),
    Fallback(u32),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GridSize {
    pub width: u32,
    pub height: u32,
}

impl std::fmt::Display for GridSize {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}x{}", self.width, self.height)
    }
}

impl GridSize {
    pub const fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
        }
    }

    #[inline]
    pub const fn index(self, x: u32, y: u32) -> u32 {
        (y * self.width) + x
    }

    /// Gets the position based on the index.
    #[inline]
    pub const fn inv_index(self, index: u32) -> (u32, u32) {
        (index % self.width, index / self.width)
    }

    #[inline]
    pub const fn iter_width(self) -> std::ops::Range<u32> {
        0..self.width
    }

    #[inline]
    pub const fn iter_height(self) -> std::ops::Range<u32> {
        0..self.height
    }
}

pub struct TraceGrid {
    cells: Box<[Option<RayHit>]>,
    size: GridSize,
}

impl TraceGrid {
    #[inline]
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            cells: (0..width as usize*height as usize).map(|_| Option::<RayHit>::None).collect(),
            size: GridSize::new(width, height),
        }
    }

    pub fn width(&self) -> u32 {
        self.size.width
    }

    pub fn height(&self) -> u32 {
        self.size.height
    }

    pub fn size(&self) -> GridSize {
        self.size
    }

    #[inline]
    pub fn hit(&mut self, x: u32, y: u32, hit: RayHit) {
        if x >= self.size.width || y >= self.size.height {
            return;
        }
        let index = self.size.index(x, y);
        self.cells[index as usize] = Some(hit);
    }

    #[inline]
    pub fn reset(&mut self, x: u32, y: u32) {
        if x >= self.size.width || y >= self.size.height {
            return;
        }
        let index = self.size.index(x, y);
        self.cells[index as usize] = None;
    }

    #[inline]
    pub fn get(&self, x: u32, y: u32) -> Option<&RayHit> {
        if x >= self.size.width || y >= self.size.height {
            return None;
        }
        let index = self.size.index(x, y);
        self.cells[index as usize].as_ref()
    }

    #[inline]
    pub fn clear(&mut self) {
        self.cells.iter_mut().for_each(|cell| *cell = None);
    }
}