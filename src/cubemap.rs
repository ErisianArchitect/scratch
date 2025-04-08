use std::path::Path;

use glam::Vec3A;
use image::*;

use crate::math::Face;

pub struct Cubemap {
    faces: [RgbImage; 6],
    width: u32,
    height: u32,
}

impl Cubemap {

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn new(
        pos_x: RgbImage,
        pos_y: RgbImage,
        pos_z: RgbImage,
        neg_x: RgbImage,
        neg_y: RgbImage,
        neg_z: RgbImage,
    ) -> Self {
        Self {
            width: pos_x.width(),
            height: pos_x.height(),
            faces: [
                pos_x,
                pos_y,
                pos_z,
                neg_x,
                neg_y,
                neg_z,
            ]
        }
    }

    pub fn from_files<P: AsRef<Path>>(files: [P; 6]) -> std::io::Result<Self> {
        let [px, py, pz, nx, ny, nz] = [
            read_texture(files[0].as_ref())?,
            read_texture(files[1].as_ref())?,
            read_texture(files[2].as_ref())?,
            read_texture(files[3].as_ref())?,
            read_texture(files[4].as_ref())?,
            read_texture(files[5].as_ref())?,
        ];
        Ok(Self::new(px, py, pz, nx, ny, nz))
    }

    pub fn sample_uv(&self, face: Face, u: f32, v: f32) -> Rgb<u8> {
        let u = u.clamp(0.0, 1.0);
        let v = v.clamp(0.0, 1.0);
        image::imageops::sample_bilinear(&self.faces[face.index()], u, v).unwrap_or_else(|| Rgb([0; 3]))
    }

    pub fn sample_dir(&self, dir: Vec3A) -> Rgb<u8> {
        let abs = dir.abs();
        if abs.x >= abs.y {
            if abs.x >= abs.z {
                let dx = 0.5 / abs.x;
                if dir.x.is_sign_negative() {
                    let face = Face::NegX;
                    let u = (-dir.z * dx) + 0.5;
                    let v = (-dir.y * dx) + 0.5;
                    self.sample_uv(face, u, v)
                } else {
                    let face = Face::PosX;
                    let u = (dir.z * dx) + 0.5;
                    let v = (-dir.y * dx) + 0.5;
                    self.sample_uv(face, u, v)
                }
            } else {
                let dz = 0.5 / abs.z;
                if dir.z.is_sign_negative() {
                    let face = Face::NegZ;
                    let u = (dir.x * dz) + 0.5;
                    let v = (-dir.y * dz) + 0.5;
                    self.sample_uv(face, u, v)
                } else {
                    let face = Face::PosZ;
                    let u = (-dir.x * dz) + 0.5;
                    let v = (-dir.y * dz) + 0.5;
                    self.sample_uv(face, u, v)
                }
            }
        } else {
            if abs.y >= abs.z {
                let dy = 0.5 / abs.y;
                if dir.y.is_sign_negative() {
                    let face = Face::NegY;
                    let u = (dir.x * dy) + 0.5;
                    let v = (dir.z * dy) + 0.5;
                    self.sample_uv(face, u, v)
                } else {
                    let face = Face::PosY;
                    let u = (dir.x * dy) + 0.5;
                    let v = (-dir.z * dy) + 0.5;
                    self.sample_uv(face, u, v)
                }
            } else {
                let dz = 0.5 / abs.z;
                if dir.z.is_sign_negative() {
                    let face = Face::NegZ;
                    let u = (dir.x * dz) + 0.5;
                    let v = (-dir.y * dz) + 0.5;
                    self.sample_uv(face, u, v)
                } else {
                    let face = Face::PosZ;
                    let u = (-dir.x * dz) + 0.5;
                    let v = (-dir.y * dz) + 0.5;
                    self.sample_uv(face, u, v)
                }
            }
        }
    }
}

pub fn read_texture<P: AsRef<Path>>(path: P) -> std::io::Result<RgbImage> {
    let file = std::fs::File::open(path.as_ref())?;
    let reader = std::io::BufReader::new(file);
    let img = image::load(reader, ImageFormat::Png).expect("Failed to load image.");
    Ok(img.into_rgb8())
}