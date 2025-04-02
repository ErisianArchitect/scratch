use glam::{
    vec2, Mat3, Mat4, Quat, Vec2, Vec3, Vec4, Vec4Swizzles
};

use crate::ray::Ray3;

pub fn rotation_from_look_at(position: Vec3, target: Vec3) -> Vec2 {
    let dir = (target - position).normalize();
    rotation_from_direction(dir)
}

pub fn rotation_from_direction(direction: Vec3) -> Vec2 {
    let yaw = (-direction.x).atan2(-direction.z);
    let pitch = direction.y.asin();
    vec2(pitch, yaw)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MoveType {
    /// Absolute movement. No rotation of the translation vector.
    Absolute,
    /// Free movement. Rotates the translation vector with the camera.
    Free,
    /// Planar movement. Rotates the translation vector with the angle around the Y axis.
    Planar,
}

#[derive(Debug, Clone)]
pub struct Camera {
    pub position: Vec3,
    pub rotation: Vec2,
    pub fov: f32,
    pub aspect_ratio: f32,
    pub z_near: f32,
    pub z_far: f32,
    pub screen_size: (u32, u32),
}

const fn aspect_ratio(size: (u32, u32)) -> f32 {
    size.0 as f32 / size.1 as f32
}

impl Camera {
    pub fn new(
        position: Vec3,
        rotation: Vec2,
        fov: f32,
        z_near: f32,
        z_far: f32,
        screen_size: (u32, u32),
    ) -> Self {
        Self {
            position,
            rotation,
            fov,
            aspect_ratio: aspect_ratio(screen_size),
            z_near,
            z_far,
            screen_size,
        }
    }

    /// Creates an unrotated camera at the given position.
    pub fn at(
        position: Vec3,
        fov: f32,
        z_near: f32,
        z_far: f32,
        screen_size: (u32, u32),
    ) -> Self {
        Self {
            position,
            rotation: Vec2::ZERO,
            fov,
            aspect_ratio: aspect_ratio(screen_size),
            z_near,
            z_far,
            screen_size,
        }
    }

    pub fn from_look_at(
        position: Vec3,
        target: Vec3,
        fov: f32,
        z_near: f32,
        z_far: f32,
        screen_size: (u32, u32),
    ) -> Self {
        let rotation = rotation_from_look_at(position, target);
        Self {
            position,
            rotation,
            fov,
            aspect_ratio: aspect_ratio(screen_size),
            z_near,
            z_far,
            screen_size,
        }
    }

    /// `look_to` means to point in the same direction as the given `direction` vector.
    pub fn from_look_to(
        position: Vec3,
        direction: Vec3,
        fov: f32,
        z_near: f32,
        z_far: f32,
        screen_size: (u32, u32),
    ) -> Self {
        let rotation = rotation_from_direction(direction);
        Self {
            position,
            rotation,
            fov,
            aspect_ratio: aspect_ratio(screen_size),
            z_near,
            z_far,
            screen_size,
        }
    }

    pub fn resize(&mut self, size: (u32, u32)) {
        self.screen_size = size;
        self.aspect_ratio = aspect_ratio(size);
    }

    pub fn rotate_vec(&self, v: Vec3) -> Vec3 {
        let rot = self.quat();
        rot * v
    }

    /// Rotates vector around the Y axis.
    pub fn rotate_vec_y(&self, v: Vec3) -> Vec3 {
        let rot = self.y_quat();
        rot * v
    }

    pub fn up(&self) -> Vec3 {
        self.rotate_vec(Vec3::Y)
    }

    pub fn down(&self) -> Vec3 {
        self.rotate_vec(Vec3::NEG_Y)
    }

    pub fn left(&self) -> Vec3 {
        self.rotate_vec(Vec3::NEG_X)
    }

    pub fn right(&self) -> Vec3 {
        self.rotate_vec(Vec3::X)
    }

    pub fn forward(&self) -> Vec3 {
        self.rotate_vec(Vec3::NEG_Z)
    }

    pub fn backward(&self) -> Vec3 {
        self.rotate_vec(Vec3::Z)
    }

    pub fn pan_forward(&self) -> Vec3 {
        self.rotate_vec_y(Vec3::NEG_Z)
    }

    pub fn pan_backward(&self) -> Vec3 {
        self.rotate_vec_y(Vec3::Z)
    }

    pub fn adv_move(&mut self, move_type: MoveType, translation: Vec3) {
        match move_type {
            MoveType::Absolute => self.translate(translation),
            MoveType::Free => self.translate_rotated(translation),
            MoveType::Planar => self.translate_planar(translation),
        }
    }

    pub fn translate(&mut self, translation: Vec3) {
        self.position += translation;
    }

    /// Translates relative to camera rotation.
    pub fn translate_rotated(&mut self, translation: Vec3) {
        if translation.length_squared() > 0.000001 {
            let rot_quat = self.quat();
            let rot_offset = rot_quat * translation;
            self.translate(rot_offset);
        }
    }

    /// For planar camera translation.
    pub fn translate_planar(&mut self, translation: Vec3) {
        if translation.length_squared() > 0.000001 {
            self.translate(self.rotate_vec_y(translation))
        }
    }

    pub fn look_at(&mut self, target: Vec3) {
        self.rotation = rotation_from_look_at(self.position, target);
    }

    pub fn look_to(&mut self, direction: Vec3) {
        self.rotation = rotation_from_direction(direction);
    }

    pub fn rotate(&mut self, rotation_radians: Vec2) {
        self.rotation += rotation_radians;
        self.rotation.x = self.rotation.x.clamp(-90f32.to_radians(), 90f32.to_radians());
        self.rotation.y = self.rotation.y.rem_euclid(360f32.to_radians());
    }

    pub fn rotate_x(&mut self, radians: f32) {
        self.rotation.x += radians;
        self.rotation.x = self.rotation.x.clamp(-90f32.to_radians(), 90f32.to_radians());
    }

    pub fn rotate_y(&mut self, radians: f32) {
        self.rotation.y += radians;
        self.rotation.y = self.rotation.y.rem_euclid(360f32.to_radians());
    }

    /// Returns the quaternion for the [Camera]'s rotation.
    pub fn quat(&self) -> Quat {
        Quat::from_euler(glam::EulerRot::YXZ, self.rotation.y, self.rotation.x, 0.)
    }

    pub fn x_quat(&self) -> Quat {
        Quat::from_axis_angle(Vec3::X, self.rotation.x)
    }

    pub fn y_quat(&self) -> Quat {
        Quat::from_axis_angle(Vec3::Y, self.rotation.y)
    }

    pub fn view_matrix(&self) -> Mat4 {
        let rot_quat = self.quat();
        let up = rot_quat * Vec3::Y;
        let dir = rot_quat * Vec3::NEG_Z;
        Mat4::look_to_rh(self.position, dir, up)
    }

    pub fn rotation_matrix(&self) -> Mat3 {
        Mat3::from_euler(glam::EulerRot::YXZ, self.rotation.y, self.rotation.x, 0.0)
    }

    pub fn projection_matrix(&self) -> Mat4 {
        Mat4::perspective_rh(self.fov, self.aspect_ratio, self.z_near, self.z_far)
    }

    pub fn projection_view_matrix(&self) -> Mat4 {
        self.projection_matrix() * self.view_matrix()
    }

    pub fn world_to_clip(&self, pos: Vec3) -> Vec4 {
        let view_proj = self.projection_view_matrix();
        let pos_w = Vec4::new(pos.x, pos.y, pos.z, 1.0);
        view_proj * pos_w
    }

    pub fn world_to_clip_ncd(&self, pos: Vec3) -> Vec3 {
        let clip = self.world_to_clip(pos);
        clip.xyz() / clip.w
    }
    
    pub fn normalized_screen_to_ray(&self, screen_pos: Vec2) -> Ray3 {
        let inv_proj_view = self.projection_view_matrix().inverse();

        let near_point = inv_proj_view * Vec4::new(screen_pos.x, -screen_pos.y, 0.0, 1.0);
        let near_point = near_point.xyz() / near_point.w;
        let far_point = inv_proj_view * Vec4::new(screen_pos.x, -screen_pos.y, self.z_far, 1.0);
        let far_point = far_point.xyz() / far_point.w;

        let direction = (near_point - far_point).normalize();

        // Ray3::new(near_point.xyz(), direction)
        Ray3::new(self.position, direction)
    }
}