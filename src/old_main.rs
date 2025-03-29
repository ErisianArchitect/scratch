#![allow(unused)]


use scratch::dag::*;
use scratch::gamepad::*;

use std::time::{Duration, Instant};

use gilrs::GamepadId;

macro_rules! print_expr {
    ($exp:expr) => {
        println!("{} == {}", stringify!($exp), $exp);
    };
}

macro_rules! prototype {
    ($($_:tt)*) => {};
}

prototype!(123 i32->u32 456);
prototype!(i32=>u32; (123) + (456));

macro_rules! check_overflow {
    (@internal; $lhs_ty:ident, $rhs_ty:ident, $lhs:expr, $op:tt, $rhs:expr) => {

    };
    ($lhs_ty:ident=>$rhs_ty:ident; $lhs:literal $op:tt $rhs:literal) => {

    };
    ($lhs_ty:ident=>$rhs_ty:ident; $lhs:literal $op:tt $rhs:ident) => {

    };
    ($lhs_ty:ident=>$rhs_ty:ident; $lhs:literal $op:tt ($rhs:expr)) => {

    };
    ($lhs_ty:ident=>$rhs_ty:ident; $lhs:ident $op:tt $rhs:literal) => {

    };
    ($lhs_ty:ident=>$rhs_ty:ident; $lhs:ident $op:tt $rhs:ident) => {

    };
    ($lhs_ty:ident=>$rhs_ty:ident; $lhs:ident $op:tt ($rhs:expr)) => {

    };
    ($lhs_ty:ident=>$rhs_ty:ident; ($lhs:expr) $op:tt $rhs:literal) => {

    };
    ($lhs_ty:ident=>$rhs_ty:ident; ($lhs:expr) $op:tt $rhs:ident) => {

    };
    ($lhs_ty:ident=>$rhs_ty:ident; ($lhs:expr) $op:tt ($rhs:expr)) => {
        check_overflow!(@internal; $lhs_ty, $rhs_ty, $lhs, $op, $rhs);
    };
}

check_overflow!(i32=>u32; (123 + 456) + (789 + 101112));

pub const fn i32_range_to_u32(i32_value: i32) -> u32 {
    (i32_value as u32) ^ 0x8000_0000
}

pub const fn u32_to_i32_range(u32_value: u32) -> i32 {
    (u32_value ^ 0x8000_0000) as i32
}

pub const fn is_zst<T>() -> bool {
    std::mem::size_of::<T>() == 0
}

#[track_caller]
pub const fn assert_zst<T>() {
    assert_sizeof::<T, 0>();
}

#[track_caller]
pub const fn assert_sizeof<T, const SIZE: usize>() {
    const {
        assert!(std::mem::size_of::<T>() == SIZE, "Size mismatch.");
    }
}

pub const fn create_fn<T: std::fmt::Display>(value: T) -> impl FnOnce() {
    move || println!("{}", value)
}

pub fn call_thrice<F: FnMut() -> ()>(mut f: F) {
    for _ in 0..3 {
        f();
    }
}

pub struct FnIterator<T, F: FnMut() -> Option<T>> {
    function: F,
}

impl<T, F: FnMut() -> Option<T>> Iterator for FnIterator<T, F> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        (self.function)()
    }
}

pub fn iter_fn<T, F: FnMut() -> Option<T>>(f: F) -> FnIterator<T, F> {
    FnIterator { function: f }
}

#[derive(Debug, Clone)]
pub struct Assertion<T, M: std::fmt::Display, F: Fn(T) -> bool> {
    function: F,
    msg: M,
    phantom: std::marker::PhantomData<T>,
}

pub const fn assertion<T, M: std::fmt::Display, F: Fn(T) -> bool>(msg: M, f: F) -> Assertion<T, M, F> {
    Assertion { function: f, msg, phantom: std::marker::PhantomData }
}

impl<T, M: std::fmt::Display, F: Fn(T) -> bool> Assertion<T, M, F> {
    #[track_caller]
    pub fn assert(&self, input: T) {
        assert!((self.function)(input), "{}", self.msg)
    }
}

macro_rules! dummy {
    ($name:ident) => {
        pub struct $name;
        impl $name {
            pub fn new() -> Self { Self }
        }
    };
    ($($name:ident),*$(,)?) => {
        $(
            dummy!($name);
        )*
    };
}

dummy!(
    Line, Grid2, Grid3, Grid4,
    OffsetLine, OffsetGrid2, OffsetGrid3, OffsetGrid4,
    ScrollLine, ScrollGrid2, ScrollGrid3, ScrollGrid4,
);

struct Dropper(&'static str, usize);

impl Drop for Dropper {
    fn drop(&mut self) {
        println!("Dropper({}, {})", self.0, self.1);
    }
}

pub struct Anon<T>(T);

macro_rules! anon {
    ($($name:ident : $value:expr),*$(,)?) => {
        {
            #[allow(non_camel_case_types)]
            struct Anonymous<$($name,)*> {
                $(
                    $name:$name,
                )*
            }
            #[allow(non_camel_case_types)]
            impl<$($name,)*> From<($($name,)*)> for Anonymous<$($name,)*> {
                fn from(($($name,)*): ($($name,)*)) -> Self {
                    Self {
                        $(
                            $name,
                        )*
                    }
                }
            }

            #[allow(non_camel_case_types)]
            impl<$($name,)*> Into<($($name,)*)> for Anonymous<$($name,)*> {
                fn into(self) -> ($($name,)*) {
                    (
                        $(
                            self.$name,
                        )*
                    )
                }
            }
            Anonymous {
                $(
                    $name: $value,
                )*
            }
        }
    };
}

pub struct Reacharound<T, const SIZE: usize> {
    array: [T; SIZE],
    wrap_index: usize,
}

impl<T, const SIZE: usize> Reacharound<T, SIZE> {
    pub fn new(array: [T; SIZE]) -> Self {
        Self {
            array,
            wrap_index: 0,
        }
    }

    fn wrapped_index(&self, index: usize) -> usize {
        ((index % SIZE) + self.wrap_index) % SIZE
    }

    pub fn get(&self, index: usize) -> &T {
        &self.array[self.wrapped_index(index)]
    }

    pub fn get_mut(&mut self, index: usize) -> &mut T {
        &mut self.array[self.wrapped_index(index)]
    }

    // pub fn push(&mut self, value: T) -> T {

    // }
}

impl<T, const SIZE: usize> std::ops::Index<usize> for Reacharound<T, SIZE> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        self.get(index)
    }
}

impl<T, const SIZE: usize> std::ops::IndexMut<usize> for Reacharound<T, SIZE> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.get_mut(index)
    }
}

// impl Octree16Level {
// }

// impl FooBar {
//     pub fn new() -> Self {
//         Self {
//             root: [0; 8]
//         }
//     }

//     pub fn sub_index(level: Octree16Level, x: u8, y: u8, z: u8) -> u8 {
//         // order: yzx
//         let shift = level as i32;
//         let bit_mask = 1u8 << (level as u32);
//         ((x & bit_mask) >> shift) | (((z & bit_mask) >> shift) << 1) | (((y & bit_mask) >> shift) << 2)
//     }

//     pub fn get(&self, x: u8, y: u8, z: u8) -> bool {
//         let root_index = Self::sub_index(Octree16Level::Root, x, y, z);
//         unsafe {
//             self.root[root_index as usize] != 0 && 
//         }
//     }
// }

// Depth0, 16x16x16
// Depth1, 8x8x8
// Depth2, 4x4x4
// Depth3, 2x2x2

pub trait Bitter {
    fn get_bit(self, index: u8) -> bool;
    fn replace_bit(&mut self, index: u8, on: bool) -> bool;
    fn set_bit(&mut self, index: u8, on: bool) -> bool;
}

impl Bitter for u8 {
    fn get_bit(self, index: u8) -> bool {
        self >> index & 1 == 1
    }

    // Replace the bit. Returns the old value.
    fn replace_bit(&mut self, index: u8, on: bool) -> bool {
        let old = *self >> index & 1 == 1;
        if on {
            *self = *self | (1 << index);
        } else {
            *self = *self & !(1 << index);
        }
        old
    }

    /// Set the bit. Returns true if the resulting value is non-zero.
    fn set_bit(&mut self, index: u8, on: bool) -> bool {
        if on {
            *self = *self | (1 << index);
        } else {
            *self = *self & !(1 << index);
        }
        *self != 0
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Depth3 {
    mask: u8,
}

impl Depth3 {
    pub fn new(mask: u8) -> Self {
        Self {
            mask
        }
    }

    pub fn new_empty() -> Self {
        Self {
            mask: 0b00000000,
        }
    }

    pub fn new_full() -> Self {
        Self {
            mask: 0b11111111
        }
    }

    pub fn index(x: u8, y: u8, z: u8) -> u8 {
        (x & 1) | ((z & 1) << 1) | ((y & 1) << 2)
    }

    pub fn get(self, x: u8, y: u8, z: u8, depth_counter: &mut u32) -> bool {
        let index = Self::index(x, y, z);
        *depth_counter = 3;
        self.mask.get_bit(index)
    }

    /// Set the bit, return true if the resulting mask is non-zero.
    pub fn set(&mut self, x: u8, y: u8, z: u8, on: bool) -> bool {
        let index = Self::index(x, y, z);
        self.mask.set_bit(index, on)
    }
}

#[derive(Debug, Default, Clone)]
pub struct Depth2 {
    mask: u8,
    nodes: [Depth3; 8],
}

impl Depth2 {
    pub fn get(&self, x: u8, y: u8, z: u8, depth_counter: &mut u32) -> bool {
        let index = Depth3::index(x>>1, y>>1, z>>1);
        *depth_counter = 2;
        self.mask.get_bit(index)
        &&
        self.nodes[index as usize].get(x, y, z, depth_counter)
    }

    /// Set bit, returning if the resulting mask is non-zero.
    pub fn set(&mut self, x: u8, y: u8, z: u8, on: bool) -> bool {
        let index = Depth3::index(x>>1, y>>1, z>>1);
        self.mask.set_bit(index, self.nodes[index as usize].set(x, y, z, on))
    }
}

#[derive(Debug, Default, Clone)]
pub struct Depth1 {
    mask: u8,
    nodes: [Depth2; 8],
}

impl Depth1 {
    pub fn get(&self, x: u8, y: u8, z: u8, depth_counter: &mut u32) -> bool {
        let index = Depth3::index(x>>2, y>>2, z>>2);
        *depth_counter = 1;
        self.mask.get_bit(index)
        &&
        self.nodes[index as usize].get(x, y, z, depth_counter)
    }

    /// Set bit, returning if the resulting mask is non-zero.
    pub fn set(&mut self, x: u8, y: u8, z: u8, on: bool) -> bool {
        let index = Depth3::index(x>>2, y>>2, z>>2);
        self.mask.set_bit(index, self.nodes[index as usize].set(x, y, z, on))
    }
}

#[derive(Debug, Default, Clone)]
pub struct Depth0 {
    mask: u8,
    nodes: [Depth1; 8],
}

impl Depth0 {
    pub fn get(&self, x: u8, y: u8, z: u8, depth_counter: &mut u32) -> bool {
        let index = Depth3::index(x>>3, y>>3, z>>3);
        *depth_counter = 0;
        self.mask.get_bit(index)
        &&
        self.nodes[index as usize].get(x, y, z, depth_counter)
    }

    // Set bit, returning if the resulting mask is non-zero.
    pub fn set(&mut self, x: u8, y: u8, z: u8, on: bool) -> bool {
        let index = Depth3::index(x>>3, y>>3, z>>3);
        self.mask.set_bit(index, self.nodes[index as usize].set(x, y, z, on))
    }
}

pub fn same_type<A: std::any::Any, B: std::any::Any>(a: &A, b: &B) -> bool {
    std::any::TypeId::of::<A>() == std::any::TypeId::of::<B>()
}

fn main() {
    let f = || {
        println!("hello, world");
    };
    let f2 = || {
        println!("This is a test.");
    };
    assert!(same_type(&f, &f));
    assert!(!same_type(&f, &f2));
    return;
    let mut octree = Depth0::default();
    let depth_counter = &mut 0;
    assert!(!octree.get(15, 15, 15, depth_counter));
    assert_eq!(*depth_counter, 0);
    assert!(octree.set(15, 15, 15, true));
    *depth_counter = 0;
    assert!(octree.get(15, 15, 15, depth_counter));
    assert_eq!(*depth_counter, 3);
    assert!(!octree.set(15, 15, 15, false));
    let depth_counter = &mut 0;
    assert!(!octree.get(15, 15, 15, depth_counter));
    assert_eq!(*depth_counter, 0);

    return;

    struct Zero;

    let z1 = Box::new(Zero);
    let z2 = Box::new(Zero);
    unsafe {
        let z1ptr = Box::leak(z1) as *mut Zero;
        let z2ptr = Box::leak(z2) as *mut Zero;
        let z1 = Box::from_raw(z1ptr);
        let z2 = Box::from_raw(z2ptr);
        assert_eq!(z1ptr, z2ptr);
    }

    return;
    // oijfsd

    use gilrs::{
        Gilrs,
        Button,
        // Axis,
        // Gamepad,
    };

    

    let mut girls = Gilrs::new().expect("Failed to get gamepad or something, idk.");

    let mut recent_buttons = [Button::C, Button::C, Button::C];
    let mut recent_wrap = 0usize;

    'event_loop: loop {
        let loop_start_time = Instant::now();
        while let Some(event) = girls.next_event() {
            match event.event {
                gilrs::EventType::ButtonPressed(button, code) => {
                    println!("Button pressed: {button:?}, code: {code:?}, id: {}", event.id);
                    if button == Button::Start {
                        break 'event_loop;
                    }
                },
                gilrs::EventType::ButtonRepeated(button, code) => {
                    // println!("Button repeated: {button:?}, code: {code:?}");

                },
                gilrs::EventType::ButtonReleased(button, code) => {
                    // println!("Button released: {button:?}, code: {code:?}");
                },
                gilrs::EventType::ButtonChanged(button, _, code) => {
                    // println!("Button changed: {button:?}, code: {code:?}");
                },
                gilrs::EventType::AxisChanged(axis, value, code) => {
                    match axis {
                        gilrs::Axis::LeftStickX => (),
                        gilrs::Axis::LeftStickY => (),
                        gilrs::Axis::LeftZ => (),
                        gilrs::Axis::RightStickX => (),
                        gilrs::Axis::RightStickY => (),
                        gilrs::Axis::RightZ => (),
                        gilrs::Axis::DPadX => (),
                        gilrs::Axis::DPadY => (),
                        gilrs::Axis::Unknown => (),
                    }
                    println!("axis changed: {axis:?} -> {value:.3}, code: {code:?}");
                },
                gilrs::EventType::Connected => {
                    println!("Connected");
                },
                gilrs::EventType::Disconnected => {
                    println!("Disconnected");
                },
                gilrs::EventType::Dropped => {
                    // print!("Dropped");
                },
                gilrs::EventType::ForceFeedbackEffectCompleted => {
                    // print!("ForceFeedbackEffectComplete");
                },
                _ => (),
            }
        }
        spin_sleep::sleep_until(loop_start_time + Duration::from_secs_f64(1.0/120.0));
    }

    return;

    let mut locals = anon!(
        test: "Hello, world",
        cool: anon!(
            beans: "Whoa!",
            why_is_this_not_builtin: "?",
            nothing: None,
        ),
    );
    locals.cool.beans = "This is a test.";
    locals.cool.nothing = Some(3);

    println!("{}", locals.cool.beans);

    drop(locals);

    return;
    // create_fn("Hello, world")();
    // print_expr!(i32::MIN);
    // print_expr!(u32_to_i32_range(u32::MIN));
    // print_expr!(i32::MAX);
    // print_expr!(u32_to_i32_range(u32::MAX));
    // print_expr!(u32::MIN);
    // print_expr!(i32_range_to_u32(i32::MIN));
    // print_expr!(u32::MAX);
    // print_expr!(i32_range_to_u32(i32::MAX));
    // macro_rules! assert_expr_msg {
    //     ($expr:expr) => {
    //         assert!($expr, "{}", stringify!($expr));
    //     };
    // }
    // assert_expr_msg!(u32_to_i32_range(u32::MIN) == i32::MIN);
    // assert_eq!(u32_to_i32_range(u32::MIN), i32::MIN);
    // assert_eq!(u32_to_i32_range(u32::MAX), i32::MAX);
    // assert_eq!(i32_range_to_u32(i32::MIN), u32::MIN);
    // assert_eq!(i32_range_to_u32(i32::MAX), u32::MAX);
    // println!("Success!");
    // macro_rules! view_expr {
    //     ($expr:expr) => {
    //         println!("{} = {}", stringify!($expr), $expr);
    //     };
    // }
    // view_expr!((-4i32).rem_euclid(3));
    // view_expr!((4i32).rem_euclid(3));
    // view_expr!((-4i32).rem_euclid(-3));
    // view_expr!((4i32).rem_euclid(-3));
    // println!("{}\n{}\n{}", u32_to_i32_range(0), min_offset, u32::MAX);
    // println!("hello, world");
}