#![allow(unused)]
use std::{cell::UnsafeCell, ptr::NonNull, sync::atomic::AtomicUsize};


pub struct SharedInner<T> {
    value: UnsafeCell<T>,
    count: AtomicUsize,
}

impl<T> SharedInner<T> {
    pub fn new(value: T) -> Self {
        Self {
            value: UnsafeCell::new(value),
            count: AtomicUsize::new(1),
        }
    }

    pub fn strong_count(&self) -> usize {
        self.count.load(std::sync::atomic::Ordering::Relaxed)
    }

    fn increment_count(&self) -> usize {
        self.count.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    }

    fn decrement_count(&self) -> usize {
        self.count.fetch_sub(1, std::sync::atomic::Ordering::Relaxed)
    }
}

impl<T> std::ops::Deref for SharedInner<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        unsafe {
            &*self.value.get()
        }
    }
}

impl<T> std::ops::DerefMut for SharedInner<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe {
            &mut *self.value.get()
        }
    }
}

pub struct Shared<T> {
    shard: NonNull<SharedInner<T>>,
}

impl<T> Shared<T> {
    pub fn new(value: T) -> Self {
        unsafe {
            let shard = SharedInner::new(value);
            let layout = std::alloc::Layout::for_value(&shard);
            let ptr = std::alloc::alloc(layout).cast::<SharedInner<T>>();
            ptr.write(shard);
            let shard = NonNull::new(ptr).expect("Allocation failed.");
            Self {
                shard,
            }
        }
    }

    pub fn strong_count(&self) -> usize {
        unsafe {
            self.shard.as_ref().strong_count()
        }
    }
}

impl<T> std::ops::Deref for Shared<T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        unsafe {
            self.shard.as_ref().deref()
        }
    }
}

impl<T> std::ops::DerefMut for Shared<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe {
            self.shard.as_mut().deref_mut()
        }
    }
}

impl<T> Drop for Shared<T> {
    fn drop(&mut self) {
        unsafe {
            let old_count = self.shard.as_mut().decrement_count();
            if old_count == 1 {
                let shard = self.shard.read();
                let layout = std::alloc::Layout::for_value(&shard);
                std::alloc::dealloc(self.shard.as_ptr().cast(), layout);
                drop(shard);
            }
        }
    }
}

impl<T> Clone for Shared<T> {
    fn clone(&self) -> Self {
        let shard = self.shard;
        unsafe {
            shard.as_ref().increment_count();
        }
        Self {
            shard
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn footgun_test() {

        struct Dropper(usize);
        impl Drop for Dropper {
            fn drop(&mut self) {
                println!("Dropper({}) dropped.", self.0);
                self.0 += 0;
            }
        }
        struct ManDrop<T> {
            value: T
        }
        impl<T> Drop for ManDrop<T> {
            fn drop(&mut self) {
            }
        }
        let man = ManDrop { value: Dropper(12345) };
        println!("Returning.");
        return;
        _=Shared::new(Dropper(69420));
        let footgun = Shared::new(Dropper(0));
        let mut mut_footgun = footgun.clone();
        let mut other_mut_footgun = mut_footgun.clone();
        let footguns: [Shared<Dropper>; 10] = std::array::from_fn(|_| footgun.clone());
        println!("Strong Count: {}", footgun.strong_count());
        println!("Dropper Index: {}", footgun.0);
        // mutation happens here.
        mut_footgun.0 = 1234;
        println!("Dropper Index: {}", footgun.0);
        other_mut_footgun.0 = 4321;
        println!("Dropper Index: {}", footgun.0);
        println!("Dropping mut_footgun");
        drop(mut_footgun);
        println!("Dropping other_mut_footgun");
        drop(other_mut_footgun);
        println!("Dropping footgun.");
        drop(footgun);
        println!("Dropping footguns.");
        drop(footguns);
        println!("Exiting scope.");
    }
}