
pub struct InlineStack<const SIZE: usize, T> {
    stack: [T; SIZE],
    length: usize,
}

impl<const SIZE: usize, T: Sized + 'static + Copy> InlineStack<SIZE, T> {
    #[inline(always)]
    pub const fn new(default: T) -> Self {
        Self {
            stack: [default; SIZE],
            length: 0,
        }
    }

    #[inline(always)]
    pub fn push(&mut self, value: T) {
        self.stack[self.length] = value;
        self.length += 1;
    }

    #[inline(always)]
    pub unsafe fn push_unchecked(&mut self, value: T) {
        *self.stack.get_unchecked_mut(self.length) = value;
        self.length = self.length.unchecked_add(1);
    }

    #[inline(always)]
    pub fn pop(&mut self) -> T {
        self.length -= 1;
        self.stack[self.length]
    }

    #[inline(always)]
    pub unsafe fn pop_unchecked(&mut self) -> T {
        self.length = self.length.unchecked_sub(1);
        *self.stack.get_unchecked(self.length)
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.length
    }
}