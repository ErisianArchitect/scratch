
pub struct DiffPlot {
    width: u32,
    height: u32,
    plot: Box<[u64]>,
    bottom: u64,
}

impl DiffPlot {
    pub fn new(width: u32, height: u32) -> Self {
        let plot = (0..width*height).map(|_| 0u64).collect::<Box<_>>();
        Self {
            plot,
            width,
            height,
            bottom: 0,
        }
    }

    fn offset_index(&self, x: u32, y: u32) -> usize {
        assert!(x < self.width && y < self.height);
        let x = x as usize;
        let y = y as usize;
        let width = self.width as usize;
        y * width + x
    }

    pub fn is_bottom(&self, x: u32, y: u32) -> bool {
        let index = self.offset_index(x, y);
        let n = self.plot[index];
        n == self.bottom
    }

    pub fn is_zero(&self, x: u32, y: u32) -> bool {
        let index = self.offset_index(x, y);
        let n = self.plot[index];
        n == 0
    }

    pub fn cmp_bottom(&self, x: u32, y: u32) -> std::cmp::Ordering {
        let index = self.offset_index(x, y);
        let n = self.plot[index];
        n.cmp(&self.bottom)
    }

    pub fn get(&self, x: u32, y: u32) -> bool {
        let index = self.offset_index(x, y);
        let n = self.plot[index];
        n > self.bottom
    }

    pub fn get_raw(&self, x: u32, y: u32) -> u64 {
        let index = self.offset_index(x, y);
        self.plot[index]
    }

    pub fn set(&mut self, x: u32, y: u32, value: bool) -> bool {
        let index = self.offset_index(x, y);
        let old = self.plot[index];
        let bottom = self.bottom;
        self.plot[index] = bottom + (value as u64);
        old > bottom
    }

    pub fn clear(&mut self) {
        assert_ne!(self.bottom, u64::MAX, "How did you clear the plot that many times?");
        self.bottom += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn diffplot_test() {
        let mut plot = DiffPlot::new(1024*16, 1024*16);
        plot.set(0, 0, false);
        assert!(!plot.get(0, 0));
        plot.set(0, 0, true);
        assert!(plot.get(0, 0));
        plot.clear();
        assert!(!plot.get(0, 0));

        let start = std::time::Instant::now();
        for _ in 0..1024 {
            plot.set(0, 0, true);
            assert!(plot.get(0, 0));
            plot.clear();
            assert!(!plot.get(0, 0));
            println!("Success!");
        }
        let elapsed = start.elapsed();
        println!("Time: {elapsed:?}");
    }
}