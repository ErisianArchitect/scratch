#![allow(unused)]

use std::collections::HashMap;
use std::sync::atomic::AtomicU32;

// color pattern

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Token(pub u32);

impl std::fmt::Display for Token {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Token {
    pub const MIN: Token = Token::new(u32::MIN);
    pub const MAX: Token = Token::new(u32::MAX);
    pub const ZERO: Token = Token::MIN;

    pub const fn new(id: u32) -> Self {
        Self(id)
    }

    pub fn fetch_add(&mut self) -> Self {
        let result = *self;
        self.0 += 1;
        result
    }

}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RunLength {
    token: Token,
    length: u32,
}

impl RunLength {
    pub const fn new(token: Token) -> Self {
        Self {
            token,
            length: 1,
        }
    }
}

impl std::fmt::Display for RunLength {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}, {})", self.token, self.length)
    }
}

pub struct ColorPattern {
    tokens: Vec<Token>,
    run_length_encoding: Vec<RunLength>,
}

pub fn color_pattern<T: std::hash::Hash + Eq, It: IntoIterator<Item = T>>(input: It) -> ColorPattern {
    let mut iterator = input.into_iter();
    let mut pattern = match iterator.size_hint() {
        (_, Some(upper)) => Vec::with_capacity(upper),
        (lower, None) => Vec::with_capacity(lower),
    };
    let mut token_counter = Token::ZERO;
    let mut map = HashMap::new();
    let mut cur_token = token_counter;
    let mut run_length = 0;
    let mut runs = Vec::new();

    for value in iterator {
        let token = *map.entry(value).or_insert_with(|| token_counter.fetch_add());
        if token == cur_token {
            run_length += 1;
        } else {
            runs.push(RunLength { token: cur_token, length: run_length });
            cur_token = token;
            run_length = 1;
        }
        pattern.push(token);
    }

    runs.push(RunLength { token: cur_token, length: run_length });

    ColorPattern {
        tokens: pattern,
        run_length_encoding: runs,
    }
}

#[cfg(test)]
mod testing_sandbox {
    // TODO: Remove this sandbox when it is no longer in use.
    use super::*;

    fn print_pattern(pattern: &ColorPattern) {
        let mut colit = pattern.tokens.iter();
            if let Some(first) = colit.next() {
                print!("Pattern: {first}");
            }
            for token in colit {
                print!(", {token}");
            }
            println!();
            let mut rle_it = pattern.run_length_encoding.iter();
            if let Some(first) = rle_it.next() {
                print!("Run Length Encoding: {first}");
            }
            for run in rle_it {
                print!(", {run}");
            }
            println!();
        if pattern.tokens.len() > pattern.run_length_encoding.len() {
            let rle_pattern = color_pattern(&pattern.run_length_encoding);
            print_pattern(&rle_pattern);
        }
    }

    #[test]
    fn sandbox() {
        let pattern = "abbc abbc abeefeebc";
        let colored = color_pattern(pattern.chars());
        println!("Input: \"{pattern}\"");
        print_pattern(&colored);
    }
}