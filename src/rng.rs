//! The per-thread random source behind init, dropout, and shuffling.

use std::cell::RefCell;

use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use rand_distr::StandardNormal;

thread_local! {
    static RNG: RefCell<StdRng> = RefCell::new(rand::make_rng());
}

/// Reseed this thread's generator.
pub fn manual_seed(seed: u64) {
    RNG.with(|r| *r.borrow_mut() = StdRng::seed_from_u64(seed));
}

/// Borrow the generator for distributions the helpers don't cover.
pub fn with_rng<T>(f: impl FnOnce(&mut StdRng) -> T) -> T {
    RNG.with(|r| f(&mut r.borrow_mut()))
}

/// `n` samples from the uniform distribution on `[lo, hi)`.
pub fn uniform(n: usize, lo: f32, hi: f32) -> Vec<f32> {
    with_rng(|rng| (0..n).map(|_| rng.random_range(lo..hi)).collect())
}

/// `n` samples from the standard normal distribution.
pub fn normal(n: usize) -> Vec<f32> {
    with_rng(|rng| (0..n).map(|_| rng.sample(StandardNormal)).collect())
}

/// `n` Bernoulli(`p`) draws, `true` meaning kept.
pub fn bernoulli(n: usize, p: f32) -> Vec<bool> {
    with_rng(|rng| (0..n).map(|_| rng.random::<f32>() < p).collect())
}

/// Shuffle in place with this thread's generator.
pub fn shuffle<T>(items: &mut [T]) {
    use rand::seq::SliceRandom;
    with_rng(|rng| items.shuffle(rng));
}
