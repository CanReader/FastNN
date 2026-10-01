//! The random source behind weight init, dropout, and shuffling.
//!
//! One generator per thread. [`manual_seed`] reseeds the calling thread, so a
//! single-threaded training script is reproducible end to end.

use std::cell::RefCell;

use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use rand_distr::StandardNormal;

thread_local! {
    static RNG: RefCell<StdRng> = RefCell::new(rand::make_rng());
}

/// Reseed this thread's generator so a run reproduces exactly.
pub fn manual_seed(seed: u64) {
    RNG.with(|r| *r.borrow_mut() = StdRng::seed_from_u64(seed));
}

/// Borrow the generator. Use this for distributions the helpers below don't cover.
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

/// `n` draws from Bernoulli(`p`), as a bitmask of `true` = kept.
pub fn bernoulli(n: usize, p: f32) -> Vec<bool> {
    with_rng(|rng| (0..n).map(|_| rng.random::<f32>() < p).collect())
}

/// Shuffle in place, using this thread's generator so `manual_seed` covers it.
pub fn shuffle<T>(items: &mut [T]) {
    use rand::seq::SliceRandom;
    with_rng(|rng| items.shuffle(rng));
}
