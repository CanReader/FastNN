//! Dropout regularization.

use std::sync::atomic::{AtomicBool, Ordering};

use crate::rng;
use crate::tensor::Tensor;

use super::module::Module;

/// Zero each element with probability `p` while training, scaling survivors by
/// `1/(1-p)` so eval mode can be a no-op.
pub struct Dropout {
    p: f32,
    training: AtomicBool,
}

impl Dropout {
    /// Panics unless `p` is in `[0, 1)`.
    pub fn new(p: f32) -> Dropout {
        assert!(
            (0.0..1.0).contains(&p),
            "dropout probability must be in [0, 1), got {p}"
        );
        Dropout {
            p,
            training: AtomicBool::new(true),
        }
    }

    pub fn probability(&self) -> f32 {
        self.p
    }
}

impl Module for Dropout {
    fn forward(&self, input: &Tensor) -> Tensor {
        if self.p == 0.0 || !self.training.load(Ordering::Relaxed) {
            return input.clone();
        }

        let keep = 1.0 - self.p;
        let scale = 1.0 / keep;
        let mask: Vec<f32> = rng::bernoulli(input.numel(), keep)
            .into_iter()
            .map(|kept| if kept { scale } else { 0.0 })
            .collect();

        input.mul(&Tensor::from_vec(mask, input.shape()).to(input.device()))
    }

    fn set_training(&self, training: bool) {
        self.training.store(training, Ordering::Relaxed);
    }
}
