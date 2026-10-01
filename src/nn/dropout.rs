//! Dropout regularization.

use std::sync::atomic::{AtomicBool, Ordering};

use crate::rng;
use crate::tensor::Tensor;

use super::module::Module;

/// Zero each element with probability `p` during training, scaling the survivors
/// by `1/(1-p)` so the expected activation is unchanged.
///
/// That rescaling is what lets eval mode be a no-op: the network sees the same
/// average signal either way. Implemented as a multiply by a random mask, so the
/// gradient is routed to exactly the units that survived.
pub struct Dropout {
    p: f32,
    training: AtomicBool,
}

impl Dropout {
    /// Panics unless `p` is in `[0, 1)` — `p = 1` would zero everything.
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
