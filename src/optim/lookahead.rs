//! Lookahead (Zhang et al., 2019): slow weights that follow any inner optimizer.

use crate::error::Result;
use crate::nn::Param;
use crate::tensor::Tensor;

use super::state::place;
use super::{Optimizer, OptimizerState};

/// Wrap any optimizer with slow-weight averaging.
///
/// ```
/// # use fastnn::prelude::*;
/// # let model = Sequential::new().add(Linear::new(4, 2));
/// let mut opt = Lookahead::new(Adam::new(model.parameters(), 1e-3), 5, 0.5);
/// ```
pub struct Lookahead<O: Optimizer> {
    inner: O,
    slow: Vec<Option<Tensor>>,
    k: usize,
    alpha: f32,
    since_sync: usize,
}

impl<O: Optimizer> Lookahead<O> {
    /// Sync every `k` inner steps, moving the slow weights `alpha` of the way
    /// to the fast ones.
    pub fn new(inner: O, k: usize, alpha: f32) -> Lookahead<O> {
        assert!(k >= 1, "lookahead needs k >= 1");
        assert!(
            (0.0..=1.0).contains(&alpha),
            "alpha must be in [0, 1], got {alpha}"
        );
        // Seeding lazily at the first sync would copy weights that already took
        // k fast steps and make that sync a no-op.
        let slow = inner
            .parameters()
            .iter()
            .map(|p| Some(p.value().detach()))
            .collect();
        Lookahead {
            inner,
            slow,
            k,
            alpha,
            since_sync: 0,
        }
    }
}

impl<O: Optimizer> Optimizer for Lookahead<O> {
    fn step(&mut self) {
        self.inner.step();
        self.since_sync += 1;
        if self.since_sync < self.k {
            return;
        }
        self.since_sync = 0;

        for (index, param) in self.inner.parameters().iter().enumerate() {
            let fast = param.value();
            // No slow copy yet if the state was loaded without one.
            let slow = match &self.slow[index] {
                Some(slow) => slow.add(&fast.sub(slow).mul_scalar(self.alpha)),
                None => fast.clone(),
            };
            param.set_value(slow.clone());
            self.slow[index] = Some(slow);
        }
    }

    fn parameters(&self) -> &[Param] {
        self.inner.parameters()
    }

    fn lr(&self) -> f32 {
        self.inner.lr()
    }

    fn set_lr(&mut self, lr: f32) {
        self.inner.set_lr(lr);
    }

    fn state(&self) -> OptimizerState {
        let mut state = self.inner.state();
        state.put("lookahead_slow", &self.slow);
        state
    }

    fn load_state(&mut self, state: OptimizerState) -> Result<()> {
        self.slow = place(
            state.take("lookahead_slow", self.inner.parameters().len())?,
            self.inner.parameters(),
        );
        self.inner.load_state(state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optim::SGD;

    #[test]
    fn sync_lands_exactly_at_the_interpolated_point() {
        let param = Param::new(Tensor::from_vec(vec![0.0], &[1]));
        let mut opt = Lookahead::new(SGD::new(vec![param.clone()], 0.1), 2, 0.5);

        for _ in 0..2 {
            param.set_grad(Tensor::from_vec(vec![1.0], &[1]));
            opt.step();
        }
        // Fast weights walk to -0.2, so the sync lands at 0.5 * -0.2.
        let got = param.value().to_vec()[0];
        assert!(
            (got - -0.1).abs() < 1e-6,
            "expected −0.1 after sync, got {got}"
        );
    }
}
