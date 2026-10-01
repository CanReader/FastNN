//! An optimizer's internal state, in a form that can be written to disk.
//!
//! Saving only the weights is not enough to resume a run. Momentum and Adam's
//! two moment estimates are what turn a raw gradient into an update, and Adam's
//! step count drives the bias correction that divides them. Restore the weights
//! alone and the optimizer restarts at step 1: the correction factor is at its
//! most aggressive, so the first update after resuming is far larger than it
//! should be and can undo a great deal of training.

use std::collections::BTreeMap;

use crate::error::{Error, Result};
use crate::nn::Param;
use crate::tensor::Tensor;

/// Everything an optimizer needs to carry on exactly where it stopped.
#[derive(Default)]
pub struct OptimizerState {
    /// Per-parameter tensors, named `"<slot>/<index>"`.
    pub tensors: BTreeMap<String, Tensor>,
    /// Updates applied so far, for bias correction.
    pub steps: u64,
}

impl OptimizerState {
    pub fn new(steps: u64) -> OptimizerState {
        OptimizerState {
            tensors: BTreeMap::new(),
            steps,
        }
    }

    /// Record one per-parameter series, such as momentum or a moment estimate.
    ///
    /// Entries still `None` — parameters that have never received a gradient —
    /// are left out, and come back as `None`.
    pub fn put(&mut self, slot: &str, series: &[Option<Tensor>]) {
        for (index, value) in series.iter().enumerate() {
            if let Some(tensor) = value {
                self.tensors
                    .insert(format!("{slot}/{index}"), tensor.detach().cpu());
            }
        }
    }

    /// Read back a series recorded by [`put`](OptimizerState::put).
    ///
    /// `len` is the current parameter count; a mismatch with the saved run is
    /// reported rather than silently misaligning moments onto the wrong weights.
    pub fn take(&self, slot: &str, len: usize) -> Result<Vec<Option<Tensor>>> {
        let prefix = format!("{slot}/");
        if let Some(name) = self.tensors.keys().find(|k| {
            k.strip_prefix(&prefix)
                .and_then(|i| i.parse::<usize>().ok())
                .is_some_and(|i| i >= len)
        }) {
            return Err(Error::Checkpoint(format!(
                "optimizer state holds '{name}' but this optimizer has only {len} parameters"
            )));
        }
        Ok((0..len)
            .map(|i| self.tensors.get(&format!("{prefix}{i}")).cloned())
            .collect())
    }
}

/// Move a restored series onto the devices of the parameters it belongs to.
///
/// State is stored on the host so a checkpoint is portable between machines. The
/// optimizer then combines it with live gradients, which sit wherever the model
/// does — so without this a run resumed onto a GPU would hit a device mismatch on
/// its first step.
pub fn place(series: Vec<Option<Tensor>>, params: &[Param]) -> Vec<Option<Tensor>> {
    series
        .into_iter()
        .zip(params)
        .map(|(value, param)| value.map(|t| t.to(param.device())))
        .collect()
}
