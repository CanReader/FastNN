//! An optimizer's internal state, in a form that can be written to disk.
//!
//! Optimizer state in a form that can be written to disk.
//!
//! Weights alone are not a resumable run: Adam's step count drives its bias
//! correction, and restarting at step 1 makes the first update far too big.

use std::collections::BTreeMap;

use crate::error::{Error, Result};
use crate::nn::Param;
use crate::tensor::Tensor;

/// What an optimizer needs to resume where it stopped.
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

    /// Record one per-parameter series. `None` entries are skipped and come back as `None`.
    pub fn put(&mut self, slot: &str, series: &[Option<Tensor>]) {
        for (index, value) in series.iter().enumerate() {
            if let Some(tensor) = value {
                self.tensors
                    .insert(format!("{slot}/{index}"), tensor.detach().cpu());
            }
        }
    }

    /// Read back a series from [`put`](OptimizerState::put). Errors if the saved
    /// run had more parameters than `len`.
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

/// Move restored state onto each parameter's device. It is saved on the host,
/// so skipping this breaks the first step of a run resumed on a GPU.
pub fn place(series: Vec<Option<Tensor>>, params: &[Param]) -> Vec<Option<Tensor>> {
    series
        .into_iter()
        .zip(params)
        .map(|(value, param)| value.map(|t| t.to(param.device())))
        .collect()
}
