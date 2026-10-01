//! Checkpointing a run, not just a model.
//!
//! [`save`](super::save) writes weights, which is what you want for a finished
//! model. Resuming an interrupted run needs more: the optimizer's momentum and
//! moment estimates, its step count, and where the schedule had got to. Without
//! them the optimizer restarts cold and the first update after resuming lands
//! far harder than it should.
//!
//! ```no_run
//! # use fastnn::prelude::*;
//! # use fastnn::serialize::{save_training, load_training};
//! # let model = Sequential::new().add(Linear::new(4, 2));
//! # let total_steps = 10_000;
//! let mut opt = Adam::new(model.parameters(), 1e-3);
//!
//! // Pick up where the last run stopped, or start fresh.
//! let mut step = load_training(&model, &mut opt, "run.fdl").unwrap_or(0);
//!
//! while step < total_steps {
//!     // ... train ...
//!     step += 1;
//!     if step % 500 == 0 {
//!         save_training(&model, &opt, step, "run.fdl")?;
//!     }
//! }
//! # Ok::<(), fastnn::Error>(())
//! ```

use std::path::Path;

use crate::error::{Error, Result};
use crate::nn::Module;
use crate::optim::{Optimizer, OptimizerState};
use crate::tensor::Tensor;

use super::checkpoint::{load_tensors, save_tensors, Checkpoint};

/// Namespaces inside the file. `/` cannot collide with the `.` that separates
/// module paths in a parameter name.
const PARAM: &str = "param/";
const BUFFER: &str = "buffer/";
const OPTIM: &str = "optim/";
const STEP: &str = "meta/step";
const OPTIM_STEPS: &str = "meta/optim_steps";

/// Write model weights, optimizer state, and the step counter to `path`.
pub fn save_training(
    model: &dyn Module,
    optimizer: &dyn Optimizer,
    step: u64,
    path: impl AsRef<Path>,
) -> Result<()> {
    let mut file = Checkpoint::new();

    for (name, param) in model.named_parameters() {
        file.insert(format!("{PARAM}{name}"), param.value().cpu());
    }
    for (name, buffer) in model.named_buffers() {
        file.insert(format!("{BUFFER}{name}"), buffer.value().cpu());
    }

    let state = optimizer.state();
    for (name, tensor) in state.tensors {
        file.insert(format!("{OPTIM}{name}"), tensor);
    }
    file.insert(STEP.into(), Tensor::scalar(step as f32));
    file.insert(OPTIM_STEPS.into(), Tensor::scalar(state.steps as f32));

    save_tensors(&file, path)
}

/// Restore what [`save_training`] wrote, returning the step it stopped at.
///
/// Errors if `path` is a plain model checkpoint rather than a training one —
/// resuming from weights alone would silently reset the optimizer, which is the
/// exact failure this module exists to prevent.
pub fn load_training(
    model: &dyn Module,
    optimizer: &mut dyn Optimizer,
    path: impl AsRef<Path>,
) -> Result<u64> {
    let file = load_tensors(path)?;
    if !file.contains_key(STEP) {
        return Err(Error::Checkpoint(
            "not a training checkpoint — it has no optimizer state, so resuming \
             from it would restart the optimizer. Use load() if that is what you want."
                .into(),
        ));
    }

    for (name, param) in model.named_parameters() {
        let value = expect(&file, &format!("{PARAM}{name}"), &param.shape())?;
        param.set_value(value.to(param.device()));
    }
    for (name, buffer) in model.named_buffers() {
        let device = buffer.value().device();
        let value = expect(&file, &format!("{BUFFER}{name}"), buffer.value().shape())?;
        buffer.set_value(value.to(device));
    }

    let mut state = OptimizerState::new(scalar(&file, OPTIM_STEPS)? as u64);
    for (name, tensor) in &file {
        if let Some(slot) = name.strip_prefix(OPTIM) {
            state.tensors.insert(slot.to_string(), tensor.clone());
        }
    }
    optimizer.load_state(state)?;

    Ok(scalar(&file, STEP)? as u64)
}

fn expect(file: &Checkpoint, name: &str, shape: &[usize]) -> Result<Tensor> {
    let tensor = file
        .get(name)
        .ok_or_else(|| Error::Checkpoint(format!("checkpoint has no '{name}'")))?;
    if tensor.shape() != shape {
        return Err(Error::Checkpoint(format!(
            "'{name}' is {:?} in the file but {shape:?} in the model",
            tensor.shape()
        )));
    }
    Ok(tensor.clone())
}

fn scalar(file: &Checkpoint, name: &str) -> Result<f32> {
    file.get(name)
        .map(|t| t.item())
        .ok_or_else(|| Error::Checkpoint(format!("checkpoint has no '{name}'")))
}
