use crate::tensor::Tensor;
use std::collections::HashMap;

/// The base trait for all neural network modules.
///
/// Every layer, model, or component implements this trait. `forward` is the only
/// required method; the rest have sensible defaults.
///
/// ## Parameters and gradients
///
/// - `parameters()` returns **clones** of each learnable tensor. These clones share
///   the same `grad` Arc as the original (Rust's `Arc<Mutex<...>>` interior
///   mutability) but **not** the same CPU storage, so they are useful for reading
///   gradients and inspecting shapes but NOT for applying weight updates.
/// - `parameters_mut()` returns `&mut Tensor` references directly into the layer,
///   which is what optimizers use to mutate weights in place.
pub trait Module: Send + Sync {
    /// Forward pass — transforms input tensor(s) to output.
    fn forward(&self, input: &Tensor) -> Tensor;

    /// Return clones of all learnable parameters. Grad Arcs are shared with the
    /// originals, so reading `.grad()` on a clone sees the same accumulated grad.
    fn parameters(&self) -> Vec<Tensor> {
        Vec::new()
    }

    /// Return mutable references to all learnable parameters. Optimizers use this
    /// to update weights in place.
    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        Vec::new()
    }

    /// Return named parameters (name -> tensor clone).
    fn named_parameters(&self) -> HashMap<String, Tensor> {
        HashMap::new()
    }

    /// Set the module to training mode.
    fn train(&mut self) {}

    /// Set the module to evaluation mode.
    fn eval(&mut self) {}

    /// Whether the module is in training mode.
    fn is_training(&self) -> bool {
        true
    }

    /// Count total number of learnable parameters.
    fn num_parameters(&self) -> usize {
        self.parameters().iter().map(|p| p.numel()).sum()
    }

    /// Zero all parameter gradients.
    fn zero_grad(&self) {
        for p in self.parameters() {
            p.zero_grad();
        }
    }

    /// Move all parameters to a device.
    fn to_device(&mut self, _device: crate::tensor::Device) {}
}
