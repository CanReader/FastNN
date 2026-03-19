use crate::tensor::Tensor;
use std::collections::HashMap;

/// The base trait for all neural network modules.
///
/// Every layer, model, or component implements this trait, providing
/// a unified interface for forward passes, parameter access, and
/// train/eval mode switching.
pub trait Module: Send + Sync {
    /// Forward pass — transforms input tensor(s) to output.
    fn forward(&self, input: &Tensor) -> Tensor;

    /// Return all learnable parameters.
    fn parameters(&self) -> Vec<Tensor> {
        Vec::new()
    }

    /// Return named parameters (name -> tensor).
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
