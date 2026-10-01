//! Chaining layers.

use crate::tensor::Tensor;

use super::module::{scoped, Module};
use super::param::{Buffer, Param};

/// Runs its layers in order, feeding each output into the next.
///
/// ```
/// use fastnn::prelude::*;
///
/// let model = Sequential::new()
///     .add(Linear::new(784, 128))
///     .add(ReLU)
///     .add(Linear::new(128, 10));
///
/// assert_eq!(model.forward(&Tensor::randn(&[4, 784])).shape(), &[4, 10]);
/// ```
///
/// Parameter names combine the layer's index with its own name, so the second
/// `Linear` above owns `"2.weight"` and `"2.bias"`.
#[derive(Default)]
pub struct Sequential {
    layers: Vec<Box<dyn Module>>,
}

impl Sequential {
    pub fn new() -> Sequential {
        Sequential::default()
    }

    /// Append a layer, returning `self` so calls chain.
    // Named for the builder pattern, not arithmetic; `std::ops::Add` is unrelated.
    #[allow(clippy::should_implement_trait)]
    pub fn add(mut self, layer: impl Module + 'static) -> Sequential {
        self.layers.push(Box::new(layer));
        self
    }

    /// Append a layer to an existing model.
    pub fn push(&mut self, layer: impl Module + 'static) {
        self.layers.push(Box::new(layer));
    }

    pub fn len(&self) -> usize {
        self.layers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }
}

impl Module for Sequential {
    fn forward(&self, input: &Tensor) -> Tensor {
        self.layers
            .iter()
            .fold(input.clone(), |x, layer| layer.forward(&x))
    }

    fn named_parameters(&self) -> Vec<(String, Param)> {
        self.layers
            .iter()
            .enumerate()
            .flat_map(|(i, layer)| scoped(i, layer.named_parameters()))
            .collect()
    }

    fn named_buffers(&self) -> Vec<(String, Buffer)> {
        self.layers
            .iter()
            .enumerate()
            .flat_map(|(i, layer)| scoped(i, layer.named_buffers()))
            .collect()
    }

    fn set_training(&self, training: bool) {
        for layer in &self.layers {
            layer.set_training(training);
        }
    }
}
