use crate::tensor::Tensor;
use crate::nn::module::Module;

/// A container that chains modules sequentially.
pub struct Sequential {
    layers: Vec<Box<dyn Module>>,
}

impl Sequential {
    pub fn new() -> Self {
        Sequential { layers: Vec::new() }
    }

    /// Add a layer to the sequence.
    pub fn add(mut self, layer: impl Module + 'static) -> Self {
        self.layers.push(Box::new(layer));
        self
    }

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
        let mut x = input.clone();
        for layer in &self.layers {
            x = layer.forward(&x);
        }
        x
    }

    fn parameters(&self) -> Vec<Tensor> {
        self.layers.iter().flat_map(|l| l.parameters()).collect()
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        self.layers.iter_mut().flat_map(|l| l.parameters_mut()).collect()
    }

    fn train(&mut self) {
        for layer in &mut self.layers {
            layer.train();
        }
    }

    fn eval(&mut self) {
        for layer in &mut self.layers {
            layer.eval();
        }
    }
}

impl Default for Sequential {
    fn default() -> Self {
        Self::new()
    }
}
