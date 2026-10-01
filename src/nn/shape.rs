//! Shape-changing layers, for use inside [`Sequential`](super::Sequential).

use crate::tensor::Tensor;

use super::module::Module;

/// Collapse everything after the batch dimension into one.
///
/// The bridge from a convolution stack to a classifier head:
/// `[N, C, H, W] → [N, C·H·W]`.
pub struct Flatten {
    from: usize,
}

impl Flatten {
    /// Keep dimension 0 as the batch, flatten the rest.
    pub fn new() -> Flatten {
        Flatten { from: 1 }
    }

    /// Keep dimensions before `from`, flatten from there on.
    pub fn from(from: usize) -> Flatten {
        Flatten { from }
    }
}

impl Default for Flatten {
    fn default() -> Self {
        Flatten::new()
    }
}

impl Module for Flatten {
    fn forward(&self, input: &Tensor) -> Tensor {
        input.flatten_from(self.from)
    }
}

/// Reshape to a fixed shape. `-1` infers one dimension, which is how you keep a
/// variable batch size.
pub struct Reshape {
    dims: Vec<i64>,
}

impl Reshape {
    pub fn new(dims: &[i64]) -> Reshape {
        Reshape {
            dims: dims.to_vec(),
        }
    }
}

impl Module for Reshape {
    fn forward(&self, input: &Tensor) -> Tensor {
        input.reshape(&self.dims)
    }
}
