//! Shape-changing layers, for use inside [`Sequential`](super::Sequential).

use crate::tensor::Tensor;

use super::module::Module;

/// Flatten everything after the batch dimension, `[N, C, H, W]` to `[N, C·H·W]`.
pub struct Flatten {
    from: usize,
}

impl Flatten {
    /// Flatten from dimension 1.
    pub fn new() -> Flatten {
        Flatten { from: 1 }
    }

    /// Flatten from dimension `from`.
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

/// Reshape to a fixed shape. `-1` infers one dimension.
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
