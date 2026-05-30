use crate::tensor::Tensor;
use crate::nn::module::Module;

/// Flattens a contiguous range of dimensions into one.
/// Default (`Flatten::new()`) flattens dims 1..end, keeping the batch dimension.
pub struct Flatten {
    start_dim: usize,
    end_dim: isize, // negative means from end
}

impl Flatten {
    /// Flatten all dims except batch (dims 1 onward).
    pub fn new() -> Self {
        Flatten { start_dim: 1, end_dim: -1 }
    }

    pub fn range(start_dim: usize, end_dim: isize) -> Self {
        Flatten { start_dim, end_dim }
    }
}

impl Default for Flatten {
    fn default() -> Self { Self::new() }
}

impl Module for Flatten {
    fn forward(&self, input: &Tensor) -> Tensor {
        let shape = input.shape();
        let ndim = shape.len();

        let end = if self.end_dim < 0 {
            ((ndim as isize) + self.end_dim) as usize
        } else {
            (self.end_dim as usize).min(ndim - 1)
        };

        let mut new_shape: Vec<i64> = shape[..self.start_dim].iter().map(|&s| s as i64).collect();
        let flat: usize = shape[self.start_dim..=end].iter().product();
        new_shape.push(flat as i64);
        for &s in &shape[end + 1..] {
            new_shape.push(s as i64);
        }

        // reshape is autograd-tracked, so Flatten participates in the graph for free
        input.reshape(&new_shape)
    }
}
