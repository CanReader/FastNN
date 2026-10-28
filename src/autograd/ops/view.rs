//! Derivatives of the rearranging ops, each the inverse rearrangement.

use crate::autograd::Backward;
use crate::tensor::shape;
use crate::tensor::Tensor;

use super::reduce_to;

/// Reshape the gradient back to the original shape.
pub struct ReshapeBackward {
    pub shape: Vec<usize>,
}

impl Backward for ReshapeBackward {
    fn backward(&self, grad: &Tensor) -> Vec<Tensor> {
        let dims: Vec<i64> = self.shape.iter().map(|&d| d as i64).collect();
        vec![grad.reshape(&dims)]
    }
    fn name(&self) -> &'static str {
        "Reshape"
    }
}

/// Apply the inverse permutation.
pub struct PermuteBackward {
    pub order: Vec<usize>,
}

impl Backward for PermuteBackward {
    fn backward(&self, grad: &Tensor) -> Vec<Tensor> {
        let mut inverse = vec![0usize; self.order.len()];
        for (position, &dim) in self.order.iter().enumerate() {
            inverse[dim] = position;
        }
        vec![grad.permute(&inverse)]
    }
    fn name(&self) -> &'static str {
        "Permute"
    }
}

/// The gradient of a copy is the sum over the copies.
pub struct ExpandBackward {
    pub shape: Vec<usize>,
}

impl Backward for ExpandBackward {
    fn backward(&self, grad: &Tensor) -> Vec<Tensor> {
        vec![reduce_to(grad, &self.shape)]
    }
    fn name(&self) -> &'static str {
        "Expand"
    }
}

/// Elements outside the slice get zero.
pub struct NarrowBackward {
    pub shape: Vec<usize>,
    pub axis: usize,
    pub start: usize,
}

impl Backward for NarrowBackward {
    fn backward(&self, grad: &Tensor) -> Vec<Tensor> {
        let (outer, size, inner) = shape::split_at_axis(&self.shape, self.axis);
        let taken = grad.dim(self.axis);
        let g = grad.to_vec();

        let mut out = vec![0.0f32; outer * size * inner];
        for o in 0..outer {
            let dst = (o * size + self.start) * inner;
            let src = o * taken * inner;
            out[dst..dst + taken * inner].copy_from_slice(&g[src..src + taken * inner]);
        }
        vec![Tensor::from_vec(out, &self.shape).to(grad.device())]
    }
    fn name(&self) -> &'static str {
        "Narrow"
    }
}

/// Split the gradient back into the pieces that were joined.
pub struct CatBackward {
    pub axis: usize,
    pub sizes: Vec<usize>,
}

impl Backward for CatBackward {
    fn backward(&self, grad: &Tensor) -> Vec<Tensor> {
        let mut start = 0;
        self.sizes
            .iter()
            .map(|&size| {
                let piece = grad.narrow(self.axis, start, size);
                start += size;
                piece
            })
            .collect()
    }
    fn name(&self) -> &'static str {
        "Cat"
    }
}
