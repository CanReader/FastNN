//! Derivatives of reductions, which broadcast the gradient back out.

use crate::autograd::Backward;
use crate::tensor::shape;
use crate::tensor::Tensor;

/// Every element gets the whole gradient.
pub struct SumBackward {
    pub shape: Vec<usize>,
}

impl Backward for SumBackward {
    fn backward(&self, grad: &Tensor) -> Vec<Tensor> {
        vec![Tensor::full(&self.shape, grad.item()).to(grad.device())]
    }
    fn name(&self) -> &'static str {
        "Sum"
    }
}

/// Like sum, divided by the element count.
pub struct MeanBackward {
    pub shape: Vec<usize>,
    pub count: usize,
}

impl Backward for MeanBackward {
    fn backward(&self, grad: &Tensor) -> Vec<Tensor> {
        let share = grad.item() / self.count as f32;
        vec![Tensor::full(&self.shape, share).to(grad.device())]
    }
    fn name(&self) -> &'static str {
        "Mean"
    }
}

/// Copy the gradient back along the collapsed axis.
pub struct SumAxisBackward {
    pub shape: Vec<usize>,
    pub axis: usize,
}

impl Backward for SumAxisBackward {
    fn backward(&self, grad: &Tensor) -> Vec<Tensor> {
        let (outer, size, inner) = shape::split_at_axis(&self.shape, self.axis);
        let g = grad.to_vec();

        let mut out = vec![0.0f32; outer * size * inner];
        for o in 0..outer {
            for a in 0..size {
                let row = (o * size + a) * inner;
                out[row..row + inner].copy_from_slice(&g[o * inner..(o + 1) * inner]);
            }
        }
        vec![Tensor::from_vec(out, &self.shape).to(grad.device())]
    }
    fn name(&self) -> &'static str {
        "SumAxis"
    }
}
