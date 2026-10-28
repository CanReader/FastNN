//! Derivatives of the cross-entropy losses.

use crate::autograd::Backward;
use crate::tensor::Tensor;

/// `(softmax - one_hot) / batch`. Fusing softmax with the NLL makes this one
/// subtraction instead of a log, a division and a gather.
pub struct CrossEntropyBackward {
    pub softmax: Tensor,
    pub targets: Vec<usize>,
    pub classes: usize,
}

impl Backward for CrossEntropyBackward {
    fn backward(&self, grad: &Tensor) -> Vec<Tensor> {
        let batch = self.targets.len();
        let scale = grad.item() / batch as f32;
        let mut out = self.softmax.to_vec();

        for (row, &target) in self.targets.iter().enumerate() {
            out[row * self.classes + target] -= 1.0;
        }
        for value in &mut out {
            *value *= scale;
        }

        vec![Tensor::from_vec(out, &[batch, self.classes]).to(grad.device())]
    }
    fn name(&self) -> &'static str {
        "CrossEntropy"
    }
}

/// Configurable cross-entropy. The forward already stored `w·(p - q)` per row
/// (zero for ignored rows), so only the reduction's scaling is left.
pub struct WeightedCrossEntropyBackward {
    pub difference: Tensor,
    pub normalizer: f32,
    pub per_row: bool,
}

impl Backward for WeightedCrossEntropyBackward {
    fn backward(&self, grad: &Tensor) -> Vec<Tensor> {
        let difference = self.difference.to(grad.device());
        let scaled = if self.per_row {
            let rows = grad.numel() as i64;
            difference.mul(&grad.reshape(&[rows, 1]))
        } else {
            difference.mul_scalar(grad.item() / self.normalizer)
        };
        vec![scaled]
    }
    fn name(&self) -> &'static str {
        "WeightedCrossEntropy"
    }
}
