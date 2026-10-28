//! Backward rules, one file per file in [`tensor::ops`](crate::tensor::ops).
//! They run with tracking off, so they can use ordinary tensor ops.

pub mod activation;
pub mod arith;
pub mod conv;
pub mod index;
pub mod loss;
pub mod matmul;
pub mod norm;
pub mod pool;
pub mod reduce;
pub mod unary;
pub mod view;

use crate::tensor::Tensor;

/// Undo broadcasting by summing `grad` down to `target`.
pub(crate) fn reduce_to(grad: &Tensor, target: &[usize]) -> Tensor {
    if grad.shape() == target {
        return grad.clone();
    }

    let mut out = grad.clone();
    while out.ndim() > target.len() {
        out = out.sum_axis(0);
    }
    for (axis, (&have, &want)) in out.shape().to_vec().iter().zip(target).enumerate() {
        if want == 1 && have != 1 {
            out = out.sum_axis_keep(axis);
        }
    }

    let dims: Vec<i64> = target.iter().map(|&d| d as i64).collect();
    out.reshape(&dims)
}
