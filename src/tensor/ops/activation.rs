//! Activation functions and the softmax family.
//!
//! Softmax normalises over the **last** dimension. Anything with a different
//! layout should `reshape` to `[rows, classes]` first — attention does exactly that.

use crate::autograd::ops::activation::{
    GeluBackward, LeakyReluBackward, LogSoftmaxBackward, ReluBackward, SigmoidBackward,
    SiluBackward, SoftmaxBackward, TanhBackward,
};
use crate::cuda::{ffi, kernels};
use crate::tensor::storage::Storage;
use crate::tensor::Tensor;

use super::unary_op;

impl Tensor {
    /// `max(0, x)`.
    pub fn relu(&self) -> Tensor {
        unary_op(self, |x| x.max(0.0), ffi::fastnn_cuda_relu).with_grad(&[self], || ReluBackward {
            input: self.detach(),
        })
    }

    /// `1 / (1 + e^-x)`.
    pub fn sigmoid(&self) -> Tensor {
        let out = unary_op(self, sigmoid_scalar, ffi::fastnn_cuda_sigmoid);
        let saved = out.clone();
        out.with_grad(&[self], || SigmoidBackward { output: saved })
    }

    /// Hyperbolic tangent.
    pub fn tanh(&self) -> Tensor {
        let out = unary_op(self, f32::tanh, ffi::fastnn_cuda_tanh_forward);
        let saved = out.clone();
        out.with_grad(&[self], || TanhBackward { output: saved })
    }

    /// Exact GELU, `x · Φ(x)` — the transformer default.
    pub fn gelu(&self) -> Tensor {
        unary_op(self, gelu_scalar, ffi::fastnn_cuda_gelu).with_grad(&[self], || GeluBackward {
            input: self.detach(),
        })
    }

    /// SiLU / swish, `x · sigmoid(x)`.
    pub fn silu(&self) -> Tensor {
        unary_op(self, |x| x * sigmoid_scalar(x), ffi::fastnn_cuda_silu).with_grad(&[self], || {
            SiluBackward {
                input: self.detach(),
            }
        })
    }

    /// ReLU with a non-zero slope for negative inputs.
    pub fn leaky_relu(&self, slope: f32) -> Tensor {
        let out = match self.storage() {
            Storage::Cpu(data) => Tensor::from_vec(
                super::map(data, move |x| if x > 0.0 { x } else { slope * x }),
                self.shape(),
            ),
            Storage::Cuda(buf) => {
                let out = kernels::leaky_relu(buf, slope, self.numel()).expect("cuda leaky_relu");
                Tensor::raw(Storage::Cuda(out), self.shape().to_vec(), self.device())
            }
        };
        out.with_grad(&[self], || LeakyReluBackward {
            input: self.detach(),
            slope,
        })
    }

    /// Softmax over the last dimension.
    pub fn softmax(&self) -> Tensor {
        let (rows, cols) = self.as_rows();
        let out = match self.storage() {
            Storage::Cpu(data) => {
                let mut result = vec![0.0f32; data.len()];
                for r in 0..rows {
                    let row = &data[r * cols..(r + 1) * cols];
                    let shift = row_max(row);
                    let total: f32 = row.iter().map(|&x| (x - shift).exp()).sum();
                    for c in 0..cols {
                        result[r * cols + c] = (row[c] - shift).exp() / total;
                    }
                }
                Tensor::from_vec(result, self.shape())
            }
            Storage::Cuda(buf) => {
                let out = kernels::softmax(buf, rows, cols).expect("cuda softmax");
                Tensor::raw(Storage::Cuda(out), self.shape().to_vec(), self.device())
            }
        };
        let saved = out.clone();
        out.with_grad(&[self], || SoftmaxBackward { output: saved })
    }

    /// `log(softmax(x))` over the last dimension, computed without the
    /// intermediate exponential so large logits do not overflow.
    pub fn log_softmax(&self) -> Tensor {
        let (rows, cols) = self.as_rows();
        let out = match self.storage() {
            Storage::Cpu(data) => {
                let mut result = vec![0.0f32; data.len()];
                for r in 0..rows {
                    let row = &data[r * cols..(r + 1) * cols];
                    let shift = row_max(row);
                    let log_sum: f32 = row.iter().map(|&x| (x - shift).exp()).sum::<f32>().ln();
                    for c in 0..cols {
                        result[r * cols + c] = row[c] - shift - log_sum;
                    }
                }
                Tensor::from_vec(result, self.shape())
            }
            Storage::Cuda(buf) => {
                let out = kernels::log_softmax(buf, rows, cols).expect("cuda log_softmax");
                Tensor::raw(Storage::Cuda(out), self.shape().to_vec(), self.device())
            }
        };
        let saved = out.clone();
        out.with_grad(&[self], || LogSoftmaxBackward { output: saved })
    }

    /// View the tensor as `(rows, last_dim)` — the layout every row-wise op uses.
    pub(crate) fn as_rows(&self) -> (usize, usize) {
        let cols = self.last_dim();
        (self.numel() / cols, cols)
    }
}

fn sigmoid_scalar(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn gelu_scalar(x: f32) -> f32 {
    0.5 * x * (1.0 + libm::erff(x * std::f32::consts::FRAC_1_SQRT_2))
}

/// Largest value in a row; subtracting it keeps `exp` in range.
fn row_max(row: &[f32]) -> f32 {
    row.iter().copied().fold(f32::NEG_INFINITY, f32::max)
}
