//! Derivatives of activations and the softmax family.
//!
//! Where `kernels.cu` has a fused backward kernel we use it; the host formula
//! below it is both the CPU path and the reference the GPU kernel must match
//! (see `tests/cuda_parity.rs`).

use crate::autograd::Backward;
use crate::cuda::{ffi, kernels};
use crate::tensor::storage::Storage;
use crate::tensor::Tensor;

use super::unary::elementwise;

/// Run a fused `(grad, saved) → grad_input` kernel when both live on the GPU.
fn fused(grad: &Tensor, saved: &Tensor, kernel: ffi::BinaryKernel) -> Option<Tensor> {
    let (Storage::Cuda(g), Storage::Cuda(s)) = (grad.storage(), saved.storage()) else {
        return None;
    };
    let out = kernels::binary(g, s, grad.numel(), kernel).expect("cuda activation backward");
    Some(Tensor::raw(
        Storage::Cuda(out),
        grad.shape().to_vec(),
        grad.device(),
    ))
}

/// `d relu(x) = g · [x > 0]`.
pub struct ReluBackward {
    pub input: Tensor,
}

impl Backward for ReluBackward {
    fn backward(&self, grad: &Tensor) -> Vec<Tensor> {
        let out = fused(grad, &self.input, ffi::fastnn_cuda_relu_backward)
            .unwrap_or_else(|| elementwise(grad, &self.input, |x| (x > 0.0) as i32 as f32));
        vec![out]
    }
    fn name(&self) -> &'static str {
        "Relu"
    }
}

/// `d σ(x) = g · σ(1 - σ)` — in terms of the forward output.
pub struct SigmoidBackward {
    pub output: Tensor,
}

impl Backward for SigmoidBackward {
    fn backward(&self, grad: &Tensor) -> Vec<Tensor> {
        let out =
            fused(grad, &self.output, ffi::fastnn_cuda_sigmoid_backward).unwrap_or_else(|| {
                let one_minus = self.output.neg().add_scalar(1.0);
                grad.mul(&self.output.mul(&one_minus))
            });
        vec![out]
    }
    fn name(&self) -> &'static str {
        "Sigmoid"
    }
}

/// `d tanh(x) = g · (1 - tanh²)`.
pub struct TanhBackward {
    pub output: Tensor,
}

impl Backward for TanhBackward {
    fn backward(&self, grad: &Tensor) -> Vec<Tensor> {
        let out = fused(grad, &self.output, ffi::fastnn_cuda_tanh_backward)
            .unwrap_or_else(|| grad.mul(&self.output.square().neg().add_scalar(1.0)));
        vec![out]
    }
    fn name(&self) -> &'static str {
        "Tanh"
    }
}

/// `d gelu(x) = g · (Φ(x) + x·φ(x))`, the exact form matching the forward.
pub struct GeluBackward {
    pub input: Tensor,
}

impl Backward for GeluBackward {
    fn backward(&self, grad: &Tensor) -> Vec<Tensor> {
        let out = fused(grad, &self.input, ffi::fastnn_cuda_gelu_backward)
            .unwrap_or_else(|| elementwise(grad, &self.input, gelu_derivative));
        vec![out]
    }
    fn name(&self) -> &'static str {
        "Gelu"
    }
}

/// `d silu(x) = g · (σ + x·σ·(1-σ))`.
pub struct SiluBackward {
    pub input: Tensor,
}

impl Backward for SiluBackward {
    fn backward(&self, grad: &Tensor) -> Vec<Tensor> {
        let out = fused(grad, &self.input, ffi::fastnn_cuda_silu_backward)
            .unwrap_or_else(|| elementwise(grad, &self.input, silu_derivative));
        vec![out]
    }
    fn name(&self) -> &'static str {
        "Silu"
    }
}

/// `d leaky_relu(x) = g · (x > 0 ? 1 : slope)`.
pub struct LeakyReluBackward {
    pub input: Tensor,
    pub slope: f32,
}

impl Backward for LeakyReluBackward {
    fn backward(&self, grad: &Tensor) -> Vec<Tensor> {
        let slope = self.slope;
        vec![elementwise(grad, &self.input, move |x| {
            if x > 0.0 {
                1.0
            } else {
                slope
            }
        })]
    }
    fn name(&self) -> &'static str {
        "LeakyRelu"
    }
}

/// `d softmax = s · (g - Σ(g·s))` per row — the Jacobian collapses to this
/// because every output in a row depends on every input in that row.
pub struct SoftmaxBackward {
    pub output: Tensor,
}

impl Backward for SoftmaxBackward {
    fn backward(&self, grad: &Tensor) -> Vec<Tensor> {
        let (rows, cols) = self.output.as_rows();

        if let (Storage::Cuda(g), Storage::Cuda(s)) = (grad.storage(), self.output.storage()) {
            let out = kernels::softmax_backward(g, s, rows, cols).expect("cuda softmax backward");
            return vec![Tensor::raw(
                Storage::Cuda(out),
                grad.shape().to_vec(),
                grad.device(),
            )];
        }

        let (s, g) = (self.output.to_vec(), grad.to_vec());
        let mut out = vec![0.0f32; s.len()];
        for r in 0..rows {
            let base = r * cols;
            let dot: f32 = (0..cols).map(|c| g[base + c] * s[base + c]).sum();
            for c in 0..cols {
                out[base + c] = s[base + c] * (g[base + c] - dot);
            }
        }
        vec![Tensor::from_vec(out, grad.shape()).to(grad.device())]
    }
    fn name(&self) -> &'static str {
        "Softmax"
    }
}

/// `d log_softmax = g - softmax · Σg` per row.
pub struct LogSoftmaxBackward {
    pub output: Tensor,
}

impl Backward for LogSoftmaxBackward {
    fn backward(&self, grad: &Tensor) -> Vec<Tensor> {
        let (rows, cols) = self.output.as_rows();
        let softmax = self.output.exp().to_vec();
        let g = grad.to_vec();

        let mut out = vec![0.0f32; g.len()];
        for r in 0..rows {
            let base = r * cols;
            let total: f32 = g[base..base + cols].iter().sum();
            for c in 0..cols {
                out[base + c] = g[base + c] - softmax[base + c] * total;
            }
        }
        vec![Tensor::from_vec(out, grad.shape()).to(grad.device())]
    }
    fn name(&self) -> &'static str {
        "LogSoftmax"
    }
}

fn gelu_derivative(x: f32) -> f32 {
    let cdf = 0.5 * (1.0 + libm::erff(x * std::f32::consts::FRAC_1_SQRT_2));
    let pdf = (-0.5 * x * x).exp() / (2.0 * std::f32::consts::PI).sqrt();
    cdf + x * pdf
}

fn silu_derivative(x: f32) -> f32 {
    let s = 1.0 / (1.0 + (-x).exp());
    s + x * s * (1.0 - s)
}
