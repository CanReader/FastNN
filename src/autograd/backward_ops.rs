//! Backward gradient functions for tensor operations.
//!
//! Each struct implements `GradFn` — given `grad_output`, it returns gradients for each input.
//! Shared between the tensor-native autograd path (`Tensor::add/matmul/...`) and the
//! legacy `Variable` wrapper.

use crate::tensor::Tensor;
use crate::autograd::graph::GradFn;

pub(crate) struct AddBackward {
    pub input_ids: Vec<u64>,
    pub a_shape: Vec<usize>,
    pub b_shape: Vec<usize>,
}
impl GradFn for AddBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        if self.input_ids.len() == 2 {
            vec![
                reduce_to_shape(grad_output, &self.a_shape),
                reduce_to_shape(grad_output, &self.b_shape),
            ]
        } else {
            vec![grad_output.clone()]
        }
    }
    fn inputs(&self) -> Vec<u64> { self.input_ids.clone() }
    fn name(&self) -> &str { "AddBackward" }
}

pub(crate) struct SubBackward {
    pub input_ids: Vec<u64>,
    pub a_shape: Vec<usize>,
    pub b_shape: Vec<usize>,
}
impl GradFn for SubBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        vec![
            reduce_to_shape(grad_output, &self.a_shape),
            reduce_to_shape(&grad_output.neg(), &self.b_shape),
        ]
    }
    fn inputs(&self) -> Vec<u64> { self.input_ids.clone() }
    fn name(&self) -> &str { "SubBackward" }
}

pub(crate) struct MulBackward {
    pub input_ids: Vec<u64>,
    pub a: Tensor,
    pub b: Tensor,
}
impl GradFn for MulBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        vec![
            reduce_to_shape(&grad_output.mul(&self.b), self.a.shape()),
            reduce_to_shape(&grad_output.mul(&self.a), self.b.shape()),
        ]
    }
    fn inputs(&self) -> Vec<u64> { self.input_ids.clone() }
    fn name(&self) -> &str { "MulBackward" }
}

pub(crate) struct DivBackward {
    pub input_ids: Vec<u64>,
    pub a: Tensor,
    pub b: Tensor,
}
impl GradFn for DivBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        // d(a/b)/da = 1/b,  d(a/b)/db = -a/b^2
        let grad_a = grad_output.div(&self.b);
        let b_sq = self.b.mul(&self.b);
        let grad_b = grad_output.mul(&self.a).div(&b_sq).neg();
        vec![
            reduce_to_shape(&grad_a, self.a.shape()),
            reduce_to_shape(&grad_b, self.b.shape()),
        ]
    }
    fn inputs(&self) -> Vec<u64> { self.input_ids.clone() }
    fn name(&self) -> &str { "DivBackward" }
}

pub(crate) struct MatmulBackward {
    pub input_ids: Vec<u64>,
    pub a: Tensor,
    pub b: Tensor,
}
impl GradFn for MatmulBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        // For C = A @ B:   dA = dC @ B^T,  dB = A^T @ dC
        let grad_a = grad_output.matmul(&self.b.transpose());
        let grad_b = self.a.transpose().matmul(grad_output);
        vec![grad_a, grad_b]
    }
    fn inputs(&self) -> Vec<u64> { self.input_ids.clone() }
    fn name(&self) -> &str { "MatmulBackward" }
}

pub(crate) struct ReluBackward {
    pub input_ids: Vec<u64>,
    pub input: Tensor,
}
impl GradFn for ReluBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        let mask_data: Vec<f32> = self.input.to_vec().iter()
            .map(|&x| if x > 0.0 { 1.0 } else { 0.0 }).collect();
        let mask = Tensor::from_vec(mask_data, self.input.shape());
        vec![grad_output.mul(&mask)]
    }
    fn inputs(&self) -> Vec<u64> { self.input_ids.clone() }
    fn name(&self) -> &str { "ReluBackward" }
}

pub(crate) struct SigmoidBackward {
    pub input_ids: Vec<u64>,
    pub output: Tensor,
}
impl GradFn for SigmoidBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        // d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
        let ones = Tensor::ones(self.output.shape());
        let grad = grad_output.mul(&self.output.mul(&ones.sub(&self.output)));
        vec![grad]
    }
    fn inputs(&self) -> Vec<u64> { self.input_ids.clone() }
    fn name(&self) -> &str { "SigmoidBackward" }
}

pub(crate) struct TanhBackward {
    pub input_ids: Vec<u64>,
    pub output: Tensor,
}
impl GradFn for TanhBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        // d/dx tanh(x) = 1 - tanh(x)^2
        let ones = Tensor::ones(self.output.shape());
        let grad = grad_output.mul(&ones.sub(&self.output.mul(&self.output)));
        vec![grad]
    }
    fn inputs(&self) -> Vec<u64> { self.input_ids.clone() }
    fn name(&self) -> &str { "TanhBackward" }
}

pub(crate) struct GeluBackward {
    pub input_ids: Vec<u64>,
    pub input: Tensor,
}
impl GradFn for GeluBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        let data = self.input.to_vec();
        let grad_data: Vec<f32> = data.iter().map(|&x| {
            let cdf = 0.5 * (1.0 + libm::erff(x * std::f32::consts::FRAC_1_SQRT_2));
            let pdf = (-0.5 * x * x).exp() * (1.0 / (2.0 * std::f32::consts::PI).sqrt());
            cdf + x * pdf
        }).collect();
        let local = Tensor::from_vec(grad_data, self.input.shape());
        vec![grad_output.mul(&local)]
    }
    fn inputs(&self) -> Vec<u64> { self.input_ids.clone() }
    fn name(&self) -> &str { "GeluBackward" }
}

pub(crate) struct SiluBackward {
    pub input_ids: Vec<u64>,
    pub input: Tensor,
}
impl GradFn for SiluBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        // silu(x) = x * sigmoid(x)
        // d/dx silu(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
        let data = self.input.to_vec();
        let grad_data: Vec<f32> = data.iter().map(|&x| {
            let s = 1.0 / (1.0 + (-x).exp());
            s + x * s * (1.0 - s)
        }).collect();
        let local = Tensor::from_vec(grad_data, self.input.shape());
        vec![grad_output.mul(&local)]
    }
    fn inputs(&self) -> Vec<u64> { self.input_ids.clone() }
    fn name(&self) -> &str { "SiluBackward" }
}

pub(crate) struct SoftmaxBackward {
    pub input_ids: Vec<u64>,
    pub output: Tensor,
}
impl GradFn for SoftmaxBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        // Jacobian-vector product for softmax along last dim:
        //   dL/dx_i = s_i * (dL/dy_i - sum_k(dL/dy_k * s_k))
        let s = &self.output;
        let shape = s.shape();
        let last = *shape.last().unwrap();
        let outer = s.numel() / last;
        let s_data = s.to_vec();
        let g_data = grad_output.to_vec();
        let mut out = vec![0.0f32; s.numel()];
        for b in 0..outer {
            let base = b * last;
            let mut dot = 0.0f32;
            for c in 0..last {
                dot += g_data[base + c] * s_data[base + c];
            }
            for c in 0..last {
                out[base + c] = s_data[base + c] * (g_data[base + c] - dot);
            }
        }
        vec![Tensor::from_vec(out, shape)]
    }
    fn inputs(&self) -> Vec<u64> { self.input_ids.clone() }
    fn name(&self) -> &str { "SoftmaxBackward" }
}

pub(crate) struct LogSoftmaxBackward {
    pub input_ids: Vec<u64>,
    pub output: Tensor,
}
impl GradFn for LogSoftmaxBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        // dL/dx = dL/dy - softmax(x) * sum(dL/dy)  per row
        let softmax = self.output.exp();
        let shape = self.output.shape();
        let last = *shape.last().unwrap();
        let outer = self.output.numel() / last;
        let s_data = softmax.to_vec();
        let g_data = grad_output.to_vec();
        let mut out = vec![0.0f32; self.output.numel()];
        for b in 0..outer {
            let base = b * last;
            let mut sum = 0.0f32;
            for c in 0..last { sum += g_data[base + c]; }
            for c in 0..last {
                out[base + c] = g_data[base + c] - s_data[base + c] * sum;
            }
        }
        vec![Tensor::from_vec(out, shape)]
    }
    fn inputs(&self) -> Vec<u64> { self.input_ids.clone() }
    fn name(&self) -> &str { "LogSoftmaxBackward" }
}

pub(crate) struct SumBackward {
    pub input_ids: Vec<u64>,
    pub input_shape: Vec<usize>,
}
impl GradFn for SumBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        let val = grad_output.item();
        vec![Tensor::full(&self.input_shape, val)]
    }
    fn inputs(&self) -> Vec<u64> { self.input_ids.clone() }
    fn name(&self) -> &str { "SumBackward" }
}

pub(crate) struct MeanBackward {
    pub input_ids: Vec<u64>,
    pub input_numel: usize,
    pub input_shape: Vec<usize>,
}
impl GradFn for MeanBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        let val = grad_output.item() / self.input_numel as f32;
        vec![Tensor::full(&self.input_shape, val)]
    }
    fn inputs(&self) -> Vec<u64> { self.input_ids.clone() }
    fn name(&self) -> &str { "MeanBackward" }
}

pub(crate) struct MulScalarBackward {
    pub input_ids: Vec<u64>,
    pub scalar: f32,
}
impl GradFn for MulScalarBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        vec![grad_output.mul_scalar(self.scalar)]
    }
    fn inputs(&self) -> Vec<u64> { self.input_ids.clone() }
    fn name(&self) -> &str { "MulScalarBackward" }
}

pub(crate) struct AddScalarBackward {
    pub input_ids: Vec<u64>,
}
impl GradFn for AddScalarBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        vec![grad_output.clone()]
    }
    fn inputs(&self) -> Vec<u64> { self.input_ids.clone() }
    fn name(&self) -> &str { "AddScalarBackward" }
}

pub(crate) struct NegBackward {
    pub input_ids: Vec<u64>,
}
impl GradFn for NegBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        vec![grad_output.neg()]
    }
    fn inputs(&self) -> Vec<u64> { self.input_ids.clone() }
    fn name(&self) -> &str { "NegBackward" }
}

pub(crate) struct ReshapeBackward {
    pub input_ids: Vec<u64>,
    pub input_shape: Vec<usize>,
}
impl GradFn for ReshapeBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        let shape: Vec<i64> = self.input_shape.iter().map(|&s| s as i64).collect();
        vec![grad_output.reshape(&shape)]
    }
    fn inputs(&self) -> Vec<u64> { self.input_ids.clone() }
    fn name(&self) -> &str { "ReshapeBackward" }
}

pub(crate) struct TransposeBackward {
    pub input_ids: Vec<u64>,
}
impl GradFn for TransposeBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        // Transpose is its own inverse (for last-two-dims transpose)
        vec![grad_output.transpose()]
    }
    fn inputs(&self) -> Vec<u64> { self.input_ids.clone() }
    fn name(&self) -> &str { "TransposeBackward" }
}

pub(crate) struct ExpandBackward {
    pub input_ids: Vec<u64>,
    pub input_shape: Vec<usize>,
}
impl GradFn for ExpandBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        vec![reduce_to_shape(grad_output, &self.input_shape)]
    }
    fn inputs(&self) -> Vec<u64> { self.input_ids.clone() }
    fn name(&self) -> &str { "ExpandBackward" }
}

pub(crate) struct LogBackward {
    pub input_ids: Vec<u64>,
    pub input: Tensor,
}
impl GradFn for LogBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        // d/dx log(x) = 1/x
        vec![grad_output.div(&self.input)]
    }
    fn inputs(&self) -> Vec<u64> { self.input_ids.clone() }
    fn name(&self) -> &str { "LogBackward" }
}

pub(crate) struct ExpBackward {
    pub input_ids: Vec<u64>,
    pub output: Tensor,
}
impl GradFn for ExpBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        // d/dx exp(x) = exp(x)
        vec![grad_output.mul(&self.output)]
    }
    fn inputs(&self) -> Vec<u64> { self.input_ids.clone() }
    fn name(&self) -> &str { "ExpBackward" }
}

/// Backward for fused cross-entropy = -mean(log_softmax(logits)[target]).
/// Cached: softmax(logits) and target indices. Grad: (softmax - one_hot) / N.
pub(crate) struct CrossEntropyBackward {
    pub input_ids: Vec<u64>,
    pub softmax: Tensor,
    pub targets: Vec<usize>,
    pub batch_size: usize,
    pub num_classes: usize,
}
impl GradFn for CrossEntropyBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        let upstream = grad_output.item(); // scalar dL/dloss
        let probs = self.softmax.to_vec();
        let mut grad = vec![0.0f32; self.batch_size * self.num_classes];
        let scale = upstream / self.batch_size as f32;
        for b in 0..self.batch_size {
            let t = self.targets[b];
            for c in 0..self.num_classes {
                let one_hot = if c == t { 1.0 } else { 0.0 };
                grad[b * self.num_classes + c] = (probs[b * self.num_classes + c] - one_hot) * scale;
            }
        }
        vec![Tensor::from_vec(grad, &[self.batch_size, self.num_classes])]
    }
    fn inputs(&self) -> Vec<u64> { self.input_ids.clone() }
    fn name(&self) -> &str { "CrossEntropyBackward" }
}

// ============================================================================
// Broadcasting-aware gradient reduction
// ============================================================================

/// Reduce `grad` to `target_shape` by summing over broadcasted dims.
/// Used when the forward op broadcast a smaller tensor up to a larger output.
pub(crate) fn reduce_to_shape(grad: &Tensor, target_shape: &[usize]) -> Tensor {
    if grad.shape() == target_shape {
        return grad.clone();
    }

    let grad_shape = grad.shape().to_vec();
    let out_ndim = grad_shape.len();
    let in_ndim = target_shape.len();

    let mut result = grad.clone();

    // 1) Sum away leading dims that target_shape doesn't have at all.
    while result.ndim() > in_ndim {
        result = result.sum_axis(0);
        // sum_axis drops the axis; if it flattened the last dim to 1, ensure shape is right
        if result.shape().len() > in_ndim {
            // sum_axis appends [1] if it became empty — our while loop handles it
        }
    }

    // 2) For dims that are size 1 in target but larger in grad, sum along that axis.
    //    `sum_axis` removes the axis; then reshape to insert the size-1 back.
    let current_shape = result.shape().to_vec();
    // Align shapes by left-padding target with 1s up to current ndim.
    let _ = out_ndim;
    for (axis, (&cur, &tgt)) in current_shape.iter().zip(target_shape.iter()).enumerate() {
        if tgt == 1 && cur != 1 {
            result = result.sum_axis(axis);
            // Re-insert the axis (sum_axis dropped it), so subsequent indices stay aligned.
            let mut new_shape: Vec<i64> = result.shape().iter().map(|&s| s as i64).collect();
            new_shape.insert(axis, 1);
            result = result.reshape(&new_shape);
        }
    }

    // Final safety: ensure shape matches target.
    if result.shape() != target_shape {
        let target_i64: Vec<i64> = target_shape.iter().map(|&s| s as i64).collect();
        result = result.reshape(&target_i64);
    }

    result
}
