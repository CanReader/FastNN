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
        // For C = A @ B:
        //   dA = dC @ B^T   →  matmul_nt(dC, B)   [no transpose buffer]
        //   dB = A^T @ dC   →  matmul_tn(A, dC)   [no transpose buffer]
        let grad_a = grad_output.matmul_nt(&self.b);
        let grad_b = self.a.matmul_tn(grad_output);
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
        #[cfg(feature = "cuda")]
        if let (crate::tensor::TensorStorage::Cuda(go), crate::tensor::TensorStorage::Cuda(inp)) =
            (&grad_output.storage, &self.input.storage)
        {
            use crate::tensor::cuda_backend;
            let gi = cuda_backend::cuda_binary_op(go, inp, grad_output.numel(),
                crate::tensor::cuda_backend::fastnn_cuda_relu_backward as _)
                .expect("CUDA relu_backward failed");
            return vec![Tensor::from_cuda_buffer(gi, grad_output.shape().to_vec(), false)];
        }
        let mask_data: Vec<f32> = self.input.to_vec().iter()
            .map(|&x| if x > 0.0 { 1.0 } else { 0.0 }).collect();
        let mask = Tensor::from_vec(mask_data, self.input.shape())
            .to_device(grad_output.device());
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
        let one_minus_s = self.output.mul_scalar(-1.0).add_scalar(1.0);
        vec![grad_output.mul(&self.output.mul(&one_minus_s))]
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
        let one_minus_sq = self.output.mul(&self.output).mul_scalar(-1.0).add_scalar(1.0);
        vec![grad_output.mul(&one_minus_sq)]
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
        #[cfg(feature = "cuda")]
        if let (crate::tensor::TensorStorage::Cuda(go), crate::tensor::TensorStorage::Cuda(inp)) =
            (&grad_output.storage, &self.input.storage)
        {
            use crate::tensor::cuda_backend;
            let gi = cuda_backend::cuda_binary_op(go, inp, grad_output.numel(),
                crate::tensor::cuda_backend::fastnn_cuda_gelu_backward as _)
                .expect("CUDA gelu_backward failed");
            return vec![Tensor::from_cuda_buffer(gi, grad_output.shape().to_vec(), false)];
        }
        let data = self.input.to_vec();
        let grad_data: Vec<f32> = data.iter().map(|&x| {
            let cdf = 0.5 * (1.0 + libm::erff(x * std::f32::consts::FRAC_1_SQRT_2));
            let pdf = (-0.5 * x * x).exp() * (1.0 / (2.0 * std::f32::consts::PI).sqrt());
            cdf + x * pdf
        }).collect();
        let local = Tensor::from_vec(grad_data, self.input.shape())
            .to_device(grad_output.device());
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
        let local = Tensor::from_vec(grad_data, self.input.shape())
            .to_device(grad_output.device());
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
        let s = &self.output;
        let shape = s.shape();
        let last = *shape.last().unwrap();
        let outer = s.numel() / last;

        #[cfg(feature = "cuda")]
        if let (crate::tensor::TensorStorage::Cuda(go_buf), crate::tensor::TensorStorage::Cuda(s_buf)) =
            (&grad_output.storage, &s.storage)
        {
            use crate::tensor::cuda_backend;
            let gi = cuda_backend::cuda_softmax_backward(go_buf, s_buf, outer, last)
                .expect("CUDA softmax_backward failed");
            return vec![Tensor::from_cuda_buffer(gi, shape.to_vec(), false)];
        }

        // CPU fallback
        let s_data = s.to_vec();
        let g_data = grad_output.to_vec();
        let mut out = vec![0.0f32; s.numel()];
        for b in 0..outer {
            let base = b * last;
            let mut dot = 0.0f32;
            for c in 0..last { dot += g_data[base + c] * s_data[base + c]; }
            for c in 0..last {
                out[base + c] = s_data[base + c] * (g_data[base + c] - dot);
            }
        }
        vec![Tensor::from_vec(out, shape).to_device(grad_output.device())]
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
        vec![Tensor::from_vec(out, shape).to_device(grad_output.device())]
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
        vec![Tensor::full(&self.input_shape, val).to_device(grad_output.device())]
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
        vec![Tensor::full(&self.input_shape, val).to_device(grad_output.device())]
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
    pub device: crate::tensor::Device,
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
        vec![Tensor::from_vec(grad, &[self.batch_size, self.num_classes]).to_device(self.device)]
    }
    fn inputs(&self) -> Vec<u64> { self.input_ids.clone() }
    fn name(&self) -> &str { "CrossEntropyBackward" }
}

/// Backward for permute — applies the inverse permutation to the upstream grad.
pub(crate) struct PermuteBackward {
    pub input_ids: Vec<u64>,
    pub dims: Vec<usize>,
}
impl GradFn for PermuteBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        let ndim = self.dims.len();
        let mut inv = vec![0usize; ndim];
        for (i, &d) in self.dims.iter().enumerate() {
            inv[d] = i;
        }
        vec![grad_output.permute(&inv)]
    }
    fn inputs(&self) -> Vec<u64> { self.input_ids.clone() }
    fn name(&self) -> &str { "PermuteBackward" }
}

/// Backward for embedding lookup — scatter-adds upstream grad into weight rows.
pub(crate) struct EmbeddingBackward {
    pub input_ids: Vec<u64>,
    pub indices: Vec<usize>,
    pub indices_buf: Option<crate::cuda::CudaBuffer>, // pre-uploaded indices for CUDA path
    pub num_embeddings: usize,
    pub embedding_dim: usize,
    pub device: crate::tensor::Device,
}
impl GradFn for EmbeddingBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        #[cfg(feature = "cuda")]
        if let (Some(idx_buf), crate::tensor::TensorStorage::Cuda(go_buf)) =
            (&self.indices_buf, &grad_output.storage)
        {
            use crate::tensor::cuda_backend;
            let gw = cuda_backend::cuda_embedding_backward(
                idx_buf, go_buf, self.indices.len(), self.embedding_dim, self.num_embeddings,
            ).expect("CUDA embedding_backward failed");
            return vec![crate::tensor::Tensor::from_cuda_buffer(
                gw, vec![self.num_embeddings, self.embedding_dim], false,
            )];
        }
        // CPU fallback
        let g = grad_output.to_vec();
        let mut weight_grad = vec![0.0f32; self.num_embeddings * self.embedding_dim];
        for (i, &idx) in self.indices.iter().enumerate() {
            let src = i * self.embedding_dim;
            let dst = idx * self.embedding_dim;
            for j in 0..self.embedding_dim {
                weight_grad[dst + j] += g[src + j];
            }
        }
        vec![Tensor::from_vec(weight_grad, &[self.num_embeddings, self.embedding_dim])
            .to_device(self.device)]
    }
    fn inputs(&self) -> Vec<u64> { self.input_ids.clone() }
    fn name(&self) -> &str { "EmbeddingBackward" }
}

/// Backward for layer normalization — propagates gradient through the normalization.
/// Gradient w.r.t. gamma and beta are not computed here (they are updated via a
/// separate accumulation if needed); this only propagates dL/dx.
pub(crate) struct LayerNormBackward {
    pub input_ids: Vec<u64>,
    pub x_hat: Tensor,       // normalized input: (x - mean) / std  (CPU tensor)
    pub gamma: Vec<f32>,
    pub inv_std: Vec<f32>,   // 1/sqrt(var + eps) per batch element
    pub normalized_size: usize,
    pub device: crate::tensor::Device,
}
impl GradFn for LayerNormBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        let dout = grad_output.to_vec();
        let x_hat = self.x_hat.to_vec();
        let n = self.normalized_size;
        let nf = n as f32;
        let batch = dout.len() / n;
        let mut dx = vec![0.0f32; dout.len()];

        for b in 0..batch {
            let off = b * n;
            let std_inv = self.inv_std[b];

            let mut sum_g = 0.0f32;
            let mut sum_g_xh = 0.0f32;
            for i in 0..n {
                let g_dout = self.gamma[i] * dout[off + i];
                sum_g += g_dout;
                sum_g_xh += g_dout * x_hat[off + i];
            }

            for i in 0..n {
                let g_dout = self.gamma[i] * dout[off + i];
                dx[off + i] = std_inv * (g_dout - sum_g / nf - x_hat[off + i] * sum_g_xh / nf);
            }
        }

        vec![Tensor::from_vec(dx, grad_output.shape()).to_device(self.device)]
    }
    fn inputs(&self) -> Vec<u64> { self.input_ids.clone() }
    fn name(&self) -> &str { "LayerNormBackward" }
}

/// Full GPU LayerNorm backward — computes grad_input, grad_gamma, grad_beta all on CUDA.
pub(crate) struct LayerNormCudaBackward {
    pub input_ids: Vec<u64>,  // [input.id, gamma.id, beta.id]
    pub input: Tensor,         // original input (CUDA)
    pub gamma: Tensor,         // gamma (CUDA)
    pub mean: Tensor,          // per-batch means (CUDA)
    pub inv_var: Tensor,       // per-batch inv_std (CUDA)
    pub normalized_size: usize,
}
impl GradFn for LayerNormCudaBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        let batch = grad_output.numel() / self.normalized_size;
        match (
            &grad_output.storage,
            &self.input.storage,
            &self.gamma.storage,
            &self.mean.storage,
            &self.inv_var.storage,
        ) {
            (
                crate::tensor::TensorStorage::Cuda(go),
                crate::tensor::TensorStorage::Cuda(inp),
                crate::tensor::TensorStorage::Cuda(gam),
                crate::tensor::TensorStorage::Cuda(mean),
                crate::tensor::TensorStorage::Cuda(inv_var),
            ) => {
                use crate::tensor::cuda_backend;
                let (gi, gg, gb) = cuda_backend::cuda_layer_norm_backward(
                    go, inp, gam, mean, inv_var, batch, self.normalized_size,
                ).expect("CUDA layer_norm_backward failed");
                let shape = grad_output.shape().to_vec();
                let ns = self.normalized_size;
                vec![
                    Tensor::from_cuda_buffer(gi, shape, false),
                    Tensor::from_cuda_buffer(gg, vec![ns], false),
                    Tensor::from_cuda_buffer(gb, vec![ns], false),
                ]
            }
            _ => {
                // Fallback: should not happen if forward used CUDA path
                vec![grad_output.clone(), Tensor::zeros(&[self.normalized_size]), Tensor::zeros(&[self.normalized_size])]
            }
        }
    }
    fn inputs(&self) -> Vec<u64> { self.input_ids.clone() }
    fn name(&self) -> &str { "LayerNormCudaBackward" }
}

/// Backward for 2D convolution.
/// inputs order: [input, weight, bias(optional)] — backward returns grads in same order.
pub(crate) struct Conv2dBackward {
    pub input_ids: Vec<u64>,
    pub input: Tensor,
    pub weight: Tensor,
    pub has_bias: bool,
    pub out_channels: usize,
    pub kernel_size: (usize, usize),
    pub stride: (usize, usize),
    pub padding: (usize, usize),
}
impl GradFn for Conv2dBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        let is = self.input.shape();
        let (n, in_c, in_h, in_w) = (is[0], is[1], is[2], is[3]);
        let gs = grad_output.shape();
        let (out_h, out_w) = (gs[2], gs[3]);
        let (kh, kw) = self.kernel_size;
        let (sh, sw) = self.stride;
        let (ph, pw) = self.padding;
        let out_c = self.out_channels;

        let input_data = self.input.to_vec();
        let weight_data = self.weight.to_vec();
        let go = grad_output.to_vec();

        let mut grad_input = vec![0.0f32; n * in_c * in_h * in_w];
        let mut grad_weight = vec![0.0f32; out_c * in_c * kh * kw];

        for b in 0..n {
            for oc in 0..out_c {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let go_val = go[((b * out_c + oc) * out_h + oh) * out_w + ow];
                        for ic in 0..in_c {
                            for ki in 0..kh {
                                for kj in 0..kw {
                                    let ih = (oh * sh + ki) as isize - ph as isize;
                                    let iw = (ow * sw + kj) as isize - pw as isize;
                                    if ih >= 0 && ih < in_h as isize && iw >= 0 && iw < in_w as isize {
                                        let ih = ih as usize;
                                        let iw = iw as usize;
                                        let w_idx = ((oc * in_c + ic) * kh + ki) * kw + kj;
                                        let i_idx = ((b * in_c + ic) * in_h + ih) * in_w + iw;
                                        grad_input[i_idx] += go_val * weight_data[w_idx];
                                        grad_weight[w_idx] += go_val * input_data[i_idx];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        let mut results = vec![
            Tensor::from_vec(grad_input, &[n, in_c, in_h, in_w]),
            Tensor::from_vec(grad_weight, &[out_c, in_c, kh, kw]),
        ];

        if self.has_bias {
            let mut grad_bias = vec![0.0f32; out_c];
            for b in 0..n {
                for oc in 0..out_c {
                    for oh in 0..out_h {
                        for ow in 0..out_w {
                            grad_bias[oc] += go[((b * out_c + oc) * out_h + oh) * out_w + ow];
                        }
                    }
                }
            }
            results.push(Tensor::from_vec(grad_bias, &[out_c]));
        }

        results
    }
    fn inputs(&self) -> Vec<u64> { self.input_ids.clone() }
    fn name(&self) -> &str { "Conv2dBackward" }
}

/// Backward for 2D max pooling — routes each gradient to the saved argmax position.
pub(crate) struct MaxPool2dBackward {
    pub input_ids: Vec<u64>,
    pub input_shape: Vec<usize>,
    pub argmax: Vec<usize>,
}
impl GradFn for MaxPool2dBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        let n_in: usize = self.input_shape.iter().product();
        let mut grad_input = vec![0.0f32; n_in];
        let go = grad_output.to_vec();
        for (i, &flat_idx) in self.argmax.iter().enumerate() {
            grad_input[flat_idx] += go[i];
        }
        vec![Tensor::from_vec(grad_input, &self.input_shape)]
    }
    fn inputs(&self) -> Vec<u64> { self.input_ids.clone() }
    fn name(&self) -> &str { "MaxPool2dBackward" }
}

/// Backward for 2D average pooling — distributes gradient evenly over each pool window.
pub(crate) struct AvgPool2dBackward {
    pub input_ids: Vec<u64>,
    pub input_shape: Vec<usize>,
    pub kernel_size: (usize, usize),
    pub stride: (usize, usize),
    pub padding: (usize, usize),
}
impl GradFn for AvgPool2dBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        let (n, c, h, w) = (self.input_shape[0], self.input_shape[1], self.input_shape[2], self.input_shape[3]);
        let (kh, kw) = self.kernel_size;
        let (sh, sw) = self.stride;
        let (ph, pw) = self.padding;
        let gs = grad_output.shape();
        let (out_h, out_w) = (gs[2], gs[3]);
        let go = grad_output.to_vec();
        let mut grad_input = vec![0.0f32; n * c * h * w];

        for b in 0..n {
            for ch in 0..c {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let go_val = go[((b * c + ch) * out_h + oh) * out_w + ow];
                        let mut count = 0usize;
                        for ki in 0..kh {
                            for kj in 0..kw {
                                let ih = (oh * sh + ki) as isize - ph as isize;
                                let iw = (ow * sw + kj) as isize - pw as isize;
                                if ih >= 0 && ih < h as isize && iw >= 0 && iw < w as isize {
                                    count += 1;
                                }
                            }
                        }
                        let g = go_val / count as f32;
                        for ki in 0..kh {
                            for kj in 0..kw {
                                let ih = (oh * sh + ki) as isize - ph as isize;
                                let iw = (ow * sw + kj) as isize - pw as isize;
                                if ih >= 0 && ih < h as isize && iw >= 0 && iw < w as isize {
                                    grad_input[((b * c + ch) * h + ih as usize) * w + iw as usize] += g;
                                }
                            }
                        }
                    }
                }
            }
        }

        vec![Tensor::from_vec(grad_input, &self.input_shape)]
    }
    fn inputs(&self) -> Vec<u64> { self.input_ids.clone() }
    fn name(&self) -> &str { "AvgPool2dBackward" }
}

/// Backward for 2D batch normalization.
/// inputs order: [input, gamma, beta] — backward returns grads in same order.
pub(crate) struct BatchNorm2dBackward {
    pub input_ids: Vec<u64>,
    pub x_hat: Tensor,
    pub gamma: Vec<f32>,
    pub inv_std: Vec<f32>,
    pub device: crate::tensor::Device,
}
impl GradFn for BatchNorm2dBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        let dout = grad_output.to_vec();
        let x_hat = self.x_hat.to_vec();
        let shape = grad_output.shape();
        let (n, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
        let spatial = h * w;
        let npf = (n * spatial) as f32;

        let mut grad_input = vec![0.0f32; dout.len()];
        let mut grad_gamma = vec![0.0f32; c];
        let mut grad_beta = vec![0.0f32; c];

        for ch in 0..c {
            for b in 0..n {
                for s in 0..spatial {
                    let idx = ((b * c + ch) * h + s / w) * w + s % w;
                    grad_gamma[ch] += dout[idx] * x_hat[idx];
                    grad_beta[ch] += dout[idx];
                }
            }

            let inv_std = self.inv_std[ch];
            let gamma = self.gamma[ch];

            for b in 0..n {
                for s in 0..spatial {
                    let idx = ((b * c + ch) * h + s / w) * w + s % w;
                    grad_input[idx] = gamma * inv_std * (
                        dout[idx]
                        - grad_beta[ch] / npf
                        - x_hat[idx] * grad_gamma[ch] / npf
                    );
                }
            }
        }

        vec![
            Tensor::from_vec(grad_input, shape).to_device(self.device),
            Tensor::from_vec(grad_gamma, &[c]).to_device(self.device),
            Tensor::from_vec(grad_beta, &[c]).to_device(self.device),
        ]
    }
    fn inputs(&self) -> Vec<u64> { self.input_ids.clone() }
    fn name(&self) -> &str { "BatchNorm2dBackward" }
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
