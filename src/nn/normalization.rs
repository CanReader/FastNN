use std::sync::Arc;
use crate::tensor::{Tensor, TensorStorage};
use crate::nn::module::Module;
use crate::autograd::graph;
use crate::autograd::backward_ops::{LayerNormBackward, LayerNormCudaBackward, BatchNorm2dBackward};

/// Batch Normalization for 2D inputs (4D tensor: [N, C, H, W]).
pub struct BatchNorm2d {
    pub gamma: Tensor,      // scale, shape [C]
    pub beta: Tensor,       // shift, shape [C]
    pub running_mean: Tensor,
    pub running_var: Tensor,
    num_features: usize,
    epsilon: f32,
    momentum: f32,
    training: bool,
}

impl BatchNorm2d {
    pub fn new(num_features: usize) -> Self {
        let mut gamma = Tensor::ones(&[num_features]);
        gamma.set_requires_grad(true);
        let mut beta = Tensor::zeros(&[num_features]);
        beta.set_requires_grad(true);

        BatchNorm2d {
            gamma,
            beta,
            running_mean: Tensor::zeros(&[num_features]),
            running_var: Tensor::ones(&[num_features]),
            num_features,
            epsilon: 1e-5,
            momentum: 0.1,
            training: true,
        }
    }
}

impl Module for BatchNorm2d {
    fn forward(&self, input: &Tensor) -> Tensor {
        let shape = input.shape();
        assert_eq!(shape.len(), 4, "BatchNorm2d expects [N, C, H, W]");
        let (n, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
        assert_eq!(c, self.num_features);

        let data = input.to_vec();
        let spatial = h * w;
        let mut output = vec![0.0f32; data.len()];
        let mut x_hat_data = vec![0.0f32; data.len()];
        let mut inv_std_per_ch = vec![0.0f32; c];

        let running_mean_data = self.running_mean.to_vec();
        let running_var_data = self.running_var.to_vec();
        let gamma_vec = self.gamma.to_vec();
        let beta_vec = self.beta.to_vec();

        for ch in 0..c {
            let (mean, var) = if self.training {
                let count = (n * spatial) as f32;
                let mut sum = 0.0f32;
                for b in 0..n {
                    for s in 0..spatial {
                        sum += data[((b * c + ch) * h + s / w) * w + s % w];
                    }
                }
                let mean = sum / count;
                let mut var_sum = 0.0f32;
                for b in 0..n {
                    for s in 0..spatial {
                        let idx = ((b * c + ch) * h + s / w) * w + s % w;
                        let diff = data[idx] - mean;
                        var_sum += diff * diff;
                    }
                }
                (mean, var_sum / count)
            } else {
                (running_mean_data[ch], running_var_data[ch])
            };

            let inv_std = 1.0 / (var + self.epsilon).sqrt();
            inv_std_per_ch[ch] = inv_std;
            let gamma = gamma_vec[ch];
            let beta = beta_vec[ch];

            for b in 0..n {
                for s in 0..spatial {
                    let idx = ((b * c + ch) * h + s / w) * w + s % w;
                    let xh = (data[idx] - mean) * inv_std;
                    x_hat_data[idx] = xh;
                    output[idx] = gamma * xh + beta;
                }
            }
        }

        // Move to input device before recording so the recorded output id stays consistent.
        let device = input.device();
        let mut out = Tensor::from_vec(output, &[n, c, h, w]).to_device(device);

        let any_requires_grad = input.requires_grad()
            || self.gamma.requires_grad()
            || self.beta.requires_grad();

        if graph::is_grad_enabled() && any_requires_grad && self.training {
            out.set_requires_grad(true);

            let input_ids = vec![input.id(), self.gamma.id(), self.beta.id()];

            let mut leaf_cells = Vec::new();
            if input.requires_grad() {
                leaf_cells.push((input.id(), input.grad_cell()));
            }
            if self.gamma.requires_grad() {
                leaf_cells.push((self.gamma.id(), self.gamma.grad_cell()));
            }
            if self.beta.requires_grad() {
                leaf_cells.push((self.beta.id(), self.beta.grad_cell()));
            }

            let grad_fn = Arc::new(BatchNorm2dBackward {
                input_ids,
                x_hat: Tensor::from_vec(x_hat_data, &[n, c, h, w]),
                gamma: gamma_vec,
                inv_std: inv_std_per_ch,
                device,
            });

            graph::record_op_with_cells(grad_fn, out.id(), leaf_cells);
        }

        out
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![self.gamma.clone(), self.beta.clone()]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![&mut self.gamma, &mut self.beta]
    }

    fn train(&mut self) { self.training = true; }
    fn eval(&mut self) { self.training = false; }
    fn is_training(&self) -> bool { self.training }

    fn to_device(&mut self, device: crate::tensor::Device) {
        self.gamma = self.gamma.to_device(device);
        self.beta = self.beta.to_device(device);
        self.running_mean = self.running_mean.to_device(device);
        self.running_var = self.running_var.to_device(device);
    }
}

/// Layer Normalization — normalizes across the feature dimension.
pub struct LayerNorm {
    pub gamma: Tensor,
    pub beta: Tensor,
    normalized_shape: Vec<usize>,
    epsilon: f32,
}

impl LayerNorm {
    pub fn new(normalized_shape: &[usize]) -> Self {
        let total: usize = normalized_shape.iter().product();
        let mut gamma = Tensor::ones(&[total]);
        gamma.set_requires_grad(true);
        let mut beta = Tensor::zeros(&[total]);
        beta.set_requires_grad(true);

        LayerNorm {
            gamma,
            beta,
            normalized_shape: normalized_shape.to_vec(),
            epsilon: 1e-5,
        }
    }
}

impl Module for LayerNorm {
    fn forward(&self, input: &Tensor) -> Tensor {
        let shape = input.shape().to_vec();
        let normalized_size: usize = self.normalized_shape.iter().product();
        let batch_size = input.numel() / normalized_size;
        let device = input.device();

        // ── CUDA fast path ──────────────────────────────────────────────────
        #[cfg(feature = "cuda")]
        if let (
            TensorStorage::Cuda(inp_buf),
            TensorStorage::Cuda(gam_buf),
            TensorStorage::Cuda(bet_buf),
        ) = (&input.storage, &self.gamma.storage, &self.beta.storage) {
            use crate::tensor::cuda_backend;
            let (out_buf, mean_buf, inv_var_buf) = cuda_backend::cuda_layer_norm_forward(
                inp_buf, gam_buf, bet_buf, batch_size, normalized_size, self.epsilon,
            ).expect("CUDA layer_norm_forward failed");

            let mut out = Tensor::from_cuda_buffer(out_buf, shape.clone(), false);

            let any_grad = input.requires_grad() || self.gamma.requires_grad() || self.beta.requires_grad();
            if graph::is_grad_enabled() && any_grad {
                out.set_requires_grad(true);
                let mean_t = Tensor::from_cuda_buffer(mean_buf, vec![batch_size], false);
                let inv_var_t = Tensor::from_cuda_buffer(inv_var_buf, vec![batch_size], false);
                let grad_fn = Arc::new(LayerNormCudaBackward {
                    input_ids: vec![input.id(), self.gamma.id(), self.beta.id()],
                    input: input.clone(),
                    gamma: self.gamma.clone(),
                    mean: mean_t,
                    inv_var: inv_var_t,
                    normalized_size,
                });
                let mut leaf_cells = vec![(input.id(), input.grad_cell())];
                if self.gamma.requires_grad() { leaf_cells.push((self.gamma.id(), self.gamma.grad_cell())); }
                if self.beta.requires_grad() { leaf_cells.push((self.beta.id(), self.beta.grad_cell())); }
                graph::record_op_with_cells(grad_fn, out.id(), leaf_cells);
            }
            return out;
        }

        // ── CPU path ────────────────────────────────────────────────────────
        let data = input.to_vec();
        let mut output = vec![0.0f32; data.len()];
        let gamma_data = self.gamma.to_vec();
        let beta_data = self.beta.to_vec();
        let mut x_hat_data = vec![0.0f32; data.len()];
        let mut inv_std_data = vec![0.0f32; batch_size];

        for b in 0..batch_size {
            let offset = b * normalized_size;
            let slice = &data[offset..offset + normalized_size];
            let mean: f32 = slice.iter().sum::<f32>() / normalized_size as f32;
            let var: f32 = slice.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / normalized_size as f32;
            let inv_std = 1.0 / (var + self.epsilon).sqrt();
            inv_std_data[b] = inv_std;
            for i in 0..normalized_size {
                let xh = (slice[i] - mean) * inv_std;
                x_hat_data[offset + i] = xh;
                output[offset + i] = gamma_data[i] * xh + beta_data[i];
            }
        }

        let mut out = Tensor::from_vec(output, &shape).to_device(device);

        if graph::is_grad_enabled() && input.requires_grad() {
            out.set_requires_grad(true);
            let grad_fn = Arc::new(LayerNormBackward {
                input_ids: vec![input.id()],
                x_hat: Tensor::from_vec(x_hat_data, &shape),
                gamma: gamma_data,
                inv_std: inv_std_data,
                normalized_size,
                device,
            });
            graph::record_op_with_cells(
                grad_fn,
                out.id(),
                vec![(input.id(), input.grad_cell())],
            );
        }

        out
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![self.gamma.clone(), self.beta.clone()]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![&mut self.gamma, &mut self.beta]
    }

    fn to_device(&mut self, device: crate::tensor::Device) {
        self.gamma = self.gamma.to_device(device);
        self.beta = self.beta.to_device(device);
    }
}

/// RMS Normalization — used in modern transformers (e.g., LLaMA).
pub struct RMSNorm {
    pub gamma: Tensor,
    normalized_size: usize,
    epsilon: f32,
}

impl RMSNorm {
    pub fn new(normalized_size: usize) -> Self {
        let mut gamma = Tensor::ones(&[normalized_size]);
        gamma.set_requires_grad(true);
        RMSNorm { gamma, normalized_size, epsilon: 1e-6 }
    }
}

impl Module for RMSNorm {
    fn forward(&self, input: &Tensor) -> Tensor {
        let shape = input.shape().to_vec();
        let batch_size = input.numel() / self.normalized_size;
        let data = input.to_vec();
        let gamma = self.gamma.to_vec();
        let mut output = vec![0.0f32; data.len()];

        for b in 0..batch_size {
            let offset = b * self.normalized_size;
            let slice = &data[offset..offset + self.normalized_size];
            let rms: f32 = (slice.iter().map(|&x| x * x).sum::<f32>() / self.normalized_size as f32 + self.epsilon).sqrt();
            let inv_rms = 1.0 / rms;
            for i in 0..self.normalized_size {
                output[offset + i] = gamma[i] * slice[i] * inv_rms;
            }
        }

        Tensor::from_vec(output, &shape).to_device(input.device())
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![self.gamma.clone()]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![&mut self.gamma]
    }

    fn to_device(&mut self, device: crate::tensor::Device) {
        self.gamma = self.gamma.to_device(device);
    }
}
