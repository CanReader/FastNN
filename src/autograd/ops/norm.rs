//! Derivatives of the fused normalization ops.
//!
//! Normalizing couples every element in a group: change one input and the mean
//! and variance move, so every output in that group moves. The two correction
//! terms below are exactly that coupling.

use crate::autograd::Backward;
use crate::cuda::kernels;
use crate::tensor::storage::Storage;
use crate::tensor::Tensor;

/// LayerNorm on the host. Returns gradients for `(input, gamma, beta)`.
pub struct LayerNormBackward {
    pub normalized: Tensor,
    pub gamma: Vec<f32>,
    pub inv_std: Vec<f32>,
    pub cols: usize,
}

impl Backward for LayerNormBackward {
    fn backward(&self, grad: &Tensor) -> Vec<Tensor> {
        let g = grad.to_vec();
        let unit = self.normalized.to_vec();
        let cols = self.cols;
        let width = cols as f32;
        let rows = g.len() / cols;

        let mut d_input = vec![0.0f32; g.len()];
        let mut d_gamma = vec![0.0f32; cols];
        let mut d_beta = vec![0.0f32; cols];

        for r in 0..rows {
            let base = r * cols;

            // Correction terms: how much the mean and the variance shift.
            let mut mean_shift = 0.0f32;
            let mut scale_shift = 0.0f32;
            for c in 0..cols {
                let scaled = self.gamma[c] * g[base + c];
                mean_shift += scaled;
                scale_shift += scaled * unit[base + c];
                d_gamma[c] += g[base + c] * unit[base + c];
                d_beta[c] += g[base + c];
            }

            for c in 0..cols {
                let scaled = self.gamma[c] * g[base + c];
                d_input[base + c] = self.inv_std[r]
                    * (scaled - mean_shift / width - unit[base + c] * scale_shift / width);
            }
        }

        let device = grad.device();
        vec![
            Tensor::from_vec(d_input, grad.shape()).to(device),
            Tensor::from_vec(d_gamma, &[cols]).to(device),
            Tensor::from_vec(d_beta, &[cols]).to(device),
        ]
    }
    fn name(&self) -> &'static str {
        "LayerNorm"
    }
}

/// LayerNorm on the GPU — one kernel produces all three gradients.
pub struct LayerNormCudaBackward {
    pub input: Tensor,
    pub gamma: Tensor,
    pub mean: Tensor,
    pub inv_std: Tensor,
    pub cols: usize,
}

impl Backward for LayerNormCudaBackward {
    fn backward(&self, grad: &Tensor) -> Vec<Tensor> {
        let rows = grad.numel() / self.cols;
        let buffers = (
            grad.storage().as_cuda(),
            self.input.storage().as_cuda(),
            self.gamma.storage().as_cuda(),
            self.mean.storage().as_cuda(),
            self.inv_std.storage().as_cuda(),
        );
        let (Some(g), Some(x), Some(gamma), Some(mean), Some(inv_std)) = buffers else {
            panic!("LayerNormCudaBackward reached with a non-CUDA gradient");
        };

        let (d_input, d_gamma, d_beta) =
            kernels::layer_norm_backward(g, x, gamma, mean, inv_std, rows, self.cols)
                .expect("cuda layer_norm backward");

        let device = grad.device();
        vec![
            Tensor::raw(Storage::Cuda(d_input), grad.shape().to_vec(), device),
            Tensor::raw(Storage::Cuda(d_gamma), vec![self.cols], device),
            Tensor::raw(Storage::Cuda(d_beta), vec![self.cols], device),
        ]
    }
    fn name(&self) -> &'static str {
        "LayerNormCuda"
    }
}

/// BatchNorm2d. Same structure as LayerNorm, but the group is a channel across
/// the whole batch rather than a row across features.
pub struct BatchNorm2dBackward {
    pub normalized: Tensor,
    pub gamma: Vec<f32>,
    pub inv_std: Vec<f32>,
}

impl Backward for BatchNorm2dBackward {
    fn backward(&self, grad: &Tensor) -> Vec<Tensor> {
        let (n, c, h, w) = (grad.dim(0), grad.dim(1), grad.dim(2), grad.dim(3));
        let plane = h * w;
        let count = (n * plane) as f32;

        let g = grad.to_vec();
        let unit = self.normalized.to_vec();
        let mut d_input = vec![0.0f32; g.len()];
        let mut d_gamma = vec![0.0f32; c];
        let mut d_beta = vec![0.0f32; c];

        for channel in 0..c {
            for image in 0..n {
                for i in 0..plane {
                    let index = (image * c + channel) * plane + i;
                    d_gamma[channel] += g[index] * unit[index];
                    d_beta[channel] += g[index];
                }
            }

            let scale = self.gamma[channel] * self.inv_std[channel];
            for image in 0..n {
                for i in 0..plane {
                    let index = (image * c + channel) * plane + i;
                    d_input[index] = scale
                        * (g[index]
                            - d_beta[channel] / count
                            - unit[index] * d_gamma[channel] / count);
                }
            }
        }

        let device = grad.device();
        vec![
            Tensor::from_vec(d_input, grad.shape()).to(device),
            Tensor::from_vec(d_gamma, &[c]).to(device),
            Tensor::from_vec(d_beta, &[c]).to(device),
        ]
    }
    fn name(&self) -> &'static str {
        "BatchNorm2d"
    }
}
