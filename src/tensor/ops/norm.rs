//! Fused normalization ops.
//!
//! Both of these could be assembled from `mean`, `sub`, `sqrt`, and `div` — and
//! [`RMSNorm`](crate::nn::RMSNorm) is. They are fused instead because the
//! statistics are needed again in the backward pass, and recomputing them
//! through a chain of eight small ops costs more memory and more kernels than
//! saving two vectors.

use crate::autograd::ops::norm::{BatchNorm2dBackward, LayerNormBackward, LayerNormCudaBackward};
use crate::cuda::kernels;
use crate::tensor::storage::Storage;
use crate::tensor::Tensor;

/// Per-channel statistics of one batch, for updating running estimates.
pub struct BatchStats {
    pub mean: Vec<f32>,
    pub var: Vec<f32>,
}

impl Tensor {
    /// Normalize each row over the last dimension, then scale and shift.
    ///
    /// `gamma` and `beta` must both be `[last_dim]`.
    pub fn layer_norm(&self, gamma: &Tensor, beta: &Tensor, eps: f32) -> Tensor {
        let (rows, cols) = self.as_rows();
        assert_eq!(gamma.numel(), cols, "layer_norm: gamma must be [{cols}]");
        assert_eq!(beta.numel(), cols, "layer_norm: beta must be [{cols}]");

        if let (Storage::Cuda(x), Storage::Cuda(g), Storage::Cuda(b)) =
            (self.storage(), gamma.storage(), beta.storage())
        {
            let (out, mean, inv_std) =
                kernels::layer_norm_forward(x, g, b, rows, cols, eps).expect("cuda layer_norm");
            let out = Tensor::raw(Storage::Cuda(out), self.shape().to_vec(), self.device());
            let mean = Tensor::raw(Storage::Cuda(mean), vec![rows], self.device());
            let inv_std = Tensor::raw(Storage::Cuda(inv_std), vec![rows], self.device());
            return out.with_grad(&[self, gamma, beta], || LayerNormCudaBackward {
                input: self.detach(),
                gamma: gamma.detach(),
                mean,
                inv_std,
                cols,
            });
        }

        let (data, g, b) = (self.to_vec(), gamma.to_vec(), beta.to_vec());
        let mut out = vec![0.0f32; data.len()];
        let mut normalized = vec![0.0f32; data.len()];
        let mut inv_std = vec![0.0f32; rows];

        for r in 0..rows {
            let row = &data[r * cols..(r + 1) * cols];
            let mean = row.iter().sum::<f32>() / cols as f32;
            let var = row.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / cols as f32;
            inv_std[r] = 1.0 / (var + eps).sqrt();
            for c in 0..cols {
                let unit = (row[c] - mean) * inv_std[r];
                normalized[r * cols + c] = unit;
                out[r * cols + c] = g[c] * unit + b[c];
            }
        }

        let shape = self.shape().to_vec();
        Tensor::from_vec(out, &shape)
            .to(self.device())
            .with_grad(&[self, gamma, beta], || LayerNormBackward {
                normalized: Tensor::from_vec(normalized, &shape),
                gamma: g,
                inv_std,
                cols,
            })
    }

    /// Normalize `[N, C, H, W]` per channel using this batch's own statistics,
    /// then scale and shift. Training-mode only — at eval time the layer applies
    /// its running statistics as a plain affine transform.
    pub fn batch_norm2d(&self, gamma: &Tensor, beta: &Tensor, eps: f32) -> (Tensor, BatchStats) {
        assert_eq!(
            self.ndim(),
            4,
            "batch_norm2d expects [N, C, H, W], got {:?}",
            self.shape()
        );
        let (n, c, h, w) = (self.dim(0), self.dim(1), self.dim(2), self.dim(3));
        let plane = h * w;
        let count = (n * plane) as f32;

        let (data, g, b) = (self.to_vec(), gamma.to_vec(), beta.to_vec());
        let mut out = vec![0.0f32; data.len()];
        let mut normalized = vec![0.0f32; data.len()];
        let mut stats = BatchStats {
            mean: vec![0.0; c],
            var: vec![0.0; c],
        };
        let mut inv_std = vec![0.0f32; c];

        for channel in 0..c {
            let values = |image: usize, i: usize| data[(image * c + channel) * plane + i];

            let mean = (0..n)
                .flat_map(|i| (0..plane).map(move |p| (i, p)))
                .map(|(i, p)| values(i, p))
                .sum::<f32>()
                / count;
            let var = (0..n)
                .flat_map(|i| (0..plane).map(move |p| (i, p)))
                .map(|(i, p)| (values(i, p) - mean).powi(2))
                .sum::<f32>()
                / count;

            stats.mean[channel] = mean;
            stats.var[channel] = var;
            inv_std[channel] = 1.0 / (var + eps).sqrt();

            for image in 0..n {
                for i in 0..plane {
                    let index = (image * c + channel) * plane + i;
                    let unit = (data[index] - mean) * inv_std[channel];
                    normalized[index] = unit;
                    out[index] = g[channel] * unit + b[channel];
                }
            }
        }

        let shape = self.shape().to_vec();
        let result =
            Tensor::from_vec(out, &shape)
                .to(self.device())
                .with_grad(&[self, gamma, beta], || BatchNorm2dBackward {
                    normalized: Tensor::from_vec(normalized, &shape),
                    gamma: g,
                    inv_std,
                });
        (result, stats)
    }
}
