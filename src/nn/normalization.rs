use crate::tensor::Tensor;
use crate::nn::module::Module;

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

        for ch in 0..c {
            let (mean, var) = if self.training {
                // Compute batch statistics
                let mut sum = 0.0f32;
                let count = (n * spatial) as f32;
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
                let var = var_sum / count;
                (mean, var)
            } else {
                (self.running_mean.data()[ch], self.running_var.data()[ch])
            };

            let inv_std = 1.0 / (var + self.epsilon).sqrt();
            let gamma = self.gamma.data()[ch];
            let beta = self.beta.data()[ch];

            for b in 0..n {
                for s in 0..spatial {
                    let idx = ((b * c + ch) * h + s / w) * w + s % w;
                    output[idx] = gamma * (data[idx] - mean) * inv_std + beta;
                }
            }
        }

        Tensor::from_vec(output, &[n, c, h, w])
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![self.gamma.clone(), self.beta.clone()]
    }

    fn train(&mut self) { self.training = true; }
    fn eval(&mut self) { self.training = false; }
    fn is_training(&self) -> bool { self.training }
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

        let data = input.to_vec();
        let mut output = vec![0.0f32; data.len()];
        let gamma = self.gamma.data();
        let beta = self.beta.data();

        for b in 0..batch_size {
            let offset = b * normalized_size;
            let slice = &data[offset..offset + normalized_size];

            let mean: f32 = slice.iter().sum::<f32>() / normalized_size as f32;
            let var: f32 = slice.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / normalized_size as f32;
            let inv_std = 1.0 / (var + self.epsilon).sqrt();

            for i in 0..normalized_size {
                output[offset + i] = gamma[i] * (slice[i] - mean) * inv_std + beta[i];
            }
        }

        Tensor::from_vec(output, &shape)
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![self.gamma.clone(), self.beta.clone()]
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
        let gamma = self.gamma.data();
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

        Tensor::from_vec(output, &shape)
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![self.gamma.clone()]
    }
}
