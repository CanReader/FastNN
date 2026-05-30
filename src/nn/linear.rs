use std::collections::HashMap;
use crate::tensor::Tensor;
use crate::nn::module::Module;

/// Fully connected (dense) layer: y = xW^T + b
pub struct Linear {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
    in_features: usize,
    out_features: usize,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        // Kaiming uniform initialization
        let weight = Tensor::kaiming_uniform(&[out_features, in_features], in_features);
        let bound = 1.0 / (in_features as f32).sqrt();
        let bias = Tensor::from_vec(
            (0..out_features).map(|_| {
                use rand::Rng;
                rand::thread_rng().gen_range(-bound..bound)
            }).collect(),
            &[out_features],
        );

        let mut w = weight;
        w.set_requires_grad(true);
        let mut b = bias;
        b.set_requires_grad(true);

        Linear {
            weight: w,
            bias: Some(b),
            in_features,
            out_features,
        }
    }

    /// Create a linear layer without bias.
    pub fn no_bias(in_features: usize, out_features: usize) -> Self {
        let weight = Tensor::kaiming_uniform(&[out_features, in_features], in_features);
        let mut w = weight;
        w.set_requires_grad(true);

        Linear {
            weight: w,
            bias: None,
            in_features,
            out_features,
        }
    }
}

impl Module for Linear {
    fn forward(&self, input: &Tensor) -> Tensor {
        // input: [batch, in_features] or [batch, seq_len, in_features]
        // weight: [out_features, in_features]
        // output: [batch, out_features] or [batch, seq_len, out_features]

        let input_shape = input.shape().to_vec();
        let ndim = input_shape.len();

        // Flatten all batch dimensions, keep last as features
        let features = input_shape[ndim - 1];
        assert_eq!(features, self.in_features,
                   "Expected input features {}, got {}", self.in_features, features);

        let batch_size: usize = input_shape[..ndim - 1].iter().product();

        // Reshape to [batch, in_features]
        let input_2d = input.reshape(&[batch_size as i64, self.in_features as i64]);

        // y = x @ W^T
        let weight_t = self.weight.transpose();
        let mut output = input_2d.matmul(&weight_t);

        // Add bias
        if let Some(ref bias) = self.bias {
            let bias_expanded = bias.reshape(&[1, self.out_features as i64])
                .expand(&[batch_size, self.out_features]);
            output = output.add(&bias_expanded);
        }

        // Reshape back to original batch dimensions
        let mut out_shape: Vec<i64> = input_shape[..ndim - 1].iter().map(|&s| s as i64).collect();
        out_shape.push(self.out_features as i64);
        output.reshape(&out_shape)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = vec![self.weight.clone()];
        if let Some(ref bias) = self.bias {
            params.push(bias.clone());
        }
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params = vec![&mut self.weight];
        if let Some(ref mut bias) = self.bias {
            params.push(bias);
        }
        params
    }

    fn named_parameters(&self) -> HashMap<String, Tensor> {
        let mut params = HashMap::new();
        params.insert("weight".to_string(), self.weight.clone());
        if let Some(ref bias) = self.bias {
            params.insert("bias".to_string(), bias.clone());
        }
        params
    }

    fn to_device(&mut self, device: crate::tensor::Device) {
        self.weight = self.weight.to_device(device);
        if let Some(ref mut b) = self.bias {
            *b = b.to_device(device);
        }
    }
}
