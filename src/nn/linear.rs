//! Fully connected layer.

use crate::tensor::Tensor;

use super::module::Module;
use super::param::Param;

/// `y = x · Wᵀ + b`.
///
/// Accepts any input whose last dimension is `in_features`; leading dimensions
/// are flattened into the batch and restored on the way out, so the same layer
/// serves `[batch, features]` and `[batch, sequence, features]`.
pub struct Linear {
    pub weight: Param,
    pub bias: Option<Param>,
    in_features: usize,
    out_features: usize,
}

impl Linear {
    /// A layer with bias, Kaiming-initialised.
    pub fn new(in_features: usize, out_features: usize) -> Linear {
        Linear {
            weight: Param::new(Tensor::kaiming_uniform(
                &[out_features, in_features],
                in_features,
            )),
            bias: Some(Param::new(uniform_bias(out_features, in_features))),
            in_features,
            out_features,
        }
    }

    /// A layer with no bias — the usual choice when a normalization layer with
    /// its own shift follows immediately.
    pub fn no_bias(in_features: usize, out_features: usize) -> Linear {
        Linear {
            bias: None,
            ..Linear::new(in_features, out_features)
        }
    }

    pub fn in_features(&self) -> usize {
        self.in_features
    }

    pub fn out_features(&self) -> usize {
        self.out_features
    }
}

impl Module for Linear {
    fn forward(&self, input: &Tensor) -> Tensor {
        assert_eq!(
            input.last_dim(),
            self.in_features,
            "Linear expects {} input features, got {:?}",
            self.in_features,
            input.shape()
        );

        // `matmul_nt` consumes the weight as stored, [out, in], with no transposed copy.
        let flat = input.reshape(&[-1, self.in_features as i64]);
        let mut out = flat.matmul_nt(&self.weight.tensor());

        if let Some(bias) = &self.bias {
            out = out.add(&bias.tensor().reshape(&[1, self.out_features as i64]));
        }

        let mut shape: Vec<i64> = input.shape()[..input.ndim() - 1]
            .iter()
            .map(|&d| d as i64)
            .collect();
        shape.push(self.out_features as i64);
        out.reshape(&shape)
    }

    fn named_parameters(&self) -> Vec<(String, Param)> {
        let mut params = vec![("weight".into(), self.weight.clone())];
        if let Some(bias) = &self.bias {
            params.push(("bias".into(), bias.clone()));
        }
        params
    }
}

/// PyTorch's linear bias init: uniform over `±1/√fan_in`.
fn uniform_bias(out_features: usize, fan_in: usize) -> Tensor {
    let bound = 1.0 / (fan_in as f32).sqrt();
    Tensor::uniform(&[out_features], -bound, bound)
}
