//! Normalization layers.

use std::sync::atomic::{AtomicBool, Ordering};

use crate::tensor::Tensor;

use super::module::Module;
use super::param::{Buffer, Param};

const LAYER_NORM_EPS: f32 = 1e-5;
const RMS_NORM_EPS: f32 = 1e-6;
const BATCH_NORM_EPS: f32 = 1e-5;
const BATCH_NORM_MOMENTUM: f32 = 0.1;

/// Normalize each sample over its last dimension, then scale and shift.
pub struct LayerNorm {
    pub gamma: Param,
    pub beta: Param,
    size: usize,
    eps: f32,
}

impl LayerNorm {
    pub fn new(size: usize) -> LayerNorm {
        LayerNorm {
            gamma: Param::new(Tensor::ones(&[size])),
            beta: Param::new(Tensor::zeros(&[size])),
            size,
            eps: LAYER_NORM_EPS,
        }
    }

    /// Override the variance floor.
    pub fn eps(mut self, eps: f32) -> LayerNorm {
        self.eps = eps;
        self
    }
}

impl Module for LayerNorm {
    fn forward(&self, input: &Tensor) -> Tensor {
        assert_eq!(
            input.last_dim(),
            self.size,
            "LayerNorm expects {} features, got {:?}",
            self.size,
            input.shape()
        );
        input.layer_norm(&self.gamma.tensor(), &self.beta.tensor(), self.eps)
    }

    fn named_parameters(&self) -> Vec<(String, Param)> {
        vec![
            ("gamma".into(), self.gamma.clone()),
            ("beta".into(), self.beta.clone()),
        ]
    }
}

/// Scale by the root-mean-square of each sample, with no mean subtraction.
pub struct RMSNorm {
    pub gamma: Param,
    size: usize,
    eps: f32,
}

impl RMSNorm {
    pub fn new(size: usize) -> RMSNorm {
        RMSNorm {
            gamma: Param::new(Tensor::ones(&[size])),
            size,
            eps: RMS_NORM_EPS,
        }
    }
}

impl Module for RMSNorm {
    fn forward(&self, input: &Tensor) -> Tensor {
        assert_eq!(
            input.last_dim(),
            self.size,
            "RMSNorm expects {} features, got {:?}",
            self.size,
            input.shape()
        );
        let axis = input.ndim() - 1;
        let scale = input
            .square()
            .mean_axis_keep(axis)
            .add_scalar(self.eps)
            .sqrt();
        input.div(&scale).mul(&self.gamma.tensor())
    }

    fn named_parameters(&self) -> Vec<(String, Param)> {
        vec![("gamma".into(), self.gamma.clone())]
    }
}

/// Normalize each channel of `[N, C, H, W]` over the batch and spatial positions.
/// Eval mode uses the running statistics instead of the batch's.
pub struct BatchNorm2d {
    pub gamma: Param,
    pub beta: Param,
    pub running_mean: Buffer,
    pub running_var: Buffer,
    channels: usize,
    eps: f32,
    momentum: f32,
    training: AtomicBool,
}

impl BatchNorm2d {
    pub fn new(channels: usize) -> BatchNorm2d {
        BatchNorm2d {
            gamma: Param::new(Tensor::ones(&[channels])),
            beta: Param::new(Tensor::zeros(&[channels])),
            running_mean: Buffer::new(Tensor::zeros(&[channels])),
            running_var: Buffer::new(Tensor::ones(&[channels])),
            channels,
            eps: BATCH_NORM_EPS,
            momentum: BATCH_NORM_MOMENTUM,
            training: AtomicBool::new(true),
        }
    }

    /// How fast the running estimates follow the batch statistics.
    pub fn momentum(mut self, momentum: f32) -> BatchNorm2d {
        self.momentum = momentum;
        self
    }

    fn update_running(&self, mean: &[f32], var: &[f32]) {
        let blend = |old: Tensor, new: &[f32]| {
            let old = old.to_vec();
            let mixed: Vec<f32> = old
                .iter()
                .zip(new)
                .map(|(&o, &n)| (1.0 - self.momentum) * o + self.momentum * n)
                .collect();
            Tensor::from_vec(mixed, &[self.channels])
        };
        let device = self.running_mean.value().device();
        self.running_mean
            .set_value(blend(self.running_mean.value(), mean).to(device));
        self.running_var
            .set_value(blend(self.running_var.value(), var).to(device));
    }
}

impl Module for BatchNorm2d {
    fn forward(&self, input: &Tensor) -> Tensor {
        assert_eq!(
            input.dim(1),
            self.channels,
            "BatchNorm2d expects {} channels, got {:?}",
            self.channels,
            input.shape()
        );
        let per_channel = [1i64, self.channels as i64, 1, 1];

        if !self.training.load(Ordering::Relaxed) {
            let scale = self.running_var.value().add_scalar(self.eps).sqrt();
            let centred = input.sub(&self.running_mean.value().reshape(&per_channel));
            return centred
                .div(&scale.reshape(&per_channel))
                .mul(&self.gamma.tensor().reshape(&per_channel))
                .add(&self.beta.tensor().reshape(&per_channel));
        }

        let (out, stats) = input.batch_norm2d(&self.gamma.tensor(), &self.beta.tensor(), self.eps);
        self.update_running(&stats.mean, &stats.var);
        out
    }

    fn named_parameters(&self) -> Vec<(String, Param)> {
        vec![
            ("gamma".into(), self.gamma.clone()),
            ("beta".into(), self.beta.clone()),
        ]
    }

    fn named_buffers(&self) -> Vec<(String, Buffer)> {
        vec![
            ("running_mean".into(), self.running_mean.clone()),
            ("running_var".into(), self.running_var.clone()),
        ]
    }

    fn set_training(&self, training: bool) {
        self.training.store(training, Ordering::Relaxed);
    }
}
