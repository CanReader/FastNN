//! Normalization layers.

use std::sync::atomic::{AtomicBool, Ordering};

use crate::tensor::Tensor;

use super::module::Module;
use super::param::{Buffer, Param};

/// Normalize each sample over its last dimension, then scale and shift.
///
/// The workhorse of transformers: it depends on nothing but the sample itself,
/// so batch size and sequence length cannot change its behaviour.
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
            eps: 1e-5,
        }
    }

    /// Override the variance floor. Larger values trade accuracy for stability.
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
///
/// Cheaper than [`LayerNorm`] and used by most recent large models. Built from
/// ordinary tensor ops, so it needs no gradient rule of its own.
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
            eps: 1e-6,
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

/// Normalize each channel of `[N, C, H, W]` across the batch and spatial positions.
///
/// In training it uses the current batch's statistics and folds them into a
/// running estimate; in eval it applies that estimate as a plain affine map, so
/// a single image gives the same answer as a large batch.
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
            eps: 1e-5,
            momentum: 0.1,
            training: AtomicBool::new(true),
        }
    }

    /// How fast the running estimates follow the batch statistics.
    pub fn momentum(mut self, momentum: f32) -> BatchNorm2d {
        self.momentum = momentum;
        self
    }

    /// Blend this batch's statistics into the running estimates.
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
            // Frozen statistics make this an affine map, which ordinary ops express.
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
