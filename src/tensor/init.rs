//! Ways to make a tensor full of values.
//!
//! Everything here builds on the CPU. Chain `.to(device)` to place the result,
//! or let a layer's `to_device` move its parameters after construction.

use crate::rng;

use super::shape;
use super::Tensor;

impl Tensor {
    /// All zeros.
    pub fn zeros(shape: &[usize]) -> Tensor {
        Tensor::full(shape, 0.0)
    }

    /// All ones.
    pub fn ones(shape: &[usize]) -> Tensor {
        Tensor::full(shape, 1.0)
    }

    /// Every element set to `value`.
    pub fn full(shape: &[usize], value: f32) -> Tensor {
        Tensor::from_vec(vec![value; shape::numel(shape)], shape)
    }

    /// Zeros shaped like `other`, on the same device.
    pub fn zeros_like(other: &Tensor) -> Tensor {
        Tensor::zeros(other.shape()).to(other.device())
    }

    /// Uniform samples from `[0, 1)`.
    pub fn rand(shape: &[usize]) -> Tensor {
        Tensor::from_vec(rng::uniform(shape::numel(shape), 0.0, 1.0), shape)
    }

    /// Standard normal samples.
    pub fn randn(shape: &[usize]) -> Tensor {
        Tensor::from_vec(rng::normal(shape::numel(shape)), shape)
    }

    /// Uniform samples from `[lo, hi)`.
    pub fn uniform(shape: &[usize], lo: f32, hi: f32) -> Tensor {
        Tensor::from_vec(rng::uniform(shape::numel(shape), lo, hi), shape)
    }

    /// Kaiming (He) uniform — the default for ReLU-family layers.
    ///
    /// Bounds are `±√(6 / fan_in)`, which keeps activation variance roughly
    /// constant through a stack of rectified layers.
    pub fn kaiming_uniform(shape: &[usize], fan_in: usize) -> Tensor {
        let bound = (6.0 / fan_in as f32).sqrt();
        Tensor::uniform(shape, -bound, bound)
    }

    /// Xavier (Glorot) uniform — for `tanh`/`sigmoid` layers, where the backward
    /// variance matters as much as the forward.
    pub fn xavier_uniform(shape: &[usize], fan_in: usize, fan_out: usize) -> Tensor {
        let bound = (6.0 / (fan_in + fan_out) as f32).sqrt();
        Tensor::uniform(shape, -bound, bound)
    }

    /// Values from `start` up to (not including) `end`, stepping by `step`.
    pub fn arange(start: f32, end: f32, step: f32) -> Tensor {
        assert!(step > 0.0, "arange: step must be positive, got {step}");
        let n = ((end - start) / step).ceil().max(0.0) as usize;
        let data: Vec<f32> = (0..n).map(|i| start + step * i as f32).collect();
        Tensor::from_vec(data, &[n])
    }

    /// `n` evenly spaced values from `start` to `end`, both included.
    pub fn linspace(start: f32, end: f32, n: usize) -> Tensor {
        let step = if n > 1 {
            (end - start) / (n - 1) as f32
        } else {
            0.0
        };
        let data: Vec<f32> = (0..n).map(|i| start + step * i as f32).collect();
        Tensor::from_vec(data, &[n])
    }

    /// The `n × n` identity matrix.
    pub fn eye(n: usize) -> Tensor {
        let mut data = vec![0.0f32; n * n];
        for i in 0..n {
            data[i * n + i] = 1.0;
        }
        Tensor::from_vec(data, &[n, n])
    }

    /// Rows of the identity, selected by `indices` — one-hot targets in `classes` classes.
    pub fn one_hot(indices: &[usize], classes: usize) -> Tensor {
        let mut data = vec![0.0f32; indices.len() * classes];
        for (row, &class) in indices.iter().enumerate() {
            assert!(
                class < classes,
                "one_hot: class {class} outside 0..{classes}"
            );
            data[row * classes + class] = 1.0;
        }
        Tensor::from_vec(data, &[indices.len(), classes])
    }
}
