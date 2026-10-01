//! 1-D convolution over sequences.
//!
//! A sequence is an image one pixel tall, so `Conv1d` *is* [`Conv2d`] with a
//! `1×k` kernel — same unfold, same GEMM, same kernels on both devices. What
//! this file adds is the sequence-specific option: causal padding, where all
//! `(k−1)·d` pad positions go on the *left*, so output `t` reads only inputs
//! `≤ t`. Stack causal layers with growing dilation and the receptive field
//! doubles per layer while each stays one convolution — the WaveNet
//! construction, and the convolutional counterpart of a causal mask.

use crate::tensor::{Tensor, Window};

use super::conv::Conv2d;
use super::module::Module;
use super::param::Param;

/// Convolution over `[N, C, L]`.
pub struct Conv1d {
    inner: Conv2d,
    /// Zeros prepended to the sequence; nonzero only for causal layers.
    left_pad: usize,
}

impl Conv1d {
    /// Symmetric padding, as in the 2-D case.
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel: usize,
        stride: usize,
        padding: usize,
    ) -> Conv1d {
        let window = Window {
            kernel: (1, kernel),
            stride: (1, stride),
            padding: (0, padding),
            dilation: (1, 1),
        };
        Conv1d {
            inner: Conv2d::with_window(in_channels, out_channels, window, true),
            left_pad: 0,
        }
    }

    /// Causal, length-preserving, optionally dilated: output `t` depends only
    /// on inputs `≤ t`.
    pub fn causal(
        in_channels: usize,
        out_channels: usize,
        kernel: usize,
        dilation: usize,
    ) -> Conv1d {
        let window = Window {
            kernel: (1, kernel),
            stride: (1, 1),
            padding: (0, 0), // all padding is explicit, on the left
            dilation: (1, dilation),
        };
        Conv1d {
            inner: Conv2d::with_window(in_channels, out_channels, window, true),
            left_pad: (kernel - 1) * dilation,
        }
    }
}

impl Module for Conv1d {
    fn forward(&self, input: &Tensor) -> Tensor {
        assert_eq!(
            input.ndim(),
            3,
            "Conv1d expects [N, C, L], got {:?}",
            input.shape()
        );

        // Left-only zero padding, done explicitly: Window padding is symmetric
        // by design, and causality is exactly the asymmetric case.
        let padded = if self.left_pad > 0 {
            let zeros =
                Tensor::zeros(&[input.dim(0), input.dim(1), self.left_pad]).to(input.device());
            Tensor::cat(&[&zeros, input], 2)
        } else {
            input.clone()
        };

        let lifted = padded.unsqueeze(2); // [N, C, 1, L]
        self.inner.forward(&lifted).squeeze(2)
    }

    fn named_parameters(&self) -> Vec<(String, Param)> {
        self.inner.named_parameters()
    }
}
