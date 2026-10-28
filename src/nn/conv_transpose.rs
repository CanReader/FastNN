//! Transposed convolution, the forward pass of which is the convolution's
//! backward pass: multiply by the transposed weight, then `col2im`.
//!
//! Output size is `(H−1)·s − 2p + (k−1)·d + 1 + output_padding`. The padding
//! picks between inputs that convolve to the same size (5 and 6 at stride 2).

use crate::tensor::{Tensor, Window};

use super::linear::uniform_bias;
use super::module::Module;
use super::param::Param;

/// Transposed 2-D convolution over `[N, C, H, W]`.
pub struct ConvTranspose2d {
    /// `[in_channels, out_channels, kh, kw]`, the reverse of [`Conv2d`](super::Conv2d)'s layout.
    pub weight: Param,
    pub bias: Option<Param>,
    window: Window,
    in_channels: usize,
    out_channels: usize,
    output_padding: (usize, usize),
}

impl ConvTranspose2d {
    /// A square kernel with the given stride and padding.
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel: usize,
        stride: usize,
        padding: usize,
    ) -> ConvTranspose2d {
        ConvTranspose2d::with_window(
            in_channels,
            out_channels,
            Window::square(kernel, stride, padding),
            true,
        )
    }

    /// Full control over the window and whether there is a bias.
    pub fn with_window(
        in_channels: usize,
        out_channels: usize,
        window: Window,
        bias: bool,
    ) -> ConvTranspose2d {
        let (kh, kw) = window.kernel;
        let fan_in = in_channels * kh * kw;
        ConvTranspose2d {
            weight: Param::new(Tensor::kaiming_uniform(
                &[in_channels, out_channels, kh, kw],
                fan_in,
            )),
            bias: bias.then(|| Param::new(uniform_bias(out_channels, fan_in))),
            window,
            in_channels,
            out_channels,
            output_padding: (0, 0),
        }
    }

    /// Extra rows and columns on the far edge. Must be smaller than the stride.
    pub fn output_padding(mut self, padding: usize) -> ConvTranspose2d {
        assert!(
            padding < self.window.stride.0 && padding < self.window.stride.1,
            "output_padding {padding} must be smaller than the stride {:?}",
            self.window.stride
        );
        self.output_padding = (padding, padding);
        self
    }

    /// Output spatial size for an input of `(height, width)`.
    pub fn output_size(&self, height: usize, width: usize) -> (usize, usize) {
        let out = |size: usize, k: usize, s: usize, p: usize, d: usize, a: usize| {
            (size - 1) * s + (k - 1) * d + 1 + a - 2 * p
        };
        (
            out(
                height,
                self.window.kernel.0,
                self.window.stride.0,
                self.window.padding.0,
                self.window.dilation.0,
                self.output_padding.0,
            ),
            out(
                width,
                self.window.kernel.1,
                self.window.stride.1,
                self.window.padding.1,
                self.window.dilation.1,
                self.output_padding.1,
            ),
        )
    }
}

impl Module for ConvTranspose2d {
    fn forward(&self, input: &Tensor) -> Tensor {
        assert_eq!(
            input.ndim(),
            4,
            "ConvTranspose2d expects [N, C, H, W], got {:?}",
            input.shape()
        );
        assert_eq!(
            input.dim(1),
            self.in_channels,
            "ConvTranspose2d expects {} channels, got {:?}",
            self.in_channels,
            input.shape()
        );

        let (batch, height, width) = (input.dim(0), input.dim(2), input.dim(3));
        let (out_h, out_w) = self.output_size(height, width);

        // matmul_tn reads the weight transposed in place and broadcasts it over the batch
        let flat = input.reshape(&[
            batch as i64,
            self.in_channels as i64,
            (height * width) as i64,
        ]);
        let kernels = self.weight.tensor().reshape(&[self.in_channels as i64, -1]);
        let columns = kernels.matmul_tn(&flat);

        let mut out = columns.col2im(self.window, (out_h, out_w));
        if let Some(bias) = &self.bias {
            out = out.add(&bias.tensor().reshape(&[1, self.out_channels as i64, 1, 1]));
        }
        out
    }

    fn named_parameters(&self) -> Vec<(String, Param)> {
        let mut params = vec![("weight".into(), self.weight.clone())];
        if let Some(bias) = &self.bias {
            params.push(("bias".into(), bias.clone()));
        }
        params
    }
}

/// Transposed 1-D convolution over `[N, C, L]`, the 2-D layer on a 1-pixel-tall image.
pub struct ConvTranspose1d {
    inner: ConvTranspose2d,
}

impl ConvTranspose1d {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel: usize,
        stride: usize,
        padding: usize,
    ) -> ConvTranspose1d {
        let window = Window {
            kernel: (1, kernel),
            stride: (1, stride),
            padding: (0, padding),
            dilation: (1, 1),
        };
        ConvTranspose1d {
            inner: ConvTranspose2d::with_window(in_channels, out_channels, window, true),
        }
    }
}

impl Module for ConvTranspose1d {
    fn forward(&self, input: &Tensor) -> Tensor {
        assert_eq!(
            input.ndim(),
            3,
            "ConvTranspose1d expects [N, C, L], got {:?}",
            input.shape()
        );
        let lifted = input.unsqueeze(2);
        self.inner.forward(&lifted).squeeze(2)
    }

    fn named_parameters(&self) -> Vec<(String, Param)> {
        self.inner.named_parameters()
    }
}
