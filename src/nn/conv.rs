//! 2-D convolution.

use crate::tensor::{Tensor, Window};

use super::module::Module;
use super::param::Param;

/// Convolution over `[N, C, H, W]`, implemented as unfold-then-multiply.
///
/// [`im2col`](Tensor::im2col) lays every sliding window out as a column, which
/// turns the convolution into one matrix multiply against a `[out_channels,
/// C·kh·kw]` weight. That is why this layer needs no gradient code of its own:
/// the derivatives of `im2col` and `matmul` already compose into the right thing.
pub struct Conv2d {
    pub weight: Param,
    pub bias: Option<Param>,
    window: Window,
    in_channels: usize,
    out_channels: usize,
    groups: usize,
}

impl Conv2d {
    /// A square kernel with the given stride and padding.
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel: usize,
        stride: usize,
        padding: usize,
    ) -> Conv2d {
        Conv2d::with_window(
            in_channels,
            out_channels,
            Window::square(kernel, stride, padding),
            true,
        )
    }

    /// `kernel × kernel`, stride 1, padded to preserve the spatial size.
    pub fn same(in_channels: usize, out_channels: usize, kernel: usize) -> Conv2d {
        assert!(
            kernel % 2 == 1,
            "same-padding needs an odd kernel, got {kernel}"
        );
        Conv2d::new(in_channels, out_channels, kernel, 1, kernel / 2)
    }

    /// Full control over the window and whether there is a bias.
    pub fn with_window(
        in_channels: usize,
        out_channels: usize,
        window: Window,
        bias: bool,
    ) -> Conv2d {
        Conv2d::build(in_channels, out_channels, window, bias, 1)
    }

    /// Split the channels into `groups` independent convolutions.
    ///
    /// Each group sees only `in/groups` input channels and produces
    /// `out/groups` outputs, cutting parameters and compute by the group
    /// count. The weight is `[out, in/groups, kh, kw]`.
    pub fn grouped(
        in_channels: usize,
        out_channels: usize,
        kernel: usize,
        stride: usize,
        padding: usize,
        groups: usize,
    ) -> Conv2d {
        Conv2d::build(
            in_channels,
            out_channels,
            Window::square(kernel, stride, padding),
            true,
            groups,
        )
    }

    /// One filter per channel — `groups == channels`, the spatial half of a
    /// depthwise-separable convolution. Follow with a 1×1 [`Conv2d`] to mix
    /// channels back together.
    pub fn depthwise(channels: usize, kernel: usize, stride: usize, padding: usize) -> Conv2d {
        Conv2d::grouped(channels, channels, kernel, stride, padding, channels)
    }

    fn build(
        in_channels: usize,
        out_channels: usize,
        window: Window,
        bias: bool,
        groups: usize,
    ) -> Conv2d {
        assert!(groups >= 1, "groups must be at least 1");
        assert_eq!(
            in_channels % groups,
            0,
            "{in_channels} input channels do not split into {groups} groups"
        );
        assert_eq!(
            out_channels % groups,
            0,
            "{out_channels} output channels do not split into {groups} groups"
        );

        let (kh, kw) = window.kernel;
        let fan_in = (in_channels / groups) * kh * kw;
        Conv2d {
            weight: Param::new(Tensor::kaiming_uniform(
                &[out_channels, in_channels / groups, kh, kw],
                fan_in,
            )),
            bias: bias.then(|| {
                let bound = 1.0 / (fan_in as f32).sqrt();
                Param::new(Tensor::uniform(&[out_channels], -bound, bound))
            }),
            window,
            in_channels,
            out_channels,
            groups,
        }
    }

    /// Output spatial size for an input of `(height, width)`.
    pub fn output_size(&self, height: usize, width: usize) -> (usize, usize) {
        self.window.output_size(height, width)
    }
}

impl Module for Conv2d {
    fn forward(&self, input: &Tensor) -> Tensor {
        assert_eq!(
            input.ndim(),
            4,
            "Conv2d expects [N, C, H, W], got {:?}",
            input.shape()
        );
        assert_eq!(
            input.dim(1),
            self.in_channels,
            "Conv2d expects {} channels, got {:?}",
            self.in_channels,
            input.shape()
        );

        let (batch, height, width) = (input.dim(0), input.dim(2), input.dim(3));
        let (out_h, out_w) = self.window.output_size(height, width);
        let out_c = self.out_channels as i64;

        // [N, C·kh·kw, out_h·out_w] against [out_channels, C·kh·kw]; with
        // groups, the same product runs once per channel slice and the results
        // stack back along the channel axis.
        let mut out = if self.groups == 1 {
            let columns = input.im2col(self.window);
            self.weight.tensor().reshape(&[out_c, -1]).matmul(&columns)
        } else {
            let (in_per, out_per) = (
                self.in_channels / self.groups,
                self.out_channels / self.groups,
            );
            let pieces: Vec<Tensor> = (0..self.groups)
                .map(|group| {
                    let columns = input.narrow(1, group * in_per, in_per).im2col(self.window);
                    let kernels = self
                        .weight
                        .tensor()
                        .narrow(0, group * out_per, out_per)
                        .reshape(&[out_per as i64, -1]);
                    kernels.matmul(&columns)
                })
                .collect();
            Tensor::cat(&pieces.iter().collect::<Vec<_>>(), 1)
        };

        if let Some(bias) = &self.bias {
            out = out.add(&bias.tensor().reshape(&[1, out_c, 1]));
        }

        out.reshape(&[batch as i64, out_c, out_h as i64, out_w as i64])
    }

    fn named_parameters(&self) -> Vec<(String, Param)> {
        let mut params = vec![("weight".into(), self.weight.clone())];
        if let Some(bias) = &self.bias {
            params.push(("bias".into(), bias.clone()));
        }
        params
    }
}
