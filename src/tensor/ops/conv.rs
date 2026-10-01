//! `im2col` and `col2im`, the pair that turns convolution into matrix algebra.
//!
//! Unfolding each sliding window into a column lets [`Conv2d`](crate::nn::Conv2d)
//! be `weight · columns`, so it inherits cuBLAS on the GPU and a ready-made
//! derivative from [`matmul`](super::matmul). `col2im` is the exact adjoint —
//! the transpose of the same linear map — which makes it both `im2col`'s
//! gradient and the forward pass of transposed convolution: the two operations
//! are each other's backward rule.

use crate::autograd::ops::conv::{Col2ImBackward, Im2ColBackward};
use crate::cuda::kernels;
use crate::tensor::storage::Storage;
use crate::tensor::Tensor;

/// The numbers that describe a 2-D sliding window.
///
/// With dilation `d`, kernel tap `k` reads offset `k·d` — the kernel's
/// footprint stretches to `(k−1)·d + 1` pixels without adding weights.
#[derive(Clone, Copy, Debug)]
pub struct Window {
    pub kernel: (usize, usize),
    pub stride: (usize, usize),
    pub padding: (usize, usize),
    pub dilation: (usize, usize),
}

impl Window {
    /// A square window with the same value on both axes.
    pub fn square(kernel: usize, stride: usize, padding: usize) -> Window {
        Window {
            kernel: (kernel, kernel),
            stride: (stride, stride),
            padding: (padding, padding),
            dilation: (1, 1),
        }
    }

    /// The same window with square dilation `d`.
    pub fn dilated(mut self, dilation: usize) -> Window {
        assert!(dilation >= 1, "dilation must be at least 1");
        self.dilation = (dilation, dilation);
        self
    }

    /// How many pixels the kernel spans along one axis: `(k−1)·d + 1`.
    fn span(kernel: usize, dilation: usize) -> usize {
        (kernel - 1) * dilation + 1
    }

    /// Output height and width for an input of `(height, width)`.
    pub fn output_size(&self, height: usize, width: usize) -> (usize, usize) {
        let out = |size: usize, k: usize, s: usize, p: usize, d: usize| {
            let span = Window::span(k, d);
            assert!(
                size + 2 * p >= span,
                "window spanning {span} does not fit in {size} with padding {p}"
            );
            (size + 2 * p - span) / s + 1
        };
        (
            out(
                height,
                self.kernel.0,
                self.stride.0,
                self.padding.0,
                self.dilation.0,
            ),
            out(
                width,
                self.kernel.1,
                self.stride.1,
                self.padding.1,
                self.dilation.1,
            ),
        )
    }

    /// Source pixel for output position `out` and kernel tap `k`, or `None`
    /// if it falls in the padding.
    fn source(
        out: usize,
        k: usize,
        stride: usize,
        pad: usize,
        dilation: usize,
        limit: usize,
    ) -> Option<usize> {
        let pos = (out * stride + k * dilation) as isize - pad as isize;
        (pos >= 0 && (pos as usize) < limit).then_some(pos as usize)
    }
}

/// Unfold `[N, C, H, W]` image data into `[N, C·kh·kw, positions]` columns.
///
/// Shared by the `im2col` forward and the `col2im` backward — the adjoint pair
/// must use the same index map, and having it once makes that a fact rather
/// than a discipline.
pub(crate) fn unfold(src: &[f32], shape: &[usize], window: Window) -> Vec<f32> {
    let (n, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
    let (kh, kw) = window.kernel;
    let (out_h, out_w) = window.output_size(h, w);
    let (patch, positions) = (c * kh * kw, out_h * out_w);

    let mut cols = vec![0.0f32; n * patch * positions];
    for image in 0..n {
        for channel in 0..c {
            for ki in 0..kh {
                for kj in 0..kw {
                    let row = (channel * kh + ki) * kw + kj;
                    let dst_base = (image * patch + row) * positions;
                    let src_base = (image * c + channel) * h * w;
                    for oh in 0..out_h {
                        let Some(ih) = Window::source(
                            oh,
                            ki,
                            window.stride.0,
                            window.padding.0,
                            window.dilation.0,
                            h,
                        ) else {
                            continue;
                        };
                        for ow in 0..out_w {
                            let Some(iw) = Window::source(
                                ow,
                                kj,
                                window.stride.1,
                                window.padding.1,
                                window.dilation.1,
                                w,
                            ) else {
                                continue;
                            };
                            cols[dst_base + oh * out_w + ow] = src[src_base + ih * w + iw];
                        }
                    }
                }
            }
        }
    }
    cols
}

/// Fold `[N, C·kh·kw, positions]` columns back into `[N, C, H, W]`, *adding*
/// where windows overlapped — a pixel copied into several columns gets the sum
/// of their values back. The other half of the adjoint pair.
pub(crate) fn fold(cols: &[f32], shape: &[usize], window: Window) -> Vec<f32> {
    let (n, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
    let (kh, kw) = window.kernel;
    let (out_h, out_w) = window.output_size(h, w);
    let (patch, positions) = (c * kh * kw, out_h * out_w);

    let mut image = vec![0.0f32; n * c * h * w];
    for index in 0..n {
        for channel in 0..c {
            for ki in 0..kh {
                for kj in 0..kw {
                    let row = (channel * kh + ki) * kw + kj;
                    let src_base = (index * patch + row) * positions;
                    let dst_base = (index * c + channel) * h * w;
                    for oh in 0..out_h {
                        let Some(ih) = Window::source(
                            oh,
                            ki,
                            window.stride.0,
                            window.padding.0,
                            window.dilation.0,
                            h,
                        ) else {
                            continue;
                        };
                        for ow in 0..out_w {
                            let Some(iw) = Window::source(
                                ow,
                                kj,
                                window.stride.1,
                                window.padding.1,
                                window.dilation.1,
                                w,
                            ) else {
                                continue;
                            };
                            image[dst_base + ih * w + iw] += cols[src_base + oh * out_w + ow];
                        }
                    }
                }
            }
        }
    }
    image
}

impl Tensor {
    /// Unfold `[N, C, H, W]` into `[N, C·kh·kw, out_h·out_w]`.
    ///
    /// Column `l` holds every input pixel that feeds output pixel `l`, so a
    /// `[out_channels, C·kh·kw]` weight matrix times these columns is the
    /// convolution.
    pub fn im2col(&self, window: Window) -> Tensor {
        assert_eq!(
            self.ndim(),
            4,
            "im2col expects [N, C, H, W], got {:?}",
            self.shape()
        );
        let (n, c, h, w) = (self.dim(0), self.dim(1), self.dim(2), self.dim(3));
        let (kh, kw) = window.kernel;
        let (out_h, out_w) = window.output_size(h, w);
        let (patch, positions) = (c * kh * kw, out_h * out_w);
        let input_shape = self.shape().to_vec();

        if let Storage::Cuda(buf) = self.storage() {
            let cols = kernels::im2col(
                buf,
                (n, c, h, w),
                window.kernel,
                window.stride,
                window.padding,
                window.dilation,
                (out_h, out_w),
            )
            .expect("cuda im2col");
            return Tensor::raw(
                Storage::Cuda(cols),
                vec![n, patch, positions],
                self.device(),
            )
            .with_grad(&[self], || Col2ImBackward {
                shape: input_shape,
                window,
            });
        }

        Tensor::from_vec(
            unfold(&self.to_vec(), &input_shape, window),
            &[n, patch, positions],
        )
        .with_grad(&[self], || Col2ImBackward {
            shape: input_shape,
            window,
        })
    }

    /// Fold `[N, C·kh·kw, positions]` columns into a `[N, C, height, width]`
    /// image, summing where windows overlap — the adjoint of [`im2col`](Self::im2col).
    ///
    /// This is the forward pass of transposed convolution: what `im2col`
    /// gathers, `col2im` scatters back. Its gradient is therefore `im2col`
    /// itself.
    pub fn col2im(&self, window: Window, size: (usize, usize)) -> Tensor {
        assert_eq!(
            self.ndim(),
            3,
            "col2im expects [N, patch, positions], got {:?}",
            self.shape()
        );
        let (height, width) = size;
        let (kh, kw) = window.kernel;
        let (out_h, out_w) = window.output_size(height, width);
        let patch = self.dim(1);
        assert!(
            patch.is_multiple_of(kh * kw),
            "col2im: {patch} rows do not divide into {kh}×{kw} kernels"
        );
        assert_eq!(
            self.dim(2),
            out_h * out_w,
            "col2im: {} positions but a {height}×{width} image yields {}",
            self.dim(2),
            out_h * out_w
        );

        let (n, c) = (self.dim(0), patch / (kh * kw));
        let shape = vec![n, c, height, width];

        if let Storage::Cuda(buf) = self.storage() {
            let image = kernels::col2im(
                buf,
                (n, c, height, width),
                window.kernel,
                window.stride,
                window.padding,
                window.dilation,
                (out_h, out_w),
            )
            .expect("cuda col2im");
            return Tensor::raw(Storage::Cuda(image), shape, self.device())
                .with_grad(&[self], || Im2ColBackward { window });
        }

        Tensor::from_vec(fold(&self.to_vec(), &shape, window), &shape)
            .with_grad(&[self], || Im2ColBackward { window })
    }
}
