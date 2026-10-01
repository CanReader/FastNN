//! 2-D pooling over `[N, C, H, W]` tensors.
//!
//! Pooling is a reduction with a routing decision, so unlike convolution it does
//! not reduce to a matrix multiply — each variant carries its own gradient rule.

use std::ops::Range;

use crate::autograd::ops::pool::{AdaptiveAvgPool2dBackward, AvgPool2dBackward, MaxPool2dBackward};
use crate::tensor::Tensor;

use super::conv::Window;

impl Tensor {
    /// Take the largest value in each window.
    ///
    /// The backward pass needs to know *which* input won, so the forward saves
    /// one flat index per output element.
    pub fn max_pool2d(&self, window: Window) -> Tensor {
        let (n, c, h, w) = self.expect_nchw("max_pool2d");
        let (out_h, out_w) = window.output_size(h, w);
        let data = self.to_vec();

        let mut out = vec![0.0f32; n * c * out_h * out_w];
        let mut winners = vec![0usize; out.len()];

        for plane in 0..n * c {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let (mut best, mut best_at) = (f32::NEG_INFINITY, plane * h * w);
                    for (ih, iw) in window.sources(oh, ow, h, w) {
                        let index = plane * h * w + ih * w + iw;
                        if data[index] > best {
                            best = data[index];
                            best_at = index;
                        }
                    }
                    let slot = (plane * out_h + oh) * out_w + ow;
                    out[slot] = best;
                    winners[slot] = best_at;
                }
            }
        }

        let shape = self.shape().to_vec();
        Tensor::from_vec(out, &[n, c, out_h, out_w])
            .to(self.device())
            .with_grad(&[self], || MaxPool2dBackward { shape, winners })
    }

    /// Average the values in each window, counting only real pixels (not padding).
    pub fn avg_pool2d(&self, window: Window) -> Tensor {
        let (n, c, h, w) = self.expect_nchw("avg_pool2d");
        let (out_h, out_w) = window.output_size(h, w);
        let data = self.to_vec();

        let mut out = vec![0.0f32; n * c * out_h * out_w];
        for plane in 0..n * c {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let mut total = 0.0f32;
                    let mut count = 0usize;
                    for (ih, iw) in window.sources(oh, ow, h, w) {
                        total += data[plane * h * w + ih * w + iw];
                        count += 1;
                    }
                    out[(plane * out_h + oh) * out_w + ow] = total / count as f32;
                }
            }
        }

        let shape = self.shape().to_vec();
        Tensor::from_vec(out, &[n, c, out_h, out_w])
            .to(self.device())
            .with_grad(&[self], || AvgPool2dBackward { shape, window })
    }

    /// Average into a fixed `(height, width)` output, whatever the input size.
    ///
    /// `(1, 1)` is global average pooling — the usual bridge from a conv stack
    /// to a classifier head.
    pub fn adaptive_avg_pool2d(&self, output: (usize, usize)) -> Tensor {
        let (n, c, h, w) = self.expect_nchw("adaptive_avg_pool2d");
        let (out_h, out_w) = output;
        let data = self.to_vec();

        let mut out = vec![0.0f32; n * c * out_h * out_w];
        for plane in 0..n * c {
            for oh in 0..out_h {
                let rows = adaptive_range(oh, h, out_h);
                for ow in 0..out_w {
                    let cols = adaptive_range(ow, w, out_w);
                    let count = (rows.len() * cols.len()) as f32;
                    let mut total = 0.0f32;
                    for ih in rows.clone() {
                        for iw in cols.clone() {
                            total += data[plane * h * w + ih * w + iw];
                        }
                    }
                    out[(plane * out_h + oh) * out_w + ow] = total / count;
                }
            }
        }

        let shape = self.shape().to_vec();
        Tensor::from_vec(out, &[n, c, out_h, out_w])
            .to(self.device())
            .with_grad(&[self], || AdaptiveAvgPool2dBackward { shape, output })
    }

    fn expect_nchw(&self, op: &str) -> (usize, usize, usize, usize) {
        assert_eq!(
            self.ndim(),
            4,
            "{op} expects [N, C, H, W], got {:?}",
            self.shape()
        );
        (self.dim(0), self.dim(1), self.dim(2), self.dim(3))
    }
}

impl Window {
    /// Input positions feeding output `(oh, ow)`, skipping padding.
    pub(crate) fn sources(
        &self,
        oh: usize,
        ow: usize,
        h: usize,
        w: usize,
    ) -> impl Iterator<Item = (usize, usize)> + '_ {
        let (kh, kw) = self.kernel;
        (0..kh).flat_map(move |ki| {
            let ih = offset(oh, ki, self.stride.0, self.padding.0, h);
            (0..kw).filter_map(move |kj| {
                let iw = offset(ow, kj, self.stride.1, self.padding.1, w);
                match (ih, iw) {
                    (Some(y), Some(x)) => Some((y, x)),
                    _ => None,
                }
            })
        })
    }
}

fn offset(out: usize, k: usize, stride: usize, pad: usize, limit: usize) -> Option<usize> {
    let pos = (out * stride + k) as isize - pad as isize;
    (pos >= 0 && (pos as usize) < limit).then_some(pos as usize)
}

/// The input span that output index `out` averages over.
pub(crate) fn adaptive_range(out: usize, in_size: usize, out_size: usize) -> Range<usize> {
    (out * in_size / out_size)..((out + 1) * in_size).div_ceil(out_size).min(in_size)
}
