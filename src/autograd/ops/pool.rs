//! Derivatives of pooling — each one routes the gradient the way the forward
//! pass routed the values.

use crate::autograd::Backward;
use crate::tensor::ops::conv::Window;
use crate::tensor::ops::pool::adaptive_range;
use crate::tensor::Tensor;

/// Only the winning input affected the output, so only it gets the gradient.
pub struct MaxPool2dBackward {
    pub shape: Vec<usize>,
    pub winners: Vec<usize>,
}

impl Backward for MaxPool2dBackward {
    fn backward(&self, grad: &Tensor) -> Vec<Tensor> {
        let g = grad.to_vec();
        let mut out = vec![0.0f32; self.shape.iter().product()];
        // `+=`, not `=`: overlapping windows can pick the same winner.
        for (slot, &winner) in self.winners.iter().enumerate() {
            out[winner] += g[slot];
        }
        vec![Tensor::from_vec(out, &self.shape).to(grad.device())]
    }
    fn name(&self) -> &'static str {
        "MaxPool2d"
    }
}

/// Every input in the window contributed `1/count`, so each gets that share.
pub struct AvgPool2dBackward {
    pub shape: Vec<usize>,
    pub window: Window,
}

impl Backward for AvgPool2dBackward {
    fn backward(&self, grad: &Tensor) -> Vec<Tensor> {
        let (n, c, h, w) = (self.shape[0], self.shape[1], self.shape[2], self.shape[3]);
        let (out_h, out_w) = self.window.output_size(h, w);
        let g = grad.to_vec();

        let mut out = vec![0.0f32; n * c * h * w];
        for plane in 0..n * c {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let count = self.window.sources(oh, ow, h, w).count() as f32;
                    let share = g[(plane * out_h + oh) * out_w + ow] / count;
                    for (ih, iw) in self.window.sources(oh, ow, h, w) {
                        out[plane * h * w + ih * w + iw] += share;
                    }
                }
            }
        }
        vec![Tensor::from_vec(out, &self.shape).to(grad.device())]
    }
    fn name(&self) -> &'static str {
        "AvgPool2d"
    }
}

/// Same idea, over the input-size-dependent spans the forward pass averaged.
pub struct AdaptiveAvgPool2dBackward {
    pub shape: Vec<usize>,
    pub output: (usize, usize),
}

impl Backward for AdaptiveAvgPool2dBackward {
    fn backward(&self, grad: &Tensor) -> Vec<Tensor> {
        let (n, c, h, w) = (self.shape[0], self.shape[1], self.shape[2], self.shape[3]);
        let (out_h, out_w) = self.output;
        let g = grad.to_vec();

        let mut out = vec![0.0f32; n * c * h * w];
        for plane in 0..n * c {
            for oh in 0..out_h {
                let rows = adaptive_range(oh, h, out_h);
                for ow in 0..out_w {
                    let cols = adaptive_range(ow, w, out_w);
                    let share =
                        g[(plane * out_h + oh) * out_w + ow] / (rows.len() * cols.len()) as f32;
                    for ih in rows.clone() {
                        for iw in cols.clone() {
                            out[plane * h * w + ih * w + iw] += share;
                        }
                    }
                }
            }
        }
        vec![Tensor::from_vec(out, &self.shape).to(grad.device())]
    }
    fn name(&self) -> &'static str {
        "AdaptiveAvgPool2d"
    }
}
