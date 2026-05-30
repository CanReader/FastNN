use std::sync::Arc;
use crate::tensor::Tensor;
use crate::nn::module::Module;
use crate::autograd::graph;
use crate::autograd::backward_ops::{MaxPool2dBackward, AvgPool2dBackward};

/// 2D Max Pooling.
pub struct MaxPool2d {
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
}

impl MaxPool2d {
    pub fn new(kernel_size: usize) -> Self {
        MaxPool2d {
            kernel_size: (kernel_size, kernel_size),
            stride: (kernel_size, kernel_size),
            padding: (0, 0),
        }
    }

    pub fn with_stride(kernel_size: usize, stride: usize) -> Self {
        MaxPool2d {
            kernel_size: (kernel_size, kernel_size),
            stride: (stride, stride),
            padding: (0, 0),
        }
    }
}

impl Module for MaxPool2d {
    fn forward(&self, input: &Tensor) -> Tensor {
        let shape = input.shape();
        assert_eq!(shape.len(), 4, "MaxPool2d expects [N, C, H, W]");
        let (n, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
        let (kh, kw) = self.kernel_size;
        let (sh, sw) = self.stride;
        let (ph, pw) = self.padding;

        let out_h = (h + 2 * ph - kh) / sh + 1;
        let out_w = (w + 2 * pw - kw) / sw + 1;

        let data = input.to_vec();
        let mut output = vec![0.0f32; n * c * out_h * out_w];
        let mut argmax = vec![0usize; n * c * out_h * out_w];

        for b in 0..n {
            for ch in 0..c {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let mut max_val = f32::NEG_INFINITY;
                        let mut max_idx = 0usize;
                        for ki in 0..kh {
                            for kj in 0..kw {
                                let ih = (oh * sh + ki) as isize - ph as isize;
                                let iw = (ow * sw + kj) as isize - pw as isize;
                                if ih >= 0 && ih < h as isize && iw >= 0 && iw < w as isize {
                                    let idx = ((b * c + ch) * h + ih as usize) * w + iw as usize;
                                    if data[idx] > max_val {
                                        max_val = data[idx];
                                        max_idx = idx;
                                    }
                                }
                            }
                        }
                        let out_idx = ((b * c + ch) * out_h + oh) * out_w + ow;
                        output[out_idx] = max_val;
                        argmax[out_idx] = max_idx;
                    }
                }
            }
        }

        let mut out = Tensor::from_vec(output, &[n, c, out_h, out_w]);

        if graph::is_grad_enabled() && input.requires_grad() {
            out.set_requires_grad(true);
            let grad_fn = Arc::new(MaxPool2dBackward {
                input_ids: vec![input.id()],
                input_shape: shape.to_vec(),
                argmax,
            });
            graph::record_op_with_cells(grad_fn, out.id(), vec![(input.id(), input.grad_cell())]);
        }

        out
    }
}

/// 2D Average Pooling.
pub struct AvgPool2d {
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
}

impl AvgPool2d {
    pub fn new(kernel_size: usize) -> Self {
        AvgPool2d {
            kernel_size: (kernel_size, kernel_size),
            stride: (kernel_size, kernel_size),
            padding: (0, 0),
        }
    }
}

impl Module for AvgPool2d {
    fn forward(&self, input: &Tensor) -> Tensor {
        let shape = input.shape();
        assert_eq!(shape.len(), 4);
        let (n, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
        let (kh, kw) = self.kernel_size;
        let (sh, sw) = self.stride;
        let (ph, pw) = self.padding;

        let out_h = (h + 2 * ph - kh) / sh + 1;
        let out_w = (w + 2 * pw - kw) / sw + 1;

        let data = input.to_vec();
        let mut output = vec![0.0f32; n * c * out_h * out_w];

        for b in 0..n {
            for ch in 0..c {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let mut sum = 0.0f32;
                        let mut count = 0;
                        for ki in 0..kh {
                            for kj in 0..kw {
                                let ih = (oh * sh + ki) as isize - ph as isize;
                                let iw = (ow * sw + kj) as isize - pw as isize;
                                if ih >= 0 && ih < h as isize && iw >= 0 && iw < w as isize {
                                    sum += data[((b * c + ch) * h + ih as usize) * w + iw as usize];
                                    count += 1;
                                }
                            }
                        }
                        output[((b * c + ch) * out_h + oh) * out_w + ow] = sum / count as f32;
                    }
                }
            }
        }

        let mut out = Tensor::from_vec(output, &[n, c, out_h, out_w]);

        if graph::is_grad_enabled() && input.requires_grad() {
            out.set_requires_grad(true);
            let grad_fn = Arc::new(AvgPool2dBackward {
                input_ids: vec![input.id()],
                input_shape: shape.to_vec(),
                kernel_size: self.kernel_size,
                stride: self.stride,
                padding: self.padding,
            });
            graph::record_op_with_cells(grad_fn, out.id(), vec![(input.id(), input.grad_cell())]);
        }

        out
    }
}

/// Adaptive Average Pooling — outputs a fixed spatial size.
pub struct AdaptiveAvgPool2d {
    output_size: (usize, usize),
}

impl AdaptiveAvgPool2d {
    pub fn new(output_size: (usize, usize)) -> Self {
        AdaptiveAvgPool2d { output_size }
    }

    /// Pool to (1,1) — global average pooling.
    pub fn global() -> Self {
        AdaptiveAvgPool2d { output_size: (1, 1) }
    }
}

impl Module for AdaptiveAvgPool2d {
    fn forward(&self, input: &Tensor) -> Tensor {
        let shape = input.shape();
        assert_eq!(shape.len(), 4);
        let (n, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
        let (oh_size, ow_size) = self.output_size;

        let data = input.to_vec();
        let mut output = vec![0.0f32; n * c * oh_size * ow_size];

        for b in 0..n {
            for ch in 0..c {
                for oh in 0..oh_size {
                    for ow in 0..ow_size {
                        let ih_start = oh * h / oh_size;
                        let ih_end = (oh + 1) * h / oh_size;
                        let iw_start = ow * w / ow_size;
                        let iw_end = (ow + 1) * w / ow_size;

                        let mut sum = 0.0f32;
                        let mut count = 0;
                        for ih in ih_start..ih_end {
                            for iw in iw_start..iw_end {
                                sum += data[((b * c + ch) * h + ih) * w + iw];
                                count += 1;
                            }
                        }
                        output[((b * c + ch) * oh_size + oh) * ow_size + ow] = sum / count as f32;
                    }
                }
            }
        }

        Tensor::from_vec(output, &[n, c, oh_size, ow_size])
    }
}
