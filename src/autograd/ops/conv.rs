//! Derivatives of `im2col` and `col2im`. Each is the other's gradient, and both
//! use the forward's index maps so they stay exact adjoints.

use crate::autograd::Backward;
use crate::cuda::kernels;
use crate::tensor::Tensor;
use crate::tensor::ops::conv::{Window, fold, unfold};
use crate::tensor::storage::Storage;

/// Gradient of `im2col`: fold the columns back, summing overlaps.
pub struct Col2ImBackward {
    pub shape: Vec<usize>,
    pub window: Window,
}

impl Backward for Col2ImBackward {
    fn backward(&self, grad: &Tensor) -> Vec<Tensor> {
        let (n, c, h, w) = (self.shape[0], self.shape[1], self.shape[2], self.shape[3]);
        let (out_h, out_w) = self.window.output_size(h, w);

        if let Storage::Cuda(buf) = grad.storage() {
            let image = kernels::col2im(
                buf,
                (n, c, h, w),
                self.window.kernel,
                self.window.stride,
                self.window.padding,
                self.window.dilation,
                (out_h, out_w),
            )
            .expect("cuda col2im");
            return vec![Tensor::raw(
                Storage::Cuda(image),
                self.shape.clone(),
                grad.device(),
            )];
        }

        vec![Tensor::from_vec(
            fold(&grad.to_vec(), &self.shape, self.window),
            &self.shape,
        )]
    }
    fn name(&self) -> &'static str {
        "Col2Im"
    }
}

/// Gradient of `col2im`: unfold the image gradient.
pub struct Im2ColBackward {
    pub window: Window,
}

impl Backward for Im2ColBackward {
    fn backward(&self, grad: &Tensor) -> Vec<Tensor> {
        let shape = grad.shape().to_vec();
        let (n, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
        let (kh, kw) = self.window.kernel;
        let (out_h, out_w) = self.window.output_size(h, w);
        let cols_shape = [n, c * kh * kw, out_h * out_w];

        if let Storage::Cuda(buf) = grad.storage() {
            let cols = kernels::im2col(
                buf,
                (n, c, h, w),
                self.window.kernel,
                self.window.stride,
                self.window.padding,
                self.window.dilation,
                (out_h, out_w),
            )
            .expect("cuda im2col");
            return vec![Tensor::raw(
                Storage::Cuda(cols),
                cols_shape.to_vec(),
                grad.device(),
            )];
        }

        vec![Tensor::from_vec(
            unfold(&grad.to_vec(), &shape, self.window),
            &cols_shape,
        )]
    }
    fn name(&self) -> &'static str {
        "Im2Col"
    }
}
