//! Derivatives of the `im2col` / `col2im` adjoint pair.
//!
//! The two operations are transposes of one linear map, so each is the
//! other's gradient: differentiating the unfold means folding the incoming
//! gradient, and differentiating the fold means unfolding it. Both rules
//! delegate to the same shared index maps the forwards use — the adjoint
//! identity holds by construction, not by keeping two loops in sync.

use crate::autograd::Backward;
use crate::cuda::kernels;
use crate::tensor::ops::conv::{fold, unfold, Window};
use crate::tensor::storage::Storage;
use crate::tensor::Tensor;

/// Gradient of `im2col`: fold columns back into the image, adding where
/// windows overlapped. A pixel covered by several windows was copied into
/// several columns, so its gradient is the sum of theirs; pixels that only
/// ever landed in padding get nothing.
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

/// Gradient of `col2im`: unfold the incoming image gradient back into columns.
///
/// Each column entry contributed to exactly one pixel, so its gradient is that
/// pixel's — which is precisely what `im2col` reads.
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
