//! Element-wise maths on a single tensor.

use crate::autograd::ops::unary::{
    AbsBackward, ClampBackward, ExpBackward, LogBackward, NegBackward, PowBackward, SqrtBackward,
};
use crate::cuda::{ffi, kernels};
use crate::tensor::storage::Storage;
use crate::tensor::Tensor;

use super::{scalar_op, unary_op};

impl Tensor {
    /// `-self`.
    pub fn neg(&self) -> Tensor {
        unary_op(self, |x| -x, ffi::fastnn_cuda_neg).with_grad(&[self], || NegBackward)
    }

    /// `e^self`.
    pub fn exp(&self) -> Tensor {
        let out = unary_op(self, f32::exp, ffi::fastnn_cuda_exp);
        let saved = out.clone();
        out.with_grad(&[self], || ExpBackward { output: saved })
    }

    /// Natural log. Undefined for non-positive inputs.
    pub fn log(&self) -> Tensor {
        unary_op(self, f32::ln, ffi::fastnn_cuda_log).with_grad(&[self], || LogBackward {
            input: self.detach(),
        })
    }

    /// `√self`.
    pub fn sqrt(&self) -> Tensor {
        let out = unary_op(self, f32::sqrt, ffi::fastnn_cuda_sqrt);
        let saved = out.clone();
        out.with_grad(&[self], || SqrtBackward { output: saved })
    }

    /// `|self|`. The derivative at exactly 0 is taken as 0.
    pub fn abs(&self) -> Tensor {
        unary_op(self, f32::abs, ffi::fastnn_cuda_abs).with_grad(&[self], || AbsBackward {
            input: self.detach(),
        })
    }

    /// `self` raised to a constant power.
    pub fn powf(&self, exponent: f32) -> Tensor {
        scalar_op(
            self,
            exponent,
            |x, p| x.powf(p),
            ffi::fastnn_cuda_pow_scalar,
        )
        .with_grad(&[self], || PowBackward {
            input: self.detach(),
            exponent,
        })
    }

    /// `self * self`.
    pub fn square(&self) -> Tensor {
        self.mul(self)
    }

    /// Clip every element into `[lo, hi]`. The gradient is zero outside the range.
    pub fn clamp(&self, lo: f32, hi: f32) -> Tensor {
        let out = match self.storage() {
            Storage::Cpu(data) => {
                Tensor::from_vec(super::map(data, |x| x.clamp(lo, hi)), self.shape())
            }
            Storage::Cuda(buf) => {
                let out = kernels::clamp(buf, lo, hi, self.numel()).expect("cuda clamp");
                Tensor::raw(Storage::Cuda(out), self.shape().to_vec(), self.device())
            }
        };
        out.with_grad(&[self], || ClampBackward {
            input: self.detach(),
            lo,
            hi,
        })
    }
}
