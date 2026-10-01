//! Element-wise arithmetic, with NumPy broadcasting.

use crate::autograd::ops::arith::{
    AddBackward, DivBackward, MulBackward, MulScalarBackward, SubBackward,
};
use crate::cuda::ffi;
use crate::tensor::Tensor;

use super::{binary_op, scalar_op};

impl Tensor {
    /// `self + other`, broadcasting.
    pub fn add(&self, other: &Tensor) -> Tensor {
        let (a, b) = (self.shape().to_vec(), other.shape().to_vec());
        binary_op(self, other, |x, y| x + y, ffi::fastnn_cuda_add)
            .with_grad(&[self, other], || AddBackward { a, b })
    }

    /// `self - other`, broadcasting.
    pub fn sub(&self, other: &Tensor) -> Tensor {
        let (a, b) = (self.shape().to_vec(), other.shape().to_vec());
        binary_op(self, other, |x, y| x - y, ffi::fastnn_cuda_sub)
            .with_grad(&[self, other], || SubBackward { a, b })
    }

    /// `self * other` element-wise (Hadamard), broadcasting.
    pub fn mul(&self, other: &Tensor) -> Tensor {
        binary_op(self, other, |x, y| x * y, ffi::fastnn_cuda_mul).with_grad(&[self, other], || {
            MulBackward {
                a: self.detach(),
                b: other.detach(),
            }
        })
    }

    /// `self / other`, broadcasting.
    pub fn div(&self, other: &Tensor) -> Tensor {
        binary_op(self, other, |x, y| x / y, ffi::fastnn_cuda_div).with_grad(&[self, other], || {
            DivBackward {
                a: self.detach(),
                b: other.detach(),
            }
        })
    }

    /// `self + s`. The gradient passes straight through.
    pub fn add_scalar(&self, s: f32) -> Tensor {
        scalar_op(self, s, |x, s| x + s, ffi::fastnn_cuda_add_scalar)
            .with_grad(&[self], || MulScalarBackward { factor: 1.0 })
    }

    /// `self - s`.
    pub fn sub_scalar(&self, s: f32) -> Tensor {
        self.add_scalar(-s)
    }

    /// `self * s`.
    pub fn mul_scalar(&self, s: f32) -> Tensor {
        scalar_op(self, s, |x, s| x * s, ffi::fastnn_cuda_mul_scalar)
            .with_grad(&[self], || MulScalarBackward { factor: s })
    }

    /// `self / s`.
    pub fn div_scalar(&self, s: f32) -> Tensor {
        self.mul_scalar(1.0 / s)
    }
}

// Operator sugar for the common infix forms. Taking references keeps the
// no-copy semantics visible at the call site: `&a + &b`.
macro_rules! impl_binary_operator {
    ($trait:ident, $method:ident, $op:ident) => {
        impl std::ops::$trait for &Tensor {
            type Output = Tensor;
            fn $method(self, rhs: &Tensor) -> Tensor {
                Tensor::$op(self, rhs)
            }
        }
    };
}

impl_binary_operator!(Add, add, add);
impl_binary_operator!(Sub, sub, sub);
impl_binary_operator!(Mul, mul, mul);
impl_binary_operator!(Div, div, div);

impl std::ops::Neg for &Tensor {
    type Output = Tensor;
    fn neg(self) -> Tensor {
        Tensor::neg(self)
    }
}

macro_rules! impl_scalar_operator {
    ($trait:ident, $method:ident, $op:ident) => {
        impl std::ops::$trait<f32> for &Tensor {
            type Output = Tensor;
            fn $method(self, rhs: f32) -> Tensor {
                Tensor::$op(self, rhs)
            }
        }
    };
}

impl_scalar_operator!(Add, add, add_scalar);
impl_scalar_operator!(Sub, sub, sub_scalar);
impl_scalar_operator!(Mul, mul, mul_scalar);
impl_scalar_operator!(Div, div, div_scalar);
