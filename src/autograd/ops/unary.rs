//! Derivatives of element-wise maths.
//!
//! Each rule multiplies the incoming gradient by a local derivative. Where that
//! derivative is a simple function of the saved input, `elementwise` builds it.

use crate::autograd::Backward;
use crate::tensor::Tensor;

/// `g · f(x)`, with `f` evaluated on the host and moved to the gradient's device.
pub(crate) fn elementwise(
    grad: &Tensor,
    input: &Tensor,
    f: impl Fn(f32) -> f32 + Send + Sync,
) -> Tensor {
    let local = Tensor::from_vec(crate::tensor::ops::map(&input.to_vec(), f), input.shape());
    grad.mul(&local.to(grad.device()))
}

/// `d(-x) = -g`.
pub struct NegBackward;

impl Backward for NegBackward {
    fn backward(&self, grad: &Tensor) -> Vec<Tensor> {
        vec![grad.neg()]
    }
    fn name(&self) -> &'static str {
        "Neg"
    }
}

/// `d(eˣ) = g·eˣ` — reuses the forward output.
pub struct ExpBackward {
    pub output: Tensor,
}

impl Backward for ExpBackward {
    fn backward(&self, grad: &Tensor) -> Vec<Tensor> {
        vec![grad.mul(&self.output)]
    }
    fn name(&self) -> &'static str {
        "Exp"
    }
}

/// `d(ln x) = g/x`.
pub struct LogBackward {
    pub input: Tensor,
}

impl Backward for LogBackward {
    fn backward(&self, grad: &Tensor) -> Vec<Tensor> {
        vec![grad.div(&self.input)]
    }
    fn name(&self) -> &'static str {
        "Log"
    }
}

/// `d(√x) = g / (2√x)` — reuses the forward output.
pub struct SqrtBackward {
    pub output: Tensor,
}

impl Backward for SqrtBackward {
    fn backward(&self, grad: &Tensor) -> Vec<Tensor> {
        vec![grad.div(&self.output.mul_scalar(2.0))]
    }
    fn name(&self) -> &'static str {
        "Sqrt"
    }
}

/// `d|x| = g·sign(x)`, taking `sign(0) = 0`.
pub struct AbsBackward {
    pub input: Tensor,
}

impl Backward for AbsBackward {
    fn backward(&self, grad: &Tensor) -> Vec<Tensor> {
        vec![elementwise(grad, &self.input, |x| {
            x.signum() * (x != 0.0) as i32 as f32
        })]
    }
    fn name(&self) -> &'static str {
        "Abs"
    }
}

/// `d(xᵖ) = g·p·xᵖ⁻¹`.
pub struct PowBackward {
    pub input: Tensor,
    pub exponent: f32,
}

impl Backward for PowBackward {
    fn backward(&self, grad: &Tensor) -> Vec<Tensor> {
        let p = self.exponent;
        vec![elementwise(grad, &self.input, move |x| p * x.powf(p - 1.0))]
    }
    fn name(&self) -> &'static str {
        "Pow"
    }
}

/// Clamping passes the gradient only where the input was inside the range;
/// elsewhere the output was constant.
pub struct ClampBackward {
    pub input: Tensor,
    pub lo: f32,
    pub hi: f32,
}

impl Backward for ClampBackward {
    fn backward(&self, grad: &Tensor) -> Vec<Tensor> {
        let (lo, hi) = (self.lo, self.hi);
        vec![elementwise(grad, &self.input, move |x| {
            ((lo..=hi).contains(&x)) as i32 as f32
        })]
    }
    fn name(&self) -> &'static str {
        "Clamp"
    }
}
