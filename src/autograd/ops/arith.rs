//! Derivatives of element-wise arithmetic.

use crate::autograd::Backward;
use crate::tensor::Tensor;

use super::reduce_to;

/// `d(a + b) = (g, g)`, each reduced back to its operand's shape.
pub struct AddBackward {
    pub a: Vec<usize>,
    pub b: Vec<usize>,
}

impl Backward for AddBackward {
    fn backward(&self, grad: &Tensor) -> Vec<Tensor> {
        vec![reduce_to(grad, &self.a), reduce_to(grad, &self.b)]
    }
    fn name(&self) -> &'static str {
        "Add"
    }
}

/// `d(a - b) = (g, -g)`.
pub struct SubBackward {
    pub a: Vec<usize>,
    pub b: Vec<usize>,
}

impl Backward for SubBackward {
    fn backward(&self, grad: &Tensor) -> Vec<Tensor> {
        vec![reduce_to(grad, &self.a), reduce_to(&grad.neg(), &self.b)]
    }
    fn name(&self) -> &'static str {
        "Sub"
    }
}

/// `d(a · b) = (g·b, g·a)`.
pub struct MulBackward {
    pub a: Tensor,
    pub b: Tensor,
}

impl Backward for MulBackward {
    fn backward(&self, grad: &Tensor) -> Vec<Tensor> {
        vec![
            reduce_to(&grad.mul(&self.b), self.a.shape()),
            reduce_to(&grad.mul(&self.a), self.b.shape()),
        ]
    }
    fn name(&self) -> &'static str {
        "Mul"
    }
}

/// `d(a / b) = (g/b, -g·a/b²)`.
pub struct DivBackward {
    pub a: Tensor,
    pub b: Tensor,
}

impl Backward for DivBackward {
    fn backward(&self, grad: &Tensor) -> Vec<Tensor> {
        let d_a = grad.div(&self.b);
        let d_b = grad.mul(&self.a).div(&self.b.square()).neg();
        vec![
            reduce_to(&d_a, self.a.shape()),
            reduce_to(&d_b, self.b.shape()),
        ]
    }
    fn name(&self) -> &'static str {
        "Div"
    }
}

/// `d(a · k) = g·k`. With `factor = 1` this is also the rule for `+ k`.
pub struct MulScalarBackward {
    pub factor: f32,
}

impl Backward for MulScalarBackward {
    fn backward(&self, grad: &Tensor) -> Vec<Tensor> {
        vec![if self.factor == 1.0 {
            grad.clone()
        } else {
            grad.mul_scalar(self.factor)
        }]
    }
    fn name(&self) -> &'static str {
        "MulScalar"
    }
}
