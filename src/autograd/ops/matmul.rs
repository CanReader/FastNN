//! Derivatives of matmul. Each is two more matmuls, picked so no transposed copy
//! is needed.

use crate::autograd::Backward;
use crate::tensor::Tensor;

use super::reduce_to;

/// `C = A·B` ⟹ `dA = dC·Bᵀ`, `dB = Aᵀ·dC`.
pub struct MatmulBackward {
    pub a: Tensor,
    pub b: Tensor,
}

impl Backward for MatmulBackward {
    fn backward(&self, grad: &Tensor) -> Vec<Tensor> {
        let d_a = grad.matmul_nt(&self.b);
        let d_b = self.a.matmul_tn(grad);
        unbatch(d_a, d_b, &self.a, &self.b)
    }
    fn name(&self) -> &'static str {
        "Matmul"
    }
}

/// `C = A·Bᵀ` ⟹ `dA = dC·B`, `dB = dCᵀ·A`.
pub struct MatmulNtBackward {
    pub a: Tensor,
    pub b: Tensor,
}

impl Backward for MatmulNtBackward {
    fn backward(&self, grad: &Tensor) -> Vec<Tensor> {
        let d_a = grad.matmul(&self.b);
        let d_b = grad.matmul_tn(&self.a);
        unbatch(d_a, d_b, &self.a, &self.b)
    }
    fn name(&self) -> &'static str {
        "MatmulNt"
    }
}

/// `C = Aᵀ·B` ⟹ `dA = B·dCᵀ`, `dB = A·dC`.
pub struct MatmulTnBackward {
    pub a: Tensor,
    pub b: Tensor,
}

impl Backward for MatmulTnBackward {
    fn backward(&self, grad: &Tensor) -> Vec<Tensor> {
        let d_a = self.b.matmul_nt(grad);
        let d_b = self.a.matmul(grad);
        unbatch(d_a, d_b, &self.a, &self.b)
    }
    fn name(&self) -> &'static str {
        "MatmulTn"
    }
}

/// Sum away batch dims the forward broadcast over, e.g. a shared weight matrix.
fn unbatch(d_a: Tensor, d_b: Tensor, a: &Tensor, b: &Tensor) -> Vec<Tensor> {
    vec![reduce_to(&d_a, a.shape()), reduce_to(&d_b, b.shape())]
}
