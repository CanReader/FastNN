//! Variable — a thin tensor wrapper that participates in autograd.
//!
//! **Status:** legacy. Prefer using `Tensor` directly — tensors now record their
//! own backward graph when `requires_grad = true`. This wrapper is kept for
//! API compatibility and for call sites that find the explicit style useful.

use std::sync::Arc;

use crate::tensor::Tensor;
use crate::autograd::graph;
use crate::autograd::backward_ops::*;

/// A differentiable tensor. Operations on a `Variable` record backward nodes
/// in the global graph; call `.backward()` on a scalar loss to compute grads.
#[derive(Clone)]
pub struct Variable {
    pub tensor: Tensor,
}

impl Variable {
    pub fn new(tensor: Tensor) -> Self {
        Variable { tensor }
    }

    pub fn requires_grad(mut self) -> Self {
        self.tensor.set_requires_grad(true);
        self
    }

    pub fn id(&self) -> u64 { self.tensor.id() }
    pub fn shape(&self) -> &[usize] { self.tensor.shape() }
    pub fn data(&self) -> &Tensor { &self.tensor }
    pub fn grad(&self) -> Option<Tensor> { self.tensor.grad() }

    /// Run backward from this variable (expected to be a scalar loss).
    pub fn backward(&self) -> std::collections::HashMap<u64, Tensor> {
        let grads = graph::backward(self.id(), self.tensor.device());
        if let Some(g) = grads.get(&self.id()) {
            self.tensor.set_grad(g.clone());
        }
        grads
    }

    // ========================================================================
    // Differentiable operations (delegate to Tensor ops, which now auto-record)
    // ========================================================================

    pub fn add(&self, other: &Variable) -> Variable {
        let result = self.tensor.add(&other.tensor);
        let out = Variable::new(result);
        if self.tensor.requires_grad() || other.tensor.requires_grad() {
            let grad_fn = Arc::new(AddBackward {
                input_ids: vec![self.id(), other.id()],
                a_shape: self.tensor.shape().to_vec(),
                b_shape: other.tensor.shape().to_vec(),
            });
            graph::record_op(grad_fn, out.id());
        }
        out
    }

    pub fn sub(&self, other: &Variable) -> Variable {
        let result = self.tensor.sub(&other.tensor);
        let out = Variable::new(result);
        if self.tensor.requires_grad() || other.tensor.requires_grad() {
            let grad_fn = Arc::new(SubBackward {
                input_ids: vec![self.id(), other.id()],
                a_shape: self.tensor.shape().to_vec(),
                b_shape: other.tensor.shape().to_vec(),
            });
            graph::record_op(grad_fn, out.id());
        }
        out
    }

    pub fn mul(&self, other: &Variable) -> Variable {
        let result = self.tensor.mul(&other.tensor);
        let out = Variable::new(result);
        if self.tensor.requires_grad() || other.tensor.requires_grad() {
            let grad_fn = Arc::new(MulBackward {
                input_ids: vec![self.id(), other.id()],
                a: self.tensor.clone(),
                b: other.tensor.clone(),
            });
            graph::record_op(grad_fn, out.id());
        }
        out
    }

    pub fn matmul(&self, other: &Variable) -> Variable {
        let result = self.tensor.matmul(&other.tensor);
        let out = Variable::new(result);
        if self.tensor.requires_grad() || other.tensor.requires_grad() {
            let grad_fn = Arc::new(MatmulBackward {
                input_ids: vec![self.id(), other.id()],
                a: self.tensor.clone(),
                b: other.tensor.clone(),
            });
            graph::record_op(grad_fn, out.id());
        }
        out
    }

    pub fn relu(&self) -> Variable {
        let result = self.tensor.relu();
        let out = Variable::new(result);
        if self.tensor.requires_grad() {
            let grad_fn = Arc::new(ReluBackward {
                input_ids: vec![self.id()],
                input: self.tensor.clone(),
            });
            graph::record_op(grad_fn, out.id());
        }
        out
    }

    pub fn sigmoid(&self) -> Variable {
        let result = self.tensor.sigmoid();
        let out = Variable::new(result.clone());
        if self.tensor.requires_grad() {
            let grad_fn = Arc::new(SigmoidBackward {
                input_ids: vec![self.id()],
                output: result,
            });
            graph::record_op(grad_fn, out.id());
        }
        out
    }

    pub fn tanh_act(&self) -> Variable {
        let result = self.tensor.tanh_act();
        let out = Variable::new(result.clone());
        if self.tensor.requires_grad() {
            let grad_fn = Arc::new(TanhBackward {
                input_ids: vec![self.id()],
                output: result,
            });
            graph::record_op(grad_fn, out.id());
        }
        out
    }

    pub fn gelu(&self) -> Variable {
        let result = self.tensor.gelu();
        let out = Variable::new(result);
        if self.tensor.requires_grad() {
            let grad_fn = Arc::new(GeluBackward {
                input_ids: vec![self.id()],
                input: self.tensor.clone(),
            });
            graph::record_op(grad_fn, out.id());
        }
        out
    }

    pub fn softmax(&self) -> Variable {
        let result = self.tensor.softmax();
        let out = Variable::new(result.clone());
        if self.tensor.requires_grad() {
            let grad_fn = Arc::new(SoftmaxBackward {
                input_ids: vec![self.id()],
                output: result,
            });
            graph::record_op(grad_fn, out.id());
        }
        out
    }

    pub fn log_softmax(&self) -> Variable {
        let result = self.tensor.log_softmax();
        let out = Variable::new(result.clone());
        if self.tensor.requires_grad() {
            let grad_fn = Arc::new(LogSoftmaxBackward {
                input_ids: vec![self.id()],
                output: result,
            });
            graph::record_op(grad_fn, out.id());
        }
        out
    }

    pub fn sum(&self) -> Variable {
        let result = self.tensor.sum();
        let out = Variable::new(result);
        if self.tensor.requires_grad() {
            let grad_fn = Arc::new(SumBackward {
                input_ids: vec![self.id()],
                input_shape: self.tensor.shape().to_vec(),
            });
            graph::record_op(grad_fn, out.id());
        }
        out
    }

    pub fn mean(&self) -> Variable {
        let result = self.tensor.mean();
        let out = Variable::new(result);
        if self.tensor.requires_grad() {
            let grad_fn = Arc::new(MeanBackward {
                input_ids: vec![self.id()],
                input_numel: self.tensor.numel(),
                input_shape: self.tensor.shape().to_vec(),
            });
            graph::record_op(grad_fn, out.id());
        }
        out
    }

    pub fn mul_scalar(&self, scalar: f32) -> Variable {
        let result = self.tensor.mul_scalar(scalar);
        let out = Variable::new(result);
        if self.tensor.requires_grad() {
            let grad_fn = Arc::new(MulScalarBackward {
                input_ids: vec![self.id()],
                scalar,
            });
            graph::record_op(grad_fn, out.id());
        }
        out
    }

    pub fn add_scalar(&self, scalar: f32) -> Variable {
        let result = self.tensor.add_scalar(scalar);
        let out = Variable::new(result);
        if self.tensor.requires_grad() {
            let grad_fn = Arc::new(AddScalarBackward {
                input_ids: vec![self.id()],
            });
            graph::record_op(grad_fn, out.id());
        }
        out
    }

    pub fn reshape(&self, shape: &[i64]) -> Variable {
        let result = self.tensor.reshape(shape);
        let out = Variable::new(result);
        if self.tensor.requires_grad() {
            let grad_fn = Arc::new(ReshapeBackward {
                input_ids: vec![self.id()],
                input_shape: self.tensor.shape().to_vec(),
            });
            graph::record_op(grad_fn, out.id());
        }
        out
    }

    pub fn transpose(&self) -> Variable {
        let result = self.tensor.transpose();
        let out = Variable::new(result);
        if self.tensor.requires_grad() {
            let grad_fn = Arc::new(TransposeBackward {
                input_ids: vec![self.id()],
            });
            graph::record_op(grad_fn, out.id());
        }
        out
    }

    pub fn detach(&self) -> Variable {
        Variable::new(self.tensor.detach())
    }
}
