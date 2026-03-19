//! Variable — a tensor wrapper that participates in autograd.
//!
//! Variables track operations and build a computation graph for backpropagation.

use std::sync::Arc;

use crate::tensor::Tensor;
use crate::autograd::graph::{self, GradFn};

/// A differentiable tensor that records operations for automatic differentiation.
#[derive(Clone)]
pub struct Variable {
    pub tensor: Tensor,
}

impl Variable {
    pub fn new(tensor: Tensor) -> Self {
        Variable { tensor }
    }

    /// Create a variable that requires gradient computation.
    pub fn requires_grad(mut self) -> Self {
        self.tensor.set_requires_grad(true);
        self
    }

    pub fn id(&self) -> u64 {
        self.tensor.id()
    }

    pub fn shape(&self) -> &[usize] {
        self.tensor.shape()
    }

    pub fn data(&self) -> &Tensor {
        &self.tensor
    }

    pub fn grad(&self) -> Option<Tensor> {
        self.tensor.grad()
    }

    /// Run backpropagation from this variable (should be a scalar loss).
    pub fn backward(&self) -> std::collections::HashMap<u64, Tensor> {
        let grads = graph::backward(self.id());

        // Also store gradients in the variable's tensor
        if let Some(g) = grads.get(&self.id()) {
            self.tensor.set_grad(g.clone());
        }

        grads
    }

    // ========================================================================
    // Differentiable operations
    // ========================================================================

    pub fn add(&self, other: &Variable) -> Variable {
        let result = self.tensor.add(&other.tensor);
        let out = Variable::new(result);

        if self.tensor.requires_grad() || other.tensor.requires_grad() {
            let grad_fn = Arc::new(AddBackward {
                input_ids: vec![self.id(), other.id()],
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
            let grad_fn = Arc::new(AddBackward {
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
        Variable::new(result)
    }

    pub fn detach(&self) -> Variable {
        Variable::new(self.tensor.detach())
    }
}

// ============================================================================
// Backward implementations
// ============================================================================

struct AddBackward {
    input_ids: Vec<u64>,
}
impl GradFn for AddBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        if self.input_ids.len() == 2 {
            vec![grad_output.clone(), grad_output.clone()]
        } else {
            vec![grad_output.clone()]
        }
    }
    fn inputs(&self) -> Vec<u64> { self.input_ids.clone() }
    fn name(&self) -> &str { "AddBackward" }
}

struct SubBackward {
    input_ids: Vec<u64>,
}
impl GradFn for SubBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        vec![grad_output.clone(), grad_output.neg()]
    }
    fn inputs(&self) -> Vec<u64> { self.input_ids.clone() }
    fn name(&self) -> &str { "SubBackward" }
}

struct MulBackward {
    input_ids: Vec<u64>,
    a: Tensor,
    b: Tensor,
}
impl GradFn for MulBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        vec![
            grad_output.mul(&self.b),
            grad_output.mul(&self.a),
        ]
    }
    fn inputs(&self) -> Vec<u64> { self.input_ids.clone() }
    fn name(&self) -> &str { "MulBackward" }
}

struct MatmulBackward {
    input_ids: Vec<u64>,
    a: Tensor,
    b: Tensor,
}
impl GradFn for MatmulBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        // For C = A @ B:
        // dA = dC @ B^T
        // dB = A^T @ dC
        let grad_a = grad_output.matmul(&self.b.transpose());
        let grad_b = self.a.transpose().matmul(grad_output);
        vec![grad_a, grad_b]
    }
    fn inputs(&self) -> Vec<u64> { self.input_ids.clone() }
    fn name(&self) -> &str { "MatmulBackward" }
}

struct ReluBackward {
    input_ids: Vec<u64>,
    input: Tensor,
}
impl GradFn for ReluBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        let mask_data: Vec<f32> = self.input.to_vec().iter().map(|&x| if x > 0.0 { 1.0 } else { 0.0 }).collect();
        let mask = Tensor::from_vec(mask_data, self.input.shape());
        vec![grad_output.mul(&mask)]
    }
    fn inputs(&self) -> Vec<u64> { self.input_ids.clone() }
    fn name(&self) -> &str { "ReluBackward" }
}

struct SigmoidBackward {
    input_ids: Vec<u64>,
    output: Tensor,
}
impl GradFn for SigmoidBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        // dsigmoid = sigmoid * (1 - sigmoid) * grad_output
        let ones = Tensor::ones(self.output.shape());
        let grad = grad_output.mul(&self.output.mul(&ones.sub(&self.output)));
        vec![grad]
    }
    fn inputs(&self) -> Vec<u64> { self.input_ids.clone() }
    fn name(&self) -> &str { "SigmoidBackward" }
}

struct TanhBackward {
    input_ids: Vec<u64>,
    output: Tensor,
}
impl GradFn for TanhBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        // dtanh = (1 - tanh^2) * grad_output
        let ones = Tensor::ones(self.output.shape());
        let grad = grad_output.mul(&ones.sub(&self.output.mul(&self.output)));
        vec![grad]
    }
    fn inputs(&self) -> Vec<u64> { self.input_ids.clone() }
    fn name(&self) -> &str { "TanhBackward" }
}

struct GeluBackward {
    input_ids: Vec<u64>,
    input: Tensor,
}
impl GradFn for GeluBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        let data = self.input.to_vec();
        let grad_data: Vec<f32> = data.iter().map(|&x| {
            let cdf = 0.5 * (1.0 + libm::erff(x * std::f32::consts::FRAC_1_SQRT_2));
            let pdf = (-0.5 * x * x).exp() * (1.0 / (2.0 * std::f32::consts::PI).sqrt());
            cdf + x * pdf
        }).collect();
        let local_grad = Tensor::from_vec(grad_data, self.input.shape());
        vec![grad_output.mul(&local_grad)]
    }
    fn inputs(&self) -> Vec<u64> { self.input_ids.clone() }
    fn name(&self) -> &str { "GeluBackward" }
}

struct SoftmaxBackward {
    input_ids: Vec<u64>,
    output: Tensor,
}
impl GradFn for SoftmaxBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        // Jacobian-vector product for softmax
        let s = &self.output;
        let dot = grad_output.mul(s).sum();
        let dot_val = dot.item();
        let dot_expanded = Tensor::full(s.shape(), dot_val);
        let grad = s.mul(&grad_output.sub(&dot_expanded));
        vec![grad]
    }
    fn inputs(&self) -> Vec<u64> { self.input_ids.clone() }
    fn name(&self) -> &str { "SoftmaxBackward" }
}

struct LogSoftmaxBackward {
    input_ids: Vec<u64>,
    output: Tensor,
}
impl GradFn for LogSoftmaxBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        let softmax = self.output.exp();
        let sum = grad_output.sum();
        let sum_val = sum.item();
        let grad = grad_output.sub(&softmax.mul_scalar(sum_val));
        vec![grad]
    }
    fn inputs(&self) -> Vec<u64> { self.input_ids.clone() }
    fn name(&self) -> &str { "LogSoftmaxBackward" }
}

struct SumBackward {
    input_ids: Vec<u64>,
    input_shape: Vec<usize>,
}
impl GradFn for SumBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        let val = grad_output.item();
        vec![Tensor::full(&self.input_shape, val)]
    }
    fn inputs(&self) -> Vec<u64> { self.input_ids.clone() }
    fn name(&self) -> &str { "SumBackward" }
}

struct MeanBackward {
    input_ids: Vec<u64>,
    input_numel: usize,
    input_shape: Vec<usize>,
}
impl GradFn for MeanBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        let val = grad_output.item() / self.input_numel as f32;
        vec![Tensor::full(&self.input_shape, val)]
    }
    fn inputs(&self) -> Vec<u64> { self.input_ids.clone() }
    fn name(&self) -> &str { "MeanBackward" }
}

struct MulScalarBackward {
    input_ids: Vec<u64>,
    scalar: f32,
}
impl GradFn for MulScalarBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        vec![grad_output.mul_scalar(self.scalar)]
    }
    fn inputs(&self) -> Vec<u64> { self.input_ids.clone() }
    fn name(&self) -> &str { "MulScalarBackward" }
}

struct ReshapeBackward {
    input_ids: Vec<u64>,
    input_shape: Vec<usize>,
}
impl GradFn for ReshapeBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        let shape: Vec<i64> = self.input_shape.iter().map(|&s| s as i64).collect();
        vec![grad_output.reshape(&shape)]
    }
    fn inputs(&self) -> Vec<u64> { self.input_ids.clone() }
    fn name(&self) -> &str { "ReshapeBackward" }
}
