//! Recurrent layers.
//!
//! Both cells are written with ordinary differentiable tensor ops, so they train
//! through backpropagation-through-time with no gradient code of their own. The
//! cost is one graph node per gate per timestep: correct, and fine for short
//! sequences, but a transformer will be far faster on long ones.
//!
//! Each layer is single-layer and unidirectional. Stack them in a
//! [`Sequential`](super::Sequential) — the `[batch, sequence, hidden]` output
//! feeds straight into the next.

use crate::tensor::Tensor;

use super::module::Module;
use super::param::Param;

/// The four weight tensors every gated recurrent cell has.
struct Gates {
    input_weight: Param,
    hidden_weight: Param,
    input_bias: Param,
    hidden_bias: Param,
}

impl Gates {
    /// `count` gates of `hidden` units each, over an `input`-wide input.
    fn new(input: usize, hidden: usize, count: usize) -> Gates {
        let width = count * hidden;
        Gates {
            input_weight: Param::new(Tensor::xavier_uniform(&[width, input], input, hidden)),
            hidden_weight: Param::new(Tensor::xavier_uniform(&[width, hidden], hidden, hidden)),
            input_bias: Param::new(Tensor::zeros(&[width])),
            hidden_bias: Param::new(Tensor::zeros(&[width])),
        }
    }

    /// `x · W_ihᵀ + b_ih` and `h · W_hhᵀ + b_hh`, each `[batch, count·hidden]`.
    fn project(&self, x: &Tensor, h: &Tensor) -> (Tensor, Tensor) {
        let from_input = x
            .matmul_nt(&self.input_weight.tensor())
            .add(&self.input_bias.tensor());
        let from_hidden = h
            .matmul_nt(&self.hidden_weight.tensor())
            .add(&self.hidden_bias.tensor());
        (from_input, from_hidden)
    }

    fn named_parameters(&self) -> Vec<(String, Param)> {
        vec![
            ("input_weight".into(), self.input_weight.clone()),
            ("hidden_weight".into(), self.hidden_weight.clone()),
            ("input_bias".into(), self.input_bias.clone()),
            ("hidden_bias".into(), self.hidden_bias.clone()),
        ]
    }
}

/// Long short-term memory.
///
/// Carries a cell state alongside the hidden state; the forget gate multiplies
/// it rather than replacing it, which is what keeps gradients alive over long
/// spans.
pub struct LSTM {
    gates: Gates,
    input_size: usize,
    hidden_size: usize,
}

impl LSTM {
    pub fn new(input_size: usize, hidden_size: usize) -> LSTM {
        let gates = Gates::new(input_size, hidden_size, 4);
        // Start the forget gate open. At zero bias the sigmoid sits at 0.5 and
        // halves the cell state every step, so early gradients vanish before the
        // model has learned what to keep.
        let mut bias = gates.input_bias.value().to_vec();
        bias[hidden_size..2 * hidden_size].fill(1.0);
        gates
            .input_bias
            .set_value(Tensor::from_vec(bias, &[4 * hidden_size]));

        LSTM {
            gates,
            input_size,
            hidden_size,
        }
    }

    /// Run over `[batch, sequence, input_size]`.
    ///
    /// Returns the per-step outputs `[batch, sequence, hidden_size]` plus the
    /// final hidden and cell states.
    pub fn run(
        &self,
        input: &Tensor,
        initial: Option<(&Tensor, &Tensor)>,
    ) -> (Tensor, Tensor, Tensor) {
        let (batch, steps) = check_sequence(input, self.input_size, "LSTM");
        let zeros = || Tensor::zeros(&[batch, self.hidden_size]).to(input.device());
        let (mut h, mut c) = match initial {
            Some((h0, c0)) => (h0.clone(), c0.clone()),
            None => (zeros(), zeros()),
        };

        let mut outputs = Vec::with_capacity(steps);
        for t in 0..steps {
            let x = step_input(input, t, batch, self.input_size);
            let (from_input, from_hidden) = self.gates.project(&x, &h);
            let gates = from_input.add(&from_hidden);

            let slice = |index: usize| gates.narrow(1, index * self.hidden_size, self.hidden_size);
            let input_gate = slice(0).sigmoid();
            let forget_gate = slice(1).sigmoid();
            let candidate = slice(2).tanh();
            let output_gate = slice(3).sigmoid();

            c = forget_gate.mul(&c).add(&input_gate.mul(&candidate));
            h = output_gate.mul(&c.tanh());
            outputs.push(h.clone());
        }

        (stack_steps(&outputs), h, c)
    }
}

impl Module for LSTM {
    fn forward(&self, input: &Tensor) -> Tensor {
        self.run(input, None).0
    }

    fn named_parameters(&self) -> Vec<(String, Param)> {
        self.gates.named_parameters()
    }
}

/// Gated recurrent unit — LSTM's gating with no separate cell state.
pub struct GRU {
    gates: Gates,
    input_size: usize,
    hidden_size: usize,
}

impl GRU {
    pub fn new(input_size: usize, hidden_size: usize) -> GRU {
        GRU {
            gates: Gates::new(input_size, hidden_size, 3),
            input_size,
            hidden_size,
        }
    }

    /// Run over `[batch, sequence, input_size]`, returning the outputs and the
    /// final hidden state.
    pub fn run(&self, input: &Tensor, initial: Option<&Tensor>) -> (Tensor, Tensor) {
        let (batch, steps) = check_sequence(input, self.input_size, "GRU");
        let mut h = initial
            .cloned()
            .unwrap_or_else(|| Tensor::zeros(&[batch, self.hidden_size]).to(input.device()));

        let mut outputs = Vec::with_capacity(steps);
        for t in 0..steps {
            let x = step_input(input, t, batch, self.input_size);
            let (from_input, from_hidden) = self.gates.project(&x, &h);

            let part = |source: &Tensor, index: usize| {
                source.narrow(1, index * self.hidden_size, self.hidden_size)
            };
            let reset = part(&from_input, 0).add(&part(&from_hidden, 0)).sigmoid();
            let update = part(&from_input, 1).add(&part(&from_hidden, 1)).sigmoid();
            // The reset gate scales the *hidden* contribution only, letting the
            // candidate ignore history without ignoring the current input.
            let candidate = part(&from_input, 2)
                .add(&reset.mul(&part(&from_hidden, 2)))
                .tanh();

            let keep = update.neg().add_scalar(1.0);
            h = keep.mul(&candidate).add(&update.mul(&h));
            outputs.push(h.clone());
        }

        (stack_steps(&outputs), h)
    }
}

impl Module for GRU {
    fn forward(&self, input: &Tensor) -> Tensor {
        self.run(input, None).0
    }

    fn named_parameters(&self) -> Vec<(String, Param)> {
        self.gates.named_parameters()
    }
}

fn check_sequence(input: &Tensor, input_size: usize, layer: &str) -> (usize, usize) {
    assert_eq!(
        input.ndim(),
        3,
        "{layer} expects [batch, sequence, features], got {:?}",
        input.shape()
    );
    assert_eq!(
        input.dim(2),
        input_size,
        "{layer} expects {input_size} features, got {:?}",
        input.shape()
    );
    (input.dim(0), input.dim(1))
}

/// Timestep `t` as `[batch, features]`.
fn step_input(input: &Tensor, t: usize, batch: usize, features: usize) -> Tensor {
    input
        .narrow(1, t, 1)
        .reshape(&[batch as i64, features as i64])
}

/// Per-step `[batch, hidden]` outputs into `[batch, sequence, hidden]`.
fn stack_steps(steps: &[Tensor]) -> Tensor {
    Tensor::stack(&steps.iter().collect::<Vec<_>>(), 1)
}
