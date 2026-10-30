//! Reverse-mode autodiff. Each differentiable tensor carries a [`Node`] with its
//! backward rule and inputs, so the graph is just the tensors and there is no tape.
//!
//! Weights live in [`Param`](crate::nn::Param) handles. Call `backward()` on a
//! scalar and each handle's gradient slot fills in.
//!
//! ```
//! use fastnn::prelude::*;
//!
//! let w = Param::new(Tensor::from_vec(vec![1.0, 2.0], &[2]));
//! let loss = w.tensor().mul_scalar(3.0).sum();
//! loss.backward();
//! assert_eq!(w.grad().unwrap().to_vec(), vec![3.0, 3.0]);
//! ```
//!
//! - [`no_grad`] runs a closure without building a graph. Use it for inference.
//! - [`detect_anomaly`] panics at the first rule that produces a NaN or inf, naming it.
//! - [`ops`] holds one backward rule per forward op, in files named like `tensor/ops`.
//!
//! A new op is a forward in `tensor/ops`, a rule in [`ops`] that returns gradients
//! in input order and saves only detached tensors, and a case in
//! `tests/gradcheck.rs`, which checks every rule against central finite differences
//! of its own forward.

pub mod anomaly;
pub mod engine;
pub mod mode;
pub mod node;
pub mod ops;

pub use anomaly::{detect_anomaly, is_detecting};
pub use engine::backward;
pub use mode::{is_enabled, no_grad};
pub use node::{Backward, GradSlot, Node, Op};
