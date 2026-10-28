//! Reverse-mode autodiff. Each differentiable tensor carries a [`Node`] with its
//! backward rule and inputs, so the graph is just the tensors and there is no tape.

pub mod anomaly;
pub mod engine;
pub mod mode;
pub mod node;
pub mod ops;

pub use anomaly::{detect_anomaly, is_detecting};
pub use engine::backward;
pub use mode::{is_enabled, no_grad};
pub use node::{Backward, GradSlot, Node, Op};
