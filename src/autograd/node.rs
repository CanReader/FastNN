//! Graph nodes: what a tensor remembers about how it was produced.

use std::sync::{Arc, Mutex};

use crate::tensor::Tensor;

/// Turns the gradient of an op's output into gradients of its inputs.
///
/// Save detached tensors in a rule, or the saved value pins the graph behind it.
pub trait Backward: Send + Sync {
    /// One gradient per input, in the order of [`Op::inputs`].
    fn backward(&self, grad: &Tensor) -> Vec<Tensor>;

    /// Used in panic messages.
    fn name(&self) -> &'static str;
}

/// The rule that produced a tensor and the tensors it consumed.
pub struct Op {
    pub rule: Box<dyn Backward>,
    pub inputs: Vec<Tensor>,
}

/// A tensor's link into the graph. Tensors without one are constants.
#[derive(Clone)]
pub enum Node {
    /// A parameter. Its gradient lands in the slot the optimizer reads.
    Leaf(GradSlot),
    /// An intermediate value.
    Op(Arc<Op>),
}

impl Node {
    pub fn op(rule: impl Backward + 'static, inputs: Vec<Tensor>) -> Node {
        Node::Op(Arc::new(Op {
            rule: Box::new(rule),
            inputs,
        }))
    }

    /// Identity for accumulating gradients. Handles to the same parameter share
    /// a slot and so an id, which is what makes weight tying work.
    pub fn id(&self) -> NodeId {
        match self {
            Node::Leaf(slot) => NodeId(Arc::as_ptr(&slot.0) as *const () as usize),
            Node::Op(op) => NodeId(Arc::as_ptr(op) as *const () as usize),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct NodeId(usize);

/// A parameter's gradient, shared between the model and the optimizer.
#[derive(Clone, Default)]
pub struct GradSlot(Arc<Mutex<Option<Tensor>>>);

impl GradSlot {
    pub fn new() -> Self {
        GradSlot::default()
    }

    /// Add to the stored gradient. Adding (not overwriting) covers parameters
    /// used twice and accumulation across micro-batches.
    pub fn accumulate(&self, grad: Tensor) {
        let mut slot = self.0.lock().unwrap();
        *slot = Some(match slot.take() {
            Some(existing) => existing.add(&grad),
            None => grad,
        });
    }

    pub fn get(&self) -> Option<Tensor> {
        self.0.lock().unwrap().clone()
    }

    pub fn clear(&self) {
        *self.0.lock().unwrap() = None;
    }
}
