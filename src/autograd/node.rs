//! Graph nodes: what a tensor remembers about how it was produced.

use std::sync::{Arc, Mutex};

use crate::tensor::Tensor;

/// How one operation turns the gradient of its output into gradients of its inputs.
///
/// A rule owns whatever the forward pass needs to save (operands, masks, pooling
/// indices). Save *detached* tensors — a saved value that still carries its own
/// node would pin the graph that produced it.
pub trait Backward: Send + Sync {
    /// Gradients for each input, in the same order as [`Op::inputs`].
    fn backward(&self, grad: &Tensor) -> Vec<Tensor>;

    /// Shown in panics and graph dumps.
    fn name(&self) -> &'static str;
}

/// An operation that produced a tensor, plus the tensors it consumed.
pub struct Op {
    pub rule: Box<dyn Backward>,
    pub inputs: Vec<Tensor>,
}

/// A tensor's link into the graph. `None` on a tensor means "constant data" —
/// gradient flow stops there.
#[derive(Clone)]
pub enum Node {
    /// A learnable value. Gradient reaching it lands in the shared slot, which
    /// is the same slot the optimizer reads.
    Leaf(GradSlot),
    /// An intermediate value. Gradient reaching it is pushed further back.
    Op(Arc<Op>),
}

impl Node {
    pub fn op(rule: impl Backward + 'static, inputs: Vec<Tensor>) -> Node {
        Node::Op(Arc::new(Op {
            rule: Box::new(rule),
            inputs,
        }))
    }

    /// Stable identity, used to accumulate gradients per node during backward.
    ///
    /// Two handles to the same parameter share a `GradSlot` and therefore an id,
    /// which is exactly what makes weight tying work.
    pub fn id(&self) -> NodeId {
        match self {
            Node::Leaf(slot) => NodeId(Arc::as_ptr(&slot.0) as *const () as usize),
            Node::Op(op) => NodeId(Arc::as_ptr(op) as *const () as usize),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct NodeId(usize);

/// The gradient of one learnable tensor, shared between the model and the optimizer.
#[derive(Clone, Default)]
pub struct GradSlot(Arc<Mutex<Option<Tensor>>>);

impl GradSlot {
    pub fn new() -> Self {
        GradSlot::default()
    }

    /// Add `grad` to whatever is already there.
    ///
    /// Accumulating rather than overwriting is what lets a parameter be used
    /// twice in one forward pass, and what makes gradient accumulation across
    /// micro-batches work without extra bookkeeping.
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
