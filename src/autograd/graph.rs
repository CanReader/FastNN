//! Computation graph for automatic differentiation (reverse-mode / backpropagation).
//!
//! The graph is a tape: each forward op appends a `GraphNode`. When `backward()` is
//! called, we walk the tape in reverse, computing per-input gradients via each node's
//! `GradFn`. For tensors whose `.grad` Arc we've captured, we write the final
//! accumulated gradient into that Arc so the user can read it via `tensor.grad()`.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::tensor::Tensor;

pub(crate) type GradCell = Arc<Mutex<Option<Box<Tensor>>>>;

/// Trait for gradient functions — each operation records one of these.
pub trait GradFn: Send + Sync {
    /// Given the gradient of the output, compute gradients for each input.
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor>;

    /// Return the IDs of input tensors this grad function depends on.
    fn inputs(&self) -> Vec<u64>;

    /// Name of the operation (for debugging).
    fn name(&self) -> &str;
}

/// A node in the backward graph.
pub struct GraphNode {
    pub grad_fn: Arc<dyn GradFn>,
    pub output_id: u64,
}

/// The backward computation graph — collects operations and runs backprop.
pub struct BackwardGraph {
    nodes: Vec<GraphNode>,
    /// Running intermediate gradients, keyed by tensor id.
    grads: HashMap<u64, Tensor>,
    /// For each input tensor that requires grad, a handle to its `.grad` Arc.
    /// At end of backward, accumulated grads are written into these Arcs so
    /// the user-facing `tensor.grad()` accessor returns them.
    leaf_grad_cells: HashMap<u64, GradCell>,
}

impl BackwardGraph {
    pub fn new() -> Self {
        BackwardGraph {
            nodes: Vec::new(),
            grads: HashMap::new(),
            leaf_grad_cells: HashMap::new(),
        }
    }

    /// Record an operation. `leaf_cells` provides `(tensor_id, grad_cell)` pairs for
    /// each input tensor that `requires_grad` — used to write final grads back.
    pub fn record(
        &mut self,
        grad_fn: Arc<dyn GradFn>,
        output_id: u64,
        leaf_cells: Vec<(u64, GradCell)>,
    ) {
        for (id, cell) in leaf_cells {
            self.leaf_grad_cells.entry(id).or_insert(cell);
        }
        self.nodes.push(GraphNode { grad_fn, output_id });
    }

    /// Run backward pass starting from the given scalar loss.
    ///
    /// `loss_device` is the device the loss tensor lives on; the seed gradient
    /// must match it, otherwise device-preserving grad fns (e.g. `SumBackward`)
    /// propagate a CPU gradient into a GPU graph and ops panic on a device
    /// mismatch.
    pub fn backward(&mut self, loss_id: u64, loss_device: crate::tensor::Device) {
        // Seed with gradient of 1.0 for the loss scalar, on the loss's device.
        self.grads.insert(loss_id, Tensor::ones(&[1]).to_device(loss_device));

        for node in self.nodes.iter().rev() {
            let grad_output = match self.grads.get(&node.output_id) {
                Some(g) => g.clone(),
                None => continue,
            };

            let input_grads = node.grad_fn.backward(&grad_output);
            let input_ids = node.grad_fn.inputs();

            assert_eq!(input_grads.len(), input_ids.len(),
                       "GradFn {} returned {} grads for {} inputs",
                       node.grad_fn.name(), input_grads.len(), input_ids.len());

            for (id, grad) in input_ids.into_iter().zip(input_grads.into_iter()) {
                self.grads.entry(id)
                    .and_modify(|existing| *existing = existing.add(&grad))
                    .or_insert(grad);
            }
        }

        // Write accumulated gradients into each captured leaf tensor's grad cell.
        for (id, cell) in &self.leaf_grad_cells {
            if let Some(g) = self.grads.get(id) {
                let mut guard = cell.lock().unwrap();
                match guard.as_mut() {
                    Some(existing) => {
                        let summed = existing.add(g);
                        **existing = summed;
                    }
                    None => *guard = Some(Box::new(g.clone())),
                }
            }
        }
    }

    /// Get the gradient for a specific tensor (by id).
    pub fn get_grad(&self, tensor_id: u64) -> Option<&Tensor> {
        self.grads.get(&tensor_id)
    }

    /// Get all computed gradients.
    pub fn gradients(&self) -> &HashMap<u64, Tensor> {
        &self.grads
    }

    pub fn clear(&mut self) {
        self.nodes.clear();
        self.grads.clear();
        self.leaf_grad_cells.clear();
    }
}

impl Default for BackwardGraph {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Thread-local global graph (one tape per thread, active during forward passes)
// ============================================================================

thread_local! {
    static GLOBAL_GRAPH: std::cell::RefCell<Option<BackwardGraph>> =
        const { std::cell::RefCell::new(None) };
}

/// Enable gradient tracking on the current thread (creates a new tape).
pub fn enable_grad() {
    GLOBAL_GRAPH.with(|g| *g.borrow_mut() = Some(BackwardGraph::new()));
}

/// Disable gradient tracking on the current thread.
pub fn disable_grad() {
    GLOBAL_GRAPH.with(|g| *g.borrow_mut() = None);
}

/// Whether gradient tracking is active on this thread.
pub fn is_grad_enabled() -> bool {
    GLOBAL_GRAPH.with(|g| g.borrow().is_some())
}

/// Record an operation in the current thread's graph with grad cells for leaf writeback.
pub(crate) fn record_op_with_cells(
    grad_fn: Arc<dyn GradFn>,
    output_id: u64,
    leaf_cells: Vec<(u64, GradCell)>,
) {
    GLOBAL_GRAPH.with(|g| {
        if let Some(graph) = g.borrow_mut().as_mut() {
            graph.record(grad_fn, output_id, leaf_cells);
        }
    });
}

/// Legacy record_op — 2-arg form used by the `Variable` API. No leaf writeback.
pub fn record_op(grad_fn: Arc<dyn GradFn>, output_id: u64) {
    record_op_with_cells(grad_fn, output_id, Vec::new());
}

/// Run backward on the current thread's graph. Returns the final gradients map.
///
/// The graph is temporarily taken out of its RefCell before running so that
/// tensor ops invoked by `GradFn::backward` implementations don't re-enter the
/// cell (they'll see `is_grad_enabled() == false` during the backward pass,
/// which is what we want — we don't record the backward computation itself).
pub fn backward(loss_id: u64, loss_device: crate::tensor::Device) -> HashMap<u64, Tensor> {
    let mut graph = match GLOBAL_GRAPH.with(|g| g.borrow_mut().take()) {
        Some(g) => g,
        None => return HashMap::new(),
    };
    graph.backward(loss_id, loss_device);
    let grads = graph.gradients().clone();
    // Don't restore — the tape is consumed. Caller must `enable_grad()` again
    // before the next training forward pass.
    grads
}

/// Clear the current thread's tape without running backward.
pub fn clear() {
    GLOBAL_GRAPH.with(|g| {
        if let Some(graph) = g.borrow_mut().as_mut() {
            graph.clear();
        }
    });
}
