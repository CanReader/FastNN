//! Computation graph for automatic differentiation (reverse-mode / backpropagation).

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::tensor::Tensor;

/// Trait for gradient functions — each operation records one of these.
pub trait GradFn: Send + Sync {
    /// Given the gradient of the output, compute gradients for each input.
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor>;

    /// Return the IDs of input Variables this grad function depends on.
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
    /// Map from variable ID to its accumulated gradient.
    grads: HashMap<u64, Tensor>,
}

impl BackwardGraph {
    pub fn new() -> Self {
        BackwardGraph {
            nodes: Vec::new(),
            grads: HashMap::new(),
        }
    }

    /// Record an operation in the graph.
    pub fn record(&mut self, grad_fn: Arc<dyn GradFn>, output_id: u64) {
        self.nodes.push(GraphNode { grad_fn, output_id });
    }

    /// Run backward pass starting from the given loss variable.
    /// The loss should be a scalar (single element).
    pub fn backward(&mut self, loss_id: u64) {
        // Start with gradient of 1.0 for the loss
        self.grads.insert(loss_id, Tensor::ones(&[1]));

        // Process nodes in reverse order (topological sort — recorded in forward order)
        for node in self.nodes.iter().rev() {
            let grad_output = match self.grads.get(&node.output_id) {
                Some(g) => g.clone(),
                None => continue, // No gradient flows to this node
            };

            let input_grads = node.grad_fn.backward(&grad_output);
            let input_ids = node.grad_fn.inputs();

            assert_eq!(input_grads.len(), input_ids.len(),
                       "Grad function {} returned {} grads but has {} inputs",
                       node.grad_fn.name(), input_grads.len(), input_ids.len());

            for (id, grad) in input_ids.into_iter().zip(input_grads.into_iter()) {
                self.grads.entry(id)
                    .and_modify(|existing| *existing = existing.add(&grad))
                    .or_insert(grad);
            }
        }
    }

    /// Get the gradient for a specific variable.
    pub fn get_grad(&self, var_id: u64) -> Option<&Tensor> {
        self.grads.get(&var_id)
    }

    /// Get all computed gradients.
    pub fn gradients(&self) -> &HashMap<u64, Tensor> {
        &self.grads
    }

    pub fn clear(&mut self) {
        self.nodes.clear();
        self.grads.clear();
    }
}

impl Default for BackwardGraph {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Thread-safe global graph for simple usage
// ============================================================================

static GLOBAL_GRAPH: Mutex<Option<BackwardGraph>> = Mutex::new(None);

/// Enable gradient tracking (creates a new backward graph).
pub fn enable_grad() {
    let mut graph = GLOBAL_GRAPH.lock().unwrap();
    *graph = Some(BackwardGraph::new());
}

/// Disable gradient tracking.
pub fn disable_grad() {
    let mut graph = GLOBAL_GRAPH.lock().unwrap();
    *graph = None;
}

/// Check if gradients are being tracked.
pub fn is_grad_enabled() -> bool {
    GLOBAL_GRAPH.lock().unwrap().is_some()
}

/// Record an operation in the global graph.
pub fn record_op(grad_fn: Arc<dyn GradFn>, output_id: u64) {
    if let Some(graph) = GLOBAL_GRAPH.lock().unwrap().as_mut() {
        graph.record(grad_fn, output_id);
    }
}

/// Run backward on the global graph.
pub fn backward(loss_id: u64) -> HashMap<u64, Tensor> {
    let mut graph_guard = GLOBAL_GRAPH.lock().unwrap();
    if let Some(graph) = graph_guard.as_mut() {
        graph.backward(loss_id);
        let grads = graph.gradients().clone();
        // Clear the graph after backward
        graph.clear();
        grads
    } else {
        HashMap::new()
    }
}

