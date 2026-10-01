//! The reverse pass: walk the graph from a scalar loss back to the leaves.

use std::collections::{HashMap, HashSet};

use crate::tensor::Tensor;

use super::node::{Node, NodeId};

/// Seed `loss` with a gradient of 1 and propagate it to every reachable leaf.
///
/// Gradients land in each parameter's slot, additively — call
/// [`Optimizer::zero_grad`](crate::optim::Optimizer::zero_grad) between steps, or
/// skip it to accumulate across micro-batches.
///
/// Panics if `loss` is not a scalar. Does nothing if `loss` carries no graph
/// (a constant, or built inside [`no_grad`](super::no_grad)).
pub fn backward(loss: &Tensor) {
    assert_eq!(
        loss.numel(),
        1,
        "backward() needs a scalar loss, got shape {:?}",
        loss.shape()
    );
    let Some(root) = loss.grad_fn() else { return };

    // Seed on the loss's own device: every rule derives its output's device from
    // the incoming gradient, so seeding on the wrong one poisons the whole pass.
    let mut grads: HashMap<NodeId, Tensor> = HashMap::new();
    grads.insert(root.id(), Tensor::ones(&[1]).to(loss.device()));

    for node in reverse_topological(root) {
        let Some(grad) = grads.remove(&node.id()) else {
            continue;
        };

        let op = match &node {
            Node::Leaf(slot) => {
                slot.accumulate(grad);
                continue;
            }
            Node::Op(op) => op,
        };

        // Rules compute with ordinary tensor ops. Suppressing tracking here means
        // a rule that forgets to detach a saved value still cannot grow a graph.
        let input_grads = super::no_grad(|| op.rule.backward(&grad));

        if super::anomaly::is_detecting() {
            super::anomaly::check(op.rule.name(), &input_grads);
        }
        assert_eq!(
            input_grads.len(),
            op.inputs.len(),
            "{} returned {} gradients for {} inputs",
            op.rule.name(),
            input_grads.len(),
            op.inputs.len()
        );

        for (input, input_grad) in op.inputs.iter().zip(input_grads) {
            let Some(input_node) = input.grad_fn() else {
                continue;
            };
            grads
                .entry(input_node.id())
                .and_modify(|acc| *acc = acc.add(&input_grad))
                .or_insert(input_grad);
        }
    }
}

/// Nodes ordered so that every consumer comes before the values it consumed.
///
/// Depth-first post-order reversed. The traversal is iterative because a
/// transformer's graph is deeper than a comfortable recursion limit.
fn reverse_topological(root: &Node) -> Vec<Node> {
    let mut order = Vec::new();
    let mut visited = HashSet::new();
    let mut stack = vec![(root.clone(), false)];

    while let Some((node, children_done)) = stack.pop() {
        if children_done {
            order.push(node);
            continue;
        }
        if !visited.insert(node.id()) {
            continue;
        }
        stack.push((node.clone(), true));
        if let Node::Op(op) = &node {
            for input in &op.inputs {
                if let Some(child) = input.grad_fn() {
                    if !visited.contains(&child.id()) {
                        stack.push((child.clone(), false));
                    }
                }
            }
        }
    }

    order.reverse();
    order
}
