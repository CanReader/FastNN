//! The reverse pass from a scalar loss back to the leaves.

use std::collections::{HashMap, HashSet};

use crate::tensor::Tensor;

use super::node::{Node, NodeId};

/// Propagate a gradient of 1 from `loss` to every reachable leaf.
///
/// Gradients add into the parameter slots, so call
/// [`Optimizer::zero_grad`](crate::optim::Optimizer::zero_grad) between steps.
/// Panics if `loss` is not a scalar, does nothing if it has no graph.
pub fn backward(loss: &Tensor) {
    assert_eq!(
        loss.numel(),
        1,
        "backward() needs a scalar loss, got shape {:?}",
        loss.shape()
    );
    let Some(root) = loss.grad_fn() else { return };

    // rules take their device from the incoming gradient, so seed on the loss's
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

        // no_grad so a rule that forgets to detach a saved value cannot grow a graph
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

/// Nodes with every consumer before the values it consumed. Iterative, since
/// deep graphs would overflow the stack.
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
