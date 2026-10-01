//! The dataset trait and simple in-memory implementations.

use crate::tensor::{shape, Tensor};

/// A finite, indexable collection of `(input, target)` pairs.
///
/// Items are written into caller-provided slices rather than returned as
/// tensors. A [`DataLoader`](super::DataLoader) allocates one buffer per batch
/// and fills it, so batching a dataset of 60,000 images does not build 60,000
/// throwaway tensors.
pub trait Dataset: Send + Sync {
    /// Number of items.
    fn len(&self) -> usize;

    /// Shape of one input, without a batch dimension.
    fn input_shape(&self) -> Vec<usize>;

    /// Shape of one target, without a batch dimension. `[1]` for a class label.
    fn target_shape(&self) -> Vec<usize>;

    /// Copy item `index` into the slices, which are exactly one item wide.
    fn write(&self, index: usize, input: &mut [f32], target: &mut [f32]);

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Elements in one input.
    fn input_size(&self) -> usize {
        shape::numel(&self.input_shape())
    }

    /// Elements in one target.
    fn target_size(&self) -> usize {
        shape::numel(&self.target_shape())
    }
}

/// A dataset held entirely in two tensors, batched along dimension 0.
pub struct TensorDataset {
    inputs: Vec<f32>,
    targets: Vec<f32>,
    input_shape: Vec<usize>,
    target_shape: Vec<usize>,
    len: usize,
}

impl TensorDataset {
    /// Both tensors must agree on their first dimension.
    pub fn new(inputs: &Tensor, targets: &Tensor) -> TensorDataset {
        assert_eq!(
            inputs.dim(0),
            targets.dim(0),
            "TensorDataset: {} inputs but {} targets",
            inputs.dim(0),
            targets.dim(0)
        );
        TensorDataset {
            len: inputs.dim(0),
            input_shape: item_shape(inputs),
            target_shape: item_shape(targets),
            inputs: inputs.to_vec(),
            targets: targets.to_vec(),
        }
    }
}

impl Dataset for TensorDataset {
    fn len(&self) -> usize {
        self.len
    }

    fn input_shape(&self) -> Vec<usize> {
        self.input_shape.clone()
    }

    fn target_shape(&self) -> Vec<usize> {
        self.target_shape.clone()
    }

    fn write(&self, index: usize, input: &mut [f32], target: &mut [f32]) {
        copy_item(&self.inputs, index, input);
        copy_item(&self.targets, index, target);
    }
}

/// Everything after the batch dimension, or `[1]` for a flat tensor of labels.
fn item_shape(t: &Tensor) -> Vec<usize> {
    if t.ndim() <= 1 {
        vec![1]
    } else {
        t.shape()[1..].to_vec()
    }
}

fn copy_item(source: &[f32], index: usize, out: &mut [f32]) {
    let start = index * out.len();
    out.copy_from_slice(&source[start..start + out.len()]);
}
