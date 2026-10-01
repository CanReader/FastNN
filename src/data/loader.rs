//! Batching and shuffling.

use crate::rng;
use crate::tensor::{Device, Tensor};

use super::dataset::Dataset;

/// One batch of inputs and targets.
pub struct Batch {
    pub inputs: Tensor,
    pub targets: Tensor,
}

impl Batch {
    /// How many items are in this batch.
    pub fn len(&self) -> usize {
        self.inputs.dim(0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Targets read as class indices — the form [`cross_entropy`](crate::nn::cross_entropy) takes.
    pub fn labels(&self) -> Vec<usize> {
        self.targets.to_vec().iter().map(|&v| v as usize).collect()
    }
}

/// Iterates a [`Dataset`] in batches, optionally shuffled.
///
/// ```no_run
/// use fastnn::prelude::*;
/// # let dataset: fastnn::data::TensorDataset = unimplemented!();
/// # let model = Sequential::new();
/// let loader = DataLoader::new(&dataset, 128).shuffle(true).drop_last(true);
///
/// for batch in loader.iter() {
///     let logits = model.forward(&batch.inputs);
///     let loss = cross_entropy(&logits, &batch.labels());
///     // ...
/// }
/// ```
///
/// A fresh shuffle happens each time [`iter`](DataLoader::iter) is called, so
/// every epoch sees a different order.
pub struct DataLoader<'a> {
    dataset: &'a dyn Dataset,
    batch_size: usize,
    shuffle: bool,
    drop_last: bool,
    device: Device,
}

impl<'a> DataLoader<'a> {
    pub fn new(dataset: &'a dyn Dataset, batch_size: usize) -> DataLoader<'a> {
        assert!(batch_size > 0, "batch_size must be positive");
        DataLoader {
            dataset,
            batch_size,
            shuffle: false,
            drop_last: false,
            device: Device::Cpu,
        }
    }

    /// Reorder items each epoch. On for training, off for evaluation.
    pub fn shuffle(mut self, shuffle: bool) -> DataLoader<'a> {
        self.shuffle = shuffle;
        self
    }

    /// Skip the final short batch. Useful when a layer needs a fixed batch size.
    pub fn drop_last(mut self, drop_last: bool) -> DataLoader<'a> {
        self.drop_last = drop_last;
        self
    }

    /// Upload each batch to `device` as it is produced.
    pub fn to_device(mut self, device: Device) -> DataLoader<'a> {
        self.device = device;
        self
    }

    /// Number of batches one pass will yield.
    pub fn batches(&self) -> usize {
        let n = self.dataset.len();
        if self.drop_last {
            n / self.batch_size
        } else {
            n.div_ceil(self.batch_size)
        }
    }

    /// Start a pass over the data.
    pub fn iter(&self) -> Batches<'a> {
        let mut order: Vec<usize> = (0..self.dataset.len()).collect();
        if self.shuffle {
            rng::shuffle(&mut order);
        }
        Batches {
            dataset: self.dataset,
            order,
            position: 0,
            batch_size: self.batch_size,
            drop_last: self.drop_last,
            device: self.device,
        }
    }
}

impl<'a> IntoIterator for &DataLoader<'a> {
    type Item = Batch;
    type IntoIter = Batches<'a>;

    fn into_iter(self) -> Batches<'a> {
        self.iter()
    }
}

/// One pass over a shuffled index order.
pub struct Batches<'a> {
    dataset: &'a dyn Dataset,
    order: Vec<usize>,
    position: usize,
    batch_size: usize,
    drop_last: bool,
    device: Device,
}

impl Iterator for Batches<'_> {
    type Item = Batch;

    fn next(&mut self) -> Option<Batch> {
        let remaining = self.order.len().saturating_sub(self.position);
        let size = remaining.min(self.batch_size);
        if size == 0 || (size < self.batch_size && self.drop_last) {
            return None;
        }

        let (input_size, target_size) = (self.dataset.input_size(), self.dataset.target_size());
        let mut inputs = vec![0.0f32; size * input_size];
        let mut targets = vec![0.0f32; size * target_size];

        for slot in 0..size {
            let index = self.order[self.position + slot];
            self.dataset.write(
                index,
                &mut inputs[slot * input_size..(slot + 1) * input_size],
                &mut targets[slot * target_size..(slot + 1) * target_size],
            );
        }
        self.position += size;

        Some(Batch {
            inputs: batched(inputs, size, &self.dataset.input_shape()).to(self.device),
            targets: batched(targets, size, &self.dataset.target_shape()).to(self.device),
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.order.len().saturating_sub(self.position);
        let batches = if self.drop_last {
            remaining / self.batch_size
        } else {
            remaining.div_ceil(self.batch_size)
        };
        (batches, Some(batches))
    }
}

/// Prepend the batch dimension. A `[1]` item shape collapses to `[batch]`, which
/// is what a vector of class labels should look like.
fn batched(data: Vec<f32>, size: usize, item_shape: &[usize]) -> Tensor {
    if item_shape == [1] {
        return Tensor::from_vec(data, &[size]);
    }
    let mut shape = vec![size];
    shape.extend_from_slice(item_shape);
    Tensor::from_vec(data, &shape)
}
