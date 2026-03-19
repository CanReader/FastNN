use crate::tensor::Tensor;
use crate::data::dataset::Dataset;
use rand::seq::SliceRandom;

/// Iterates over a dataset in batches with optional shuffling.
pub struct DataLoader<'a> {
    dataset: &'a dyn Dataset,
    batch_size: usize,
    shuffle: bool,
    drop_last: bool,
}

impl<'a> DataLoader<'a> {
    pub fn new(dataset: &'a dyn Dataset, batch_size: usize) -> Self {
        DataLoader {
            dataset,
            batch_size,
            shuffle: false,
            drop_last: false,
        }
    }

    pub fn shuffle(mut self, shuffle: bool) -> Self {
        self.shuffle = shuffle;
        self
    }

    pub fn drop_last(mut self, drop_last: bool) -> Self {
        self.drop_last = drop_last;
        self
    }

    /// Number of batches.
    pub fn num_batches(&self) -> usize {
        let n = self.dataset.len();
        if self.drop_last {
            n / self.batch_size
        } else {
            (n + self.batch_size - 1) / self.batch_size
        }
    }

    /// Iterate over batches. Returns Vec of (input_batch, target_batch).
    pub fn iter<'b>(&'b self) -> DataLoaderIterator<'b> {
        let n = self.dataset.len();
        let mut indices: Vec<usize> = (0..n).collect();

        if self.shuffle {
            let mut rng = rand::thread_rng();
            indices.shuffle(&mut rng);
        }

        DataLoaderIterator {
            dataset: self.dataset,
            indices,
            batch_size: self.batch_size,
            position: 0,
            drop_last: self.drop_last,
        }
    }
}

pub struct DataLoaderIterator<'a> {
    dataset: &'a dyn Dataset,
    indices: Vec<usize>,
    batch_size: usize,
    position: usize,
    drop_last: bool,
}

impl<'a> Iterator for DataLoaderIterator<'a> {
    type Item = (Tensor, Tensor);

    fn next(&mut self) -> Option<Self::Item> {
        if self.position >= self.indices.len() {
            return None;
        }

        let remaining = self.indices.len() - self.position;
        let actual_batch = remaining.min(self.batch_size);

        if actual_batch < self.batch_size && self.drop_last {
            return None;
        }

        let batch_indices: Vec<usize> = self.indices[self.position..self.position + actual_batch].to_vec();
        self.position += actual_batch;

        // Collect samples
        let mut input_vecs = Vec::new();
        let mut target_vecs = Vec::new();
        let mut input_shape = Vec::new();
        let mut target_shape = Vec::new();

        for (i, &idx) in batch_indices.iter().enumerate() {
            let (input, target) = self.dataset.get(idx);
            if i == 0 {
                input_shape = input.shape().to_vec();
                target_shape = target.shape().to_vec();
            }
            input_vecs.extend_from_slice(&input.to_vec());
            target_vecs.extend_from_slice(&target.to_vec());
        }

        // Build batched tensors
        let mut batch_input_shape = vec![actual_batch];
        batch_input_shape.extend(&input_shape);
        let mut batch_target_shape = vec![actual_batch];
        batch_target_shape.extend(&target_shape);

        let input_batch = Tensor::from_vec(input_vecs, &batch_input_shape);
        let target_batch = Tensor::from_vec(target_vecs, &batch_target_shape);

        Some((input_batch, target_batch))
    }
}

impl<'a> IntoIterator for &'a DataLoader<'a> {
    type Item = (Tensor, Tensor);
    type IntoIter = DataLoaderIterator<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}
