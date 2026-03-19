use crate::tensor::Tensor;

/// Trait for datasets — provides indexed access to samples.
pub trait Dataset: Send + Sync {
    /// Number of samples in the dataset.
    fn len(&self) -> usize;

    fn is_empty(&self) -> bool { self.len() == 0 }

    /// Get a single sample: (input, target).
    fn get(&self, index: usize) -> (Tensor, Tensor);
}

/// In-memory dataset backed by tensors.
pub struct TensorDataset {
    inputs: Tensor,
    targets: Tensor,
}

impl TensorDataset {
    /// Create from full input and target tensors.
    /// Both tensors must have the same first dimension (batch size).
    pub fn new(inputs: Tensor, targets: Tensor) -> Self {
        assert_eq!(inputs.shape()[0], targets.shape()[0],
                   "Input and target batch sizes must match");
        TensorDataset { inputs, targets }
    }
}

impl Dataset for TensorDataset {
    fn len(&self) -> usize {
        self.inputs.shape()[0]
    }

    fn get(&self, index: usize) -> (Tensor, Tensor) {
        let input_data = self.inputs.to_vec();
        let target_data = self.targets.to_vec();

        let input_sample_size: usize = self.inputs.shape()[1..].iter().product();
        let target_sample_size: usize = if self.targets.ndim() > 1 {
            self.targets.shape()[1..].iter().product()
        } else {
            1
        };

        let input_start = index * input_sample_size;
        let input_vec = input_data[input_start..input_start + input_sample_size].to_vec();
        let input_shape: Vec<usize> = self.inputs.shape()[1..].to_vec();

        let target_start = index * target_sample_size;
        let target_vec = target_data[target_start..target_start + target_sample_size].to_vec();
        let target_shape: Vec<usize> = if self.targets.ndim() > 1 {
            self.targets.shape()[1..].to_vec()
        } else {
            vec![1]
        };

        (
            Tensor::from_vec(input_vec, &input_shape),
            Tensor::from_vec(target_vec, &target_shape),
        )
    }
}

/// Dataset from a vector of (input, target) pairs.
pub struct VecDataset {
    samples: Vec<(Tensor, Tensor)>,
}

impl VecDataset {
    pub fn new(samples: Vec<(Tensor, Tensor)>) -> Self {
        VecDataset { samples }
    }
}

impl Dataset for VecDataset {
    fn len(&self) -> usize { self.samples.len() }

    fn get(&self, index: usize) -> (Tensor, Tensor) {
        let (input, target) = &self.samples[index];
        (input.clone(), target.clone())
    }
}
