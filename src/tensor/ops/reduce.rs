//! Reductions over everything or along one axis. `max`, `min` and `argmax`
//! return plain numbers, since they are selections with no gradient.

use crate::autograd::ops::reduce::{MeanBackward, SumAxisBackward, SumBackward};
use crate::cuda::kernels;
use crate::tensor::shape;
use crate::tensor::storage::Storage;
use crate::tensor::Tensor;

impl Tensor {
    /// Sum of every element, as a `[1]` tensor.
    pub fn sum(&self) -> Tensor {
        let out = match self.storage() {
            Storage::Cpu(data) => Tensor::scalar(data.iter().sum()),
            Storage::Cuda(buf) => {
                let out = kernels::sum(buf, self.numel()).expect("cuda sum");
                Tensor::raw(Storage::Cuda(out), vec![1], self.device())
            }
        };
        let shape = self.shape().to_vec();
        out.with_grad(&[self], || SumBackward { shape })
    }

    /// Mean of every element, as a `[1]` tensor.
    pub fn mean(&self) -> Tensor {
        let out = match self.storage() {
            Storage::Cpu(data) => Tensor::scalar(data.iter().sum::<f32>() / data.len() as f32),
            Storage::Cuda(buf) => {
                let out = kernels::mean(buf, self.numel()).expect("cuda mean");
                Tensor::raw(Storage::Cuda(out), vec![1], self.device())
            }
        };
        let (shape, count) = (self.shape().to_vec(), self.numel());
        out.with_grad(&[self], || MeanBackward { shape, count })
    }

    /// Sum along `axis`, dropping it from the shape.
    pub fn sum_axis(&self, axis: usize) -> Tensor {
        let (outer, size, inner) = shape::split_at_axis(self.shape(), axis);

        let mut out_shape = self.shape().to_vec();
        out_shape.remove(axis);
        if out_shape.is_empty() {
            out_shape.push(1);
        }

        let out = match self.storage() {
            Storage::Cuda(buf) => {
                let out = kernels::sum_axis(buf, self.shape(), axis).expect("cuda sum_axis");
                Tensor::raw(Storage::Cuda(out), out_shape.clone(), self.device())
            }
            Storage::Cpu(data) => {
                let mut acc = vec![0.0f32; outer * inner];
                for o in 0..outer {
                    for a in 0..size {
                        for i in 0..inner {
                            acc[o * inner + i] += data[(o * size + a) * inner + i];
                        }
                    }
                }
                Tensor::from_vec(acc, &out_shape)
            }
        };

        let shape = self.shape().to_vec();
        out.with_grad(&[self], || SumAxisBackward { shape, axis })
    }

    /// Sum along `axis`, keeping it as a dimension of size 1.
    pub fn sum_axis_keep(&self, axis: usize) -> Tensor {
        self.sum_axis(axis).unsqueeze(axis)
    }

    /// Mean along `axis`, dropping it from the shape.
    pub fn mean_axis(&self, axis: usize) -> Tensor {
        let size = self.dim(axis) as f32;
        self.sum_axis(axis).div_scalar(size)
    }

    /// Mean along `axis`, keeping it as a dimension of size 1.
    pub fn mean_axis_keep(&self, axis: usize) -> Tensor {
        self.mean_axis(axis).unsqueeze(axis)
    }

    /// Largest element.
    pub fn max(&self) -> f32 {
        match self.storage() {
            Storage::Cpu(data) => data.iter().copied().fold(f32::NEG_INFINITY, f32::max),
            Storage::Cuda(buf) => single(kernels::max(buf, self.numel()).expect("cuda max")),
        }
    }

    /// Smallest element.
    pub fn min(&self) -> f32 {
        match self.storage() {
            Storage::Cpu(data) => data.iter().copied().fold(f32::INFINITY, f32::min),
            Storage::Cuda(buf) => single(kernels::min(buf, self.numel()).expect("cuda min")),
        }
    }

    /// Population variance of every element.
    ///
    /// Panics if the tensor is empty.
    pub fn variance(&self) -> f32 {
        assert!(self.numel() > 0, "variance() of an empty tensor");
        let data = self.to_vec();
        let mean = data.iter().sum::<f32>() / data.len() as f32;
        data.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / data.len() as f32
    }

    /// Index of the largest element along `axis`, for each remaining position.
    ///
    /// For `[batch, classes]` and `axis = 1` this is the predicted class per row.
    pub fn argmax(&self, axis: usize) -> Vec<usize> {
        let (outer, size, inner) = shape::split_at_axis(self.shape(), axis);
        let data = self.to_vec();

        let mut out = vec![0usize; outer * inner];
        for o in 0..outer {
            for i in 0..inner {
                let mut best = f32::NEG_INFINITY;
                for a in 0..size {
                    let value = data[(o * size + a) * inner + i];
                    if value > best {
                        best = value;
                        out[o * inner + i] = a;
                    }
                }
            }
        }
        out
    }
}

fn single(buf: crate::cuda::CudaBuffer) -> f32 {
    buf.to_vec().expect("cuda download")[0]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[should_panic(expected = "variance() of an empty tensor")]
    fn variance_rejects_an_empty_vector() {
        let tensor = Tensor::from_vec(vec![], &[0]);
        assert_eq!(tensor.numel(), 0);
        tensor.variance();
    }

    #[test]
    #[should_panic(expected = "variance() of an empty tensor")]
    fn variance_rejects_a_zero_sized_dimension() {
        let tensor = Tensor::from_vec(vec![], &[2, 0]);
        assert_eq!(tensor.numel(), 0);
        tensor.variance();
    }

    #[test]
    fn variance_of_a_singleton_is_zero() {
        assert_eq!(Tensor::from_vec(vec![3.0], &[1]).variance(), 0.0);
    }

    #[test]
    fn variance_uses_the_population_denominator() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
        assert_eq!(tensor.variance(), 1.25);
    }
}
