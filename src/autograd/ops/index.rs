//! Derivative of row selection.

use crate::autograd::Backward;
use crate::cuda::kernels;
use crate::tensor::storage::Storage;
use crate::tensor::Tensor;

/// Scatter-add each gathered row's gradient back into its source row.
///
/// Rows selected more than once accumulate, which is how a token repeated in a
/// batch contributes to its embedding once per occurrence.
pub struct IndexSelectBackward {
    pub ids: Vec<usize>,
    pub rows: usize,
    pub cols: usize,
}

impl Backward for IndexSelectBackward {
    fn backward(&self, grad: &Tensor) -> Vec<Tensor> {
        if let Storage::Cuda(g) = grad.storage() {
            let ids_i32: Vec<i32> = self.ids.iter().map(|&i| i as i32).collect();
            let ids_buf = kernels::upload_ids(&ids_i32).expect("cuda upload ids");
            let table =
                kernels::embedding_backward(&ids_buf, g, self.ids.len(), self.cols, self.rows)
                    .expect("cuda embedding backward");
            return vec![Tensor::raw(
                Storage::Cuda(table),
                vec![self.rows, self.cols],
                grad.device(),
            )];
        }

        let g = grad.to_vec();
        let mut table = vec![0.0f32; self.rows * self.cols];
        for (position, &id) in self.ids.iter().enumerate() {
            let (dst, src) = (id * self.cols, position * self.cols);
            for c in 0..self.cols {
                table[dst + c] += g[src + c];
            }
        }
        vec![Tensor::from_vec(table, &[self.rows, self.cols]).to(grad.device())]
    }
    fn name(&self) -> &'static str {
        "IndexSelect"
    }
}
