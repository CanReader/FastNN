//! Row selection, the lookup behind embeddings.

use crate::autograd::ops::index::IndexSelectBackward;
use crate::cuda::kernels;
use crate::tensor::Tensor;
use crate::tensor::storage::Storage;

/// Convert a float index without truncating or saturating it.
pub(crate) fn index_from_f32(value: f32) -> usize {
    assert!(
        value.is_finite()
            && value >= 0.0
            && value.fract() == 0.0
            && f64::from(value) < usize::MAX as f64 + 1.0,
        "invalid index {value}: expected a finite, non-negative integer that fits usize"
    );
    value as usize
}

impl Tensor {
    /// Gather rows `ids` from a `[rows, cols]` table. Repeated ids are fine.
    pub fn index_select(&self, ids: &[usize]) -> Tensor {
        assert_eq!(
            self.ndim(),
            2,
            "index_select needs a 2-D table, got {:?}",
            self.shape()
        );
        let (rows, cols) = (self.dim(0), self.dim(1));
        for &id in ids {
            assert!(id < rows, "index_select: row {id} outside 0..{rows}");
        }

        let out = match self.storage() {
            Storage::Cuda(weight) => {
                let ids_i32: Vec<i32> = ids.iter().map(|&i| i as i32).collect();
                let ids_buf = kernels::upload_ids(&ids_i32).expect("cuda upload ids");
                let gathered = kernels::embedding_forward(&ids_buf, weight, ids.len(), cols)
                    .expect("cuda embedding_forward");
                Tensor::raw(
                    Storage::Cuda(gathered),
                    vec![ids.len(), cols],
                    self.device(),
                )
            }
            Storage::Cpu(table) => {
                let mut data = Vec::with_capacity(ids.len() * cols);
                for &id in ids {
                    data.extend_from_slice(&table[id * cols..(id + 1) * cols]);
                }
                Tensor::from_vec(data, &[ids.len(), cols])
            }
        };

        let ids = ids.to_vec();
        out.with_grad(&[self], || IndexSelectBackward { ids, rows, cols })
    }
}
