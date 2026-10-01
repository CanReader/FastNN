//! Ops that rearrange elements without doing arithmetic.
//!
//! Tensors are always contiguous, so `reshape` is free (it shares storage and
//! only changes the shape vector) while `permute`, `expand`, and `cat` write a
//! new buffer.

use crate::autograd::ops::view::{
    CatBackward, ExpandBackward, NarrowBackward, PermuteBackward, ReshapeBackward,
};
use crate::cuda::kernels;
use crate::tensor::shape;
use crate::tensor::storage::Storage;
use crate::tensor::Tensor;

impl Tensor {
    /// Reinterpret the elements under a new shape. One dimension may be `-1`.
    ///
    /// Free: the result shares storage with the input.
    pub fn reshape(&self, dims: &[i64]) -> Tensor {
        let new_shape = shape::resolve(dims, self.numel());
        let old_shape = self.shape().to_vec();
        reshaped(self, new_shape).with_grad(&[self], || ReshapeBackward { shape: old_shape })
    }

    /// Collapse everything into one dimension.
    pub fn flatten(&self) -> Tensor {
        self.reshape(&[-1])
    }

    /// Collapse dimensions `from..` into one, keeping the dimensions before it.
    ///
    /// `flatten_from(1)` is the usual conv-to-linear bridge: `[N, C, H, W] → [N, C*H*W]`.
    pub fn flatten_from(&self, from: usize) -> Tensor {
        let mut dims: Vec<i64> = self.shape()[..from].iter().map(|&d| d as i64).collect();
        dims.push(-1);
        self.reshape(&dims)
    }

    /// Insert a dimension of size 1 at `axis`.
    pub fn unsqueeze(&self, axis: usize) -> Tensor {
        assert!(
            axis <= self.ndim(),
            "unsqueeze: axis {axis} past shape {:?}",
            self.shape()
        );
        let mut dims: Vec<i64> = self.shape().iter().map(|&d| d as i64).collect();
        dims.insert(axis, 1);
        self.reshape(&dims)
    }

    /// Drop the dimension at `axis`, which must have size 1.
    pub fn squeeze(&self, axis: usize) -> Tensor {
        assert_eq!(
            self.dim(axis),
            1,
            "squeeze: axis {axis} has size {} in shape {:?}",
            self.dim(axis),
            self.shape()
        );
        let mut dims: Vec<i64> = self.shape().iter().map(|&d| d as i64).collect();
        dims.remove(axis);
        if dims.is_empty() {
            dims.push(1);
        }
        self.reshape(&dims)
    }

    /// Drop every dimension of size 1.
    pub fn squeeze_all(&self) -> Tensor {
        let dims: Vec<i64> = self
            .shape()
            .iter()
            .filter(|&&d| d != 1)
            .map(|&d| d as i64)
            .collect();
        self.reshape(if dims.is_empty() { &[1] } else { &dims })
    }

    /// Reorder dimensions: output dimension `i` is input dimension `order[i]`.
    pub fn permute(&self, order: &[usize]) -> Tensor {
        assert_eq!(
            order.len(),
            self.ndim(),
            "permute: {order:?} does not cover shape {:?}",
            self.shape()
        );
        let out_shape: Vec<usize> = order.iter().map(|&d| self.dim(d)).collect();
        let in_strides = shape::strides_for(self.shape());
        let order = order.to_vec();

        let out = gather(self, out_shape, &in_strides, &order);
        out.with_grad(&[self], || PermuteBackward { order })
    }

    /// Swap the last two dimensions.
    pub fn transpose(&self) -> Tensor {
        assert!(
            self.ndim() >= 2,
            "transpose needs 2+ dimensions, got {:?}",
            self.shape()
        );
        let n = self.ndim();
        let mut order: Vec<usize> = (0..n).collect();
        order.swap(n - 2, n - 1);
        self.permute(&order)
    }

    /// Alias for [`transpose`](Tensor::transpose).
    pub fn t(&self) -> Tensor {
        self.transpose()
    }

    /// Repeat size-1 dimensions out to `to`, which must have the same rank.
    pub fn expand(&self, to: &[usize]) -> Tensor {
        assert_eq!(
            to.len(),
            self.ndim(),
            "expand: {to:?} does not match rank of {:?}",
            self.shape()
        );
        let from = self.shape().to_vec();
        expanded(self, to).with_grad(&[self], || ExpandBackward { shape: from })
    }

    /// Tile each dimension `times[i]` times.
    pub fn repeat(&self, times: &[usize]) -> Tensor {
        let to: Vec<usize> = self
            .shape()
            .iter()
            .zip(times)
            .map(|(&d, &n)| d * n)
            .collect();
        self.expand(&to)
    }

    /// A `len`-long slice of dimension `axis`, starting at `start`.
    pub fn narrow(&self, axis: usize, start: usize, len: usize) -> Tensor {
        let (outer, size, inner) = shape::split_at_axis(self.shape(), axis);
        assert!(
            start + len <= size,
            "narrow: {start}..{} outside axis {axis} of shape {:?}",
            start + len,
            self.shape()
        );

        let src = self.to_vec();
        let mut data = Vec::with_capacity(outer * len * inner);
        for o in 0..outer {
            let row = (o * size + start) * inner;
            data.extend_from_slice(&src[row..row + len * inner]);
        }

        let mut out_shape = self.shape().to_vec();
        out_shape[axis] = len;
        let in_shape = self.shape().to_vec();

        Tensor::from_vec(data, &out_shape)
            .to(self.device())
            .with_grad(&[self], || NarrowBackward {
                shape: in_shape,
                axis,
                start,
            })
    }

    /// Join tensors along an existing dimension. All other dimensions must match.
    pub fn cat(parts: &[&Tensor], axis: usize) -> Tensor {
        assert!(!parts.is_empty(), "cat: needs at least one tensor");
        let first = parts[0];

        let mut out_shape = first.shape().to_vec();
        out_shape[axis] = parts.iter().map(|t| t.dim(axis)).sum();
        for t in parts {
            assert_eq!(
                t.ndim(),
                first.ndim(),
                "cat: rank mismatch {:?} vs {:?}",
                t.shape(),
                first.shape()
            );
            for d in 0..t.ndim() {
                assert!(
                    d == axis || t.dim(d) == first.dim(d),
                    "cat: shapes {:?} and {:?} differ outside axis {axis}",
                    t.shape(),
                    first.shape()
                );
            }
        }

        // Interleave: for each outer slice, append every part's chunk in order.
        let (outer, _, inner) = shape::split_at_axis(&out_shape, axis);
        let sources: Vec<Vec<f32>> = parts.iter().map(|t| t.to_vec()).collect();
        let mut data = Vec::with_capacity(shape::numel(&out_shape));
        for o in 0..outer {
            for (part, src) in parts.iter().zip(&sources) {
                let chunk = part.dim(axis) * inner;
                data.extend_from_slice(&src[o * chunk..(o + 1) * chunk]);
            }
        }

        let sizes: Vec<usize> = parts.iter().map(|t| t.dim(axis)).collect();
        let refs: Vec<&Tensor> = parts.to_vec();
        Tensor::from_vec(data, &out_shape)
            .to(first.device())
            .with_grad(&refs, || CatBackward { axis, sizes })
    }

    /// Join tensors along a new dimension inserted at `axis`.
    pub fn stack(parts: &[&Tensor], axis: usize) -> Tensor {
        let lifted: Vec<Tensor> = parts.iter().map(|t| t.unsqueeze(axis)).collect();
        Tensor::cat(&lifted.iter().collect::<Vec<_>>(), axis)
    }
}

// ── Shared machinery, also used by the backward rules ────────────────────────

/// Same shape change as [`Tensor::reshape`], with no graph node.
pub(crate) fn reshaped(x: &Tensor, shape: Vec<usize>) -> Tensor {
    debug_assert_eq!(shape::numel(&shape), x.numel());
    Tensor::raw_shared(x.storage_arc(), shape, x.device())
}

/// Same expansion as [`Tensor::expand`], with no graph node.
pub(crate) fn expanded(x: &Tensor, to: &[usize]) -> Tensor {
    if x.shape() == to {
        return x.detach();
    }
    // Stride 0 on a broadcast dimension makes every output index read slot 0,
    // which is exactly what the permute kernel needs to double as `expand`.
    let in_strides: Vec<usize> = shape::strides_for(x.shape())
        .iter()
        .enumerate()
        .map(|(d, &s)| if x.dim(d) == 1 { 0 } else { s })
        .collect();
    let identity: Vec<usize> = (0..x.ndim()).collect();
    gather(x, to.to_vec(), &in_strides, &identity)
}

/// Reshape then expand, so a lower-rank tensor can broadcast against `to`.
pub(crate) fn broadcast_to(x: &Tensor, to: &[usize]) -> Tensor {
    if x.shape() == to {
        return x.detach();
    }
    let padded = shape::pad_left(x.shape(), to.len());
    expanded(&reshaped(x, padded), to)
}

/// Build `out_shape` by reading each output position from `in_strides[order[d]]`.
///
/// This one gather covers both `permute` (a real reordering) and `expand`
/// (identity order, zero strides on broadcast dimensions).
fn gather(x: &Tensor, out_shape: Vec<usize>, in_strides: &[usize], order: &[usize]) -> Tensor {
    let numel = shape::numel(&out_shape);
    let out_strides = shape::strides_for(&out_shape);

    if let Storage::Cuda(buf) = x.storage() {
        let out = kernels::permute_nd(buf, &out_strides, in_strides, order, numel)
            .expect("cuda permute_nd");
        return Tensor::raw(Storage::Cuda(out), out_shape, x.device());
    }

    let src = x.to_vec();
    let ndim = out_shape.len();
    let data: Vec<f32> = (0..numel)
        .map(|flat| {
            let (mut rest, mut src_index) = (flat, 0usize);
            for d in 0..ndim {
                let coord = rest / out_strides[d];
                rest %= out_strides[d];
                src_index += coord * in_strides[order[d]];
            }
            src[src_index]
        })
        .collect();

    Tensor::from_vec(data, &out_shape)
}
