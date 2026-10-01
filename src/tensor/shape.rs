//! Shape arithmetic: strides, broadcasting, and flat-index math.
//!
//! Tensors are always C-contiguous, so strides are derived from the shape rather
//! than stored as an independent source of truth. Every view op (`reshape`,
//! `permute`, `expand`) materialises a new contiguous buffer.

/// Row-major strides for `shape`, in elements.
pub fn strides_for(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

/// Total element count.
pub fn numel(shape: &[usize]) -> usize {
    shape.iter().product()
}

/// Resolve a shape that may contain a single `-1` placeholder against `numel`.
///
/// Panics if more than one dimension is `-1` or the element count does not divide.
pub fn resolve(shape: &[i64], numel: usize) -> Vec<usize> {
    let mut placeholder = None;
    let mut known = 1usize;
    for (i, &d) in shape.iter().enumerate() {
        if d == -1 {
            assert!(
                placeholder.is_none(),
                "reshape: at most one -1, got {shape:?}"
            );
            placeholder = Some(i);
        } else {
            assert!(
                d > 0,
                "reshape: dimensions must be positive or -1, got {shape:?}"
            );
            known *= d as usize;
        }
    }

    match placeholder {
        Some(i) => {
            assert!(
                known > 0 && numel.is_multiple_of(known),
                "reshape: cannot fit {numel} elements into {shape:?}"
            );
            let mut out: Vec<usize> = shape.iter().map(|&d| d.max(0) as usize).collect();
            out[i] = numel / known;
            out
        }
        None => {
            assert_eq!(
                known, numel,
                "reshape: {shape:?} holds {known} elements, need {numel}"
            );
            shape.iter().map(|&d| d as usize).collect()
        }
    }
}

/// The shape two operands broadcast to, following NumPy rules.
///
/// Panics if the shapes are incompatible.
pub fn broadcast(a: &[usize], b: &[usize]) -> Vec<usize> {
    let ndim = a.len().max(b.len());
    (0..ndim)
        .map(|i| {
            let da = dim_right_aligned(a, ndim, i);
            let db = dim_right_aligned(b, ndim, i);
            assert!(
                da == db || da == 1 || db == 1,
                "cannot broadcast shapes {a:?} and {b:?}"
            );
            da.max(db)
        })
        .collect()
}

/// Left-pad `shape` with 1s to `ndim` dimensions.
pub fn pad_left(shape: &[usize], ndim: usize) -> Vec<usize> {
    let mut out = vec![1usize; ndim - shape.len()];
    out.extend_from_slice(shape);
    out
}

/// Strides for reading `shape` (already padded to `out.len()`) while broadcasting
/// to `out`. Broadcast dimensions get stride 0 so every output index reads slot 0.
pub fn broadcast_strides(shape: &[usize], out: &[usize]) -> Vec<usize> {
    debug_assert_eq!(shape.len(), out.len());
    strides_for(shape)
        .iter()
        .zip(shape)
        .map(|(&s, &d)| if d == 1 { 0 } else { s })
        .collect()
}

/// Split `shape` around `axis` into (product before, size at, product after).
///
/// Most reductions and axis-wise scatters reduce to a triple loop over these.
pub fn split_at_axis(shape: &[usize], axis: usize) -> (usize, usize, usize) {
    assert!(
        axis < shape.len(),
        "axis {axis} out of range for shape {shape:?}"
    );
    (
        numel(&shape[..axis]),
        shape[axis],
        numel(&shape[axis + 1..]),
    )
}

fn dim_right_aligned(shape: &[usize], ndim: usize, i: usize) -> usize {
    let offset = ndim - shape.len();
    if i < offset {
        1
    } else {
        shape[i - offset]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strides_are_row_major() {
        assert_eq!(strides_for(&[2, 3, 4]), vec![12, 4, 1]);
        assert_eq!(strides_for(&[5]), vec![1]);
    }

    #[test]
    fn resolve_infers_placeholder() {
        assert_eq!(resolve(&[2, -1], 12), vec![2, 6]);
        assert_eq!(resolve(&[3, 4], 12), vec![3, 4]);
    }

    #[test]
    fn broadcast_right_aligns() {
        assert_eq!(broadcast(&[3, 1], &[4]), vec![3, 4]);
        assert_eq!(broadcast(&[2, 1, 4], &[3, 1]), vec![2, 3, 4]);
    }

    #[test]
    fn axis_split_matches_shape() {
        assert_eq!(split_at_axis(&[2, 3, 4], 1), (2, 3, 4));
        assert_eq!(split_at_axis(&[2, 3, 4], 0), (1, 2, 12));
    }
}
