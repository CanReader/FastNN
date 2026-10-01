//! Printing tensors.
//!
//! `{:?}` gives the one-line summary you want in a log; `{}` prints values, with
//! long runs elided so a debug print never floods a terminal.

use std::fmt;

use super::Tensor;

/// Values shown at each end before eliding the middle.
const EDGE: usize = 3;
/// Rows shown before eliding the rest of a matrix.
const MAX_ROWS: usize = 6;

impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Tensor{:?} on {}{}",
            self.shape(),
            self.device(),
            if self.grad_fn().is_some() {
                ", tracked"
            } else {
                ""
            }
        )
    }
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let data = self.to_vec();
        match self.ndim() {
            0 | 1 => {
                write!(f, "[")?;
                write_row(f, &data)?;
                write!(f, "]  {:?}", self)
            }
            2 => {
                let (rows, cols) = (self.dim(0), self.dim(1));
                writeln!(f, "[")?;
                for r in 0..rows.min(MAX_ROWS) {
                    write!(f, "  [")?;
                    write_row(f, &data[r * cols..(r + 1) * cols])?;
                    writeln!(f, "]")?;
                }
                if rows > MAX_ROWS {
                    writeln!(f, "  ... {} more rows", rows - MAX_ROWS)?;
                }
                write!(f, "]  {:?}", self)
            }
            // Higher ranks have no layout that reads well in a terminal; the
            // summary plus the first few values is more useful than a wall of text.
            _ => {
                write!(f, "[")?;
                write_row(f, &data)?;
                write!(f, "]  {:?}", self)
            }
        }
    }
}

fn write_row(f: &mut fmt::Formatter<'_>, values: &[f32]) -> fmt::Result {
    let separated = |f: &mut fmt::Formatter<'_>, slice: &[f32], leading: bool| -> fmt::Result {
        for (i, v) in slice.iter().enumerate() {
            if leading || i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{v:.4}")?;
        }
        Ok(())
    };

    if values.len() <= EDGE * 2 + 1 {
        return separated(f, values, false);
    }
    separated(f, &values[..EDGE], false)?;
    write!(f, ", ...")?;
    separated(f, &values[values.len() - EDGE..], true)
}
