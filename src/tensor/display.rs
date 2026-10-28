//! `{:?}` is a one-line summary, `{}` prints the values (long runs elided).

use std::fmt;

use super::Tensor;

/// Values kept at each end of an elided run.
const EDGE: usize = 3;
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
            // no layout reads well past 2-D, so print the flat edges
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
