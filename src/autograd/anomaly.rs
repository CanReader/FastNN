//! `detect_anomaly`: panic at the backward rule that first produces a NaN or inf,
//! instead of letting it reach the optimizer. It downloads every gradient on a
//! GPU, so leave it off outside debugging.

use std::cell::Cell;

use crate::tensor::Tensor;

thread_local! {
    static ENABLED: Cell<bool> = const { Cell::new(false) };
}

/// Whether the reverse pass is checking gradients on this thread.
pub fn is_detecting() -> bool {
    ENABLED.with(|flag| flag.get())
}

/// Run `f` with gradient checking on.
pub fn detect_anomaly<T>(f: impl FnOnce() -> T) -> T {
    let _guard = Enabled::new();
    f()
}

/// Panic if `grads` holds a non-finite value, naming the op and input.
pub(crate) fn check(op: &str, grads: &[Tensor]) {
    for (index, grad) in grads.iter().enumerate() {
        if let Some(report) = first_bad(grad) {
            panic!("anomaly: {op} produced {report} in the gradient for input {index}");
        }
    }
}

fn first_bad(t: &Tensor) -> Option<String> {
    let data = t.to_vec();
    let (slot, value) = data.iter().enumerate().find(|(_, v)| !v.is_finite())?;
    let kind = if value.is_nan() { "NaN" } else { "inf" };
    Some(format!("{kind} at element {slot} of {:?}", t.shape()))
}

/// Guard, so a panic inside `detect_anomaly` cannot leave checking on.
struct Enabled(bool);

impl Enabled {
    fn new() -> Self {
        Enabled(ENABLED.with(|flag| flag.replace(true)))
    }
}

impl Drop for Enabled {
    fn drop(&mut self) {
        ENABLED.with(|flag| flag.set(self.0));
    }
}

impl Tensor {
    /// Whether every element is finite. Reads the whole tensor, so keep it out of hot loops.
    pub fn is_finite(&self) -> bool {
        self.to_vec().iter().all(|v| v.is_finite())
    }
}
