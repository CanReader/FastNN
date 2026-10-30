//! GPU layer: device setup, memory ([`buffer`]), raw bindings ([`ffi`]) and safe wrappers ([`kernels`]).
//!
//! A training script never calls this module directly. Pick a device with
//! [`Device::best`](crate::tensor::Device::best) or
//! [`Device::cuda`](crate::tensor::Device::cuda) and move the model and the loader
//! to it.
//!
//! Without the `cuda` feature, `cuda/stubs.c` stands in for the kernels and
//! [`device_count`] is 0. With it, `build.rs` compiles `cuda/kernels.cu` with
//! `nvcc`. Matmul goes through cuBLAS. The other kernels use fast math, so GPU
//! results differ from the CPU in the last few digits, and `tests/cuda_parity.rs`
//! checks each one against the CPU path.

pub mod buffer;
pub mod ffi;
pub mod kernels;

pub use buffer::{CudaBuffer, empty_cache};

use std::sync::OnceLock;

use crate::error::{Error, Result};

/// Sets up the CUDA context on `device`. Only the first call does anything.
pub fn init(device: usize) -> Result<()> {
    static INIT: OnceLock<std::result::Result<(), String>> = OnceLock::new();
    INIT.get_or_init(|| {
        let code = unsafe { ffi::fastnn_cuda_init(device as i32) };
        if code == 0 {
            Ok(())
        } else {
            Err(format!("could not initialise device {device}"))
        }
    })
    .clone()
    .map_err(Error::Cuda)
}

/// Number of visible CUDA devices, 0 in a CPU-only build.
pub fn device_count() -> usize {
    unsafe { ffi::fastnn_cuda_device_count().max(0) as usize }
}

/// Waits for every queued kernel to finish.
pub fn synchronize() -> Result<()> {
    match unsafe { ffi::fastnn_cuda_synchronize() } {
        0 => Ok(()),
        code => Err(Error::Cuda(format!("synchronize failed (status {code})"))),
    }
}

/// `(free, total)` device memory in bytes.
pub fn memory_info() -> (usize, usize) {
    let (mut free, mut total) = (0usize, 0usize);
    unsafe { ffi::fastnn_cuda_get_memory_info(&mut free, &mut total) };
    (free, total)
}
