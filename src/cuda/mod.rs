//! GPU layer: device setup, memory, and the kernel bindings.
//!
//! - [`ffi`] — raw `extern "C"` declarations, one per symbol in `cuda/kernels.cu`
//! - [`kernels`] — safe wrappers that allocate outputs and check status codes
//! - [`buffer`] — [`CudaBuffer`], RAII GPU memory over a caching allocator
//!
//! In a `--no-default-features` build, `cuda/stubs.c` supplies the symbols so
//! everything links; [`device_count`] then reports 0 and CPU paths run instead.

pub mod buffer;
pub mod ffi;
pub mod kernels;

pub use buffer::{empty_cache, CudaBuffer};

use std::sync::OnceLock;

use crate::error::{Error, Result};

/// Initialise the CUDA context on `device`. Idempotent; safe to call per tensor.
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

/// Number of CUDA devices visible to the process. 0 in a CPU-only build.
pub fn device_count() -> usize {
    unsafe { ffi::fastnn_cuda_device_count().max(0) as usize }
}

/// Block until every queued kernel has finished.
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
