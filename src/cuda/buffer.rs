//! GPU memory: an RAII buffer over a caching allocator.

use std::collections::HashMap;
use std::fmt;
use std::sync::Mutex;

use super::ffi;
use crate::error::{Error, Result};

const F32: usize = std::mem::size_of::<f32>();

/// An owned block of `len` floats in GPU memory.
///
/// Dropping returns the block to a free list instead of calling `cudaFree`, which is slow.
pub struct CudaBuffer {
    ptr: *mut f32,
    len: usize,
}

// Device pointers are not tied to a thread, and the runtime serialises memory calls.
unsafe impl Send for CudaBuffer {}
unsafe impl Sync for CudaBuffer {}

impl CudaBuffer {
    /// Allocates `len` floats, reusing a cached block when one is free.
    ///
    /// On failure the cache is drained and the allocation retried once, since a changed shape
    /// (a short last batch, a longer sequence) can leave memory parked in the wrong sizes.
    pub fn new(len: usize) -> Result<Self> {
        if let Some(ptr) = cache::take(len) {
            return Ok(CudaBuffer { ptr, len });
        }
        match raw_alloc(len) {
            Ok(ptr) => Ok(CudaBuffer { ptr, len }),
            Err(_) => {
                cache::drain();
                raw_alloc(len).map(|ptr| CudaBuffer { ptr, len })
            }
        }
    }

    /// Allocates `len` zeroed floats.
    pub fn zeros(len: usize) -> Result<Self> {
        let buf = Self::new(len)?;
        check(
            unsafe { ffi::fastnn_cuda_memset(buf.ptr, 0, len * F32) },
            || "memset failed".into(),
        )?;
        Ok(buf)
    }

    /// Allocates a buffer holding `data`.
    pub fn from_slice(data: &[f32]) -> Result<Self> {
        let buf = Self::new(data.len())?;
        buf.upload(data)?;
        Ok(buf)
    }

    /// Copies host memory in. `data` must fit.
    pub fn upload(&self, data: &[f32]) -> Result<()> {
        assert!(
            data.len() <= self.len,
            "upload: {} floats into a buffer of {}",
            data.len(),
            self.len
        );
        check(
            unsafe { ffi::fastnn_cuda_memcpy_h2d(self.ptr, data.as_ptr(), data.len() * F32) },
            || "host-to-device copy failed".into(),
        )
    }

    /// Copies the whole buffer to the host.
    pub fn to_vec(&self) -> Result<Vec<f32>> {
        let mut out = vec![0.0f32; self.len];
        check(
            unsafe { ffi::fastnn_cuda_memcpy_d2h(out.as_mut_ptr(), self.ptr, self.len * F32) },
            || "device-to-host copy failed".into(),
        )?;
        Ok(out)
    }

    /// Copies `src` into this buffer on the device.
    pub fn copy_from(&self, src: &CudaBuffer) -> Result<()> {
        assert!(
            src.len <= self.len,
            "copy_from: {} floats into a buffer of {}",
            src.len,
            self.len
        );
        check(
            unsafe { ffi::fastnn_cuda_memcpy_d2d(self.ptr, src.ptr, src.len * F32) },
            || "device-to-device copy failed".into(),
        )
    }

    pub fn as_ptr(&self) -> *const f32 {
        self.ptr
    }

    pub fn as_mut_ptr(&self) -> *mut f32 {
        self.ptr
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl Drop for CudaBuffer {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            cache::give_back(self.len, self.ptr);
            self.ptr = std::ptr::null_mut();
        }
    }
}

impl Clone for CudaBuffer {
    fn clone(&self) -> Self {
        let copy = CudaBuffer::new(self.len).expect("cuda: clone allocation failed");
        copy.copy_from(self).expect("cuda: clone copy failed");
        copy
    }
}

impl fmt::Debug for CudaBuffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CudaBuffer(len={})", self.len)
    }
}

fn check(code: std::ffi::c_int, msg: impl FnOnce() -> String) -> Result<()> {
    if code == 0 {
        Ok(())
    } else {
        Err(Error::Cuda(msg()))
    }
}

/// One `cudaMalloc`, with no cache involved.
fn raw_alloc(len: usize) -> Result<*mut f32> {
    let mut ptr: *mut f32 = std::ptr::null_mut();
    check(
        unsafe { ffi::fastnn_cuda_malloc(&mut ptr, len * F32) },
        || {
            let (free, total) = super::memory_info();
            format!(
                "out of memory: wanted {} bytes, {free} of {total} free",
                len * F32
            )
        },
    )?;
    Ok(ptr)
}

/// Free list of blocks keyed by element count. Safe to reuse right away because every kernel
/// runs on the ordered default stream.
mod cache {
    use super::{HashMap, Mutex, ffi};

    /// Spare blocks kept per size before memory goes back to the driver.
    const MAX_PER_SIZE: usize = 32;

    struct Block(*mut f32);
    unsafe impl Send for Block {}

    static FREE_LIST: Mutex<Option<HashMap<usize, Vec<Block>>>> = Mutex::new(None);

    pub fn take(len: usize) -> Option<*mut f32> {
        let mut guard = FREE_LIST.lock().ok()?;
        guard.as_mut()?.get_mut(&len)?.pop().map(|b| b.0)
    }

    pub fn give_back(len: usize, ptr: *mut f32) {
        if ptr.is_null() {
            return;
        }
        let Ok(mut guard) = FREE_LIST.lock() else {
            return;
        };
        let blocks = guard
            .get_or_insert_with(HashMap::new)
            .entry(len)
            .or_default();
        if blocks.len() < MAX_PER_SIZE {
            blocks.push(Block(ptr));
        } else {
            unsafe { ffi::fastnn_cuda_free(ptr) };
        }
    }

    pub(super) fn drain() {
        let Ok(mut guard) = FREE_LIST.lock() else {
            return;
        };
        if let Some(map) = guard.as_mut() {
            for (_, blocks) in map.drain() {
                for b in blocks {
                    unsafe { ffi::fastnn_cuda_free(b.0) };
                }
            }
        }
    }
}

/// Frees every cached block. Only needed when another library wants the memory.
pub fn empty_cache() {
    cache::drain();
}
