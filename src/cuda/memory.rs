use std::collections::HashMap;
use std::fmt;
use std::sync::Mutex;

extern "C" {
    fn fastnn_cuda_malloc(ptr: *mut *mut f32, size: usize) -> i32;
    fn fastnn_cuda_free(ptr: *mut f32) -> i32;
    fn fastnn_cuda_memcpy_h2d(dst: *mut f32, src: *const f32, size: usize) -> i32;
    fn fastnn_cuda_memcpy_d2h(dst: *mut f32, src: *const f32, size: usize) -> i32;
    fn fastnn_cuda_memcpy_d2d(dst: *mut f32, src: *const f32, size: usize) -> i32;
    fn fastnn_cuda_memset(ptr: *mut f32, value: i32, size: usize) -> i32;
}

// ─── Buffer cache ─────────────────────────────────────────────────────────────
// All CUDA ops use the null stream (stream 0), which is ordered: op B that runs
// after op A on the same stream is guaranteed to see A's writes. So it is safe
// to reuse a just-freed buffer for the next kernel on the same stream.
//
// This mirrors PyTorch's caching allocator and eliminates the ~3ms-per-call
// overhead of raw cudaMalloc/cudaFree for the many small temporaries created
// during a forward+backward pass.

struct RawPtr(*mut f32);
unsafe impl Send for RawPtr {}

// How many spare buffers to keep per size class before actually freeing.
const MAX_CACHE_PER_SIZE: usize = 32;

static BUFFER_CACHE: Mutex<Option<HashMap<usize, Vec<RawPtr>>>> = Mutex::new(None);

fn cache_put(len: usize, ptr: *mut f32) {
    if ptr.is_null() { return; }
    let mut guard = BUFFER_CACHE.lock().unwrap();
    let map = guard.get_or_insert_with(HashMap::new);
    let list = map.entry(len).or_default();
    if list.len() < MAX_CACHE_PER_SIZE {
        list.push(RawPtr(ptr));
    } else {
        unsafe { fastnn_cuda_free(ptr) };
    }
}

fn cache_get(len: usize) -> Option<*mut f32> {
    let mut guard = BUFFER_CACHE.lock().unwrap();
    guard.as_mut()?.get_mut(&len)?.pop().map(|p| p.0)
}

// ─── CudaBuffer ───────────────────────────────────────────────────────────────

/// RAII wrapper for GPU memory allocation.
pub struct CudaBuffer {
    ptr: *mut f32,
    len: usize, // number of f32 elements
}

// CudaBuffer is Send+Sync because GPU pointers are valid across threads
// (CUDA runtime is thread-safe for memory operations)
unsafe impl Send for CudaBuffer {}
unsafe impl Sync for CudaBuffer {}

impl CudaBuffer {
    /// Allocate `len` floats on the GPU. Returns a cached buffer if one is
    /// available, otherwise calls cudaMalloc.
    pub fn new(len: usize) -> Result<Self, String> {
        if let Some(ptr) = cache_get(len) {
            return Ok(CudaBuffer { ptr, len });
        }
        let mut ptr: *mut f32 = std::ptr::null_mut();
        let size = len * std::mem::size_of::<f32>();
        let ret = unsafe { fastnn_cuda_malloc(&mut ptr, size) };
        if ret != 0 {
            return Err(format!("Failed to allocate {} bytes on GPU", size));
        }
        Ok(CudaBuffer { ptr, len })
    }

    /// Allocate and zero-initialize.
    pub fn zeros(len: usize) -> Result<Self, String> {
        let buf = Self::new(len)?;
        let ret = unsafe { fastnn_cuda_memset(buf.ptr, 0, len * std::mem::size_of::<f32>()) };
        if ret != 0 {
            return Err("Failed to zero GPU memory".to_string());
        }
        Ok(buf)
    }

    /// Copy data from host (CPU) to this GPU buffer.
    pub fn copy_from_host(&self, data: &[f32]) -> Result<(), String> {
        assert!(data.len() <= self.len, "Source data larger than GPU buffer");
        let size = data.len() * std::mem::size_of::<f32>();
        let ret = unsafe { fastnn_cuda_memcpy_h2d(self.ptr, data.as_ptr(), size) };
        if ret != 0 {
            Err("H2D memcpy failed".to_string())
        } else {
            Ok(())
        }
    }

    /// Copy data from this GPU buffer to host (CPU).
    pub fn copy_to_host(&self, data: &mut [f32]) -> Result<(), String> {
        assert!(data.len() <= self.len, "Destination buffer smaller than GPU buffer");
        let size = data.len() * std::mem::size_of::<f32>();
        let ret = unsafe { fastnn_cuda_memcpy_d2h(data.as_mut_ptr(), self.ptr, size) };
        if ret != 0 {
            Err("D2H memcpy failed".to_string())
        } else {
            Ok(())
        }
    }

    /// Copy data from another GPU buffer to this one.
    pub fn copy_from_device(&self, other: &CudaBuffer) -> Result<(), String> {
        assert!(other.len <= self.len, "Source buffer larger than destination");
        let size = other.len * std::mem::size_of::<f32>();
        let ret = unsafe { fastnn_cuda_memcpy_d2d(self.ptr, other.ptr, size) };
        if ret != 0 {
            Err("D2D memcpy failed".to_string())
        } else {
            Ok(())
        }
    }

    /// Create a GPU buffer from host data.
    pub fn from_slice(data: &[f32]) -> Result<Self, String> {
        let buf = Self::new(data.len())?;
        buf.copy_from_host(data)?;
        Ok(buf)
    }

    /// Download all data to a Vec on the host.
    pub fn to_vec(&self) -> Result<Vec<f32>, String> {
        let mut data = vec![0.0f32; self.len];
        self.copy_to_host(&mut data)?;
        Ok(data)
    }

    pub fn ptr(&self) -> *mut f32 {
        self.ptr
    }

    pub fn as_ptr(&self) -> *const f32 {
        self.ptr as *const f32
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn size_bytes(&self) -> usize {
        self.len * std::mem::size_of::<f32>()
    }
}

impl Drop for CudaBuffer {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            cache_put(self.len, self.ptr);
            self.ptr = std::ptr::null_mut();
        }
    }
}

impl Clone for CudaBuffer {
    fn clone(&self) -> Self {
        let new_buf = CudaBuffer::new(self.len).expect("Failed to clone CudaBuffer");
        new_buf.copy_from_device(self).expect("Failed to copy CudaBuffer");
        new_buf
    }
}

impl fmt::Debug for CudaBuffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CudaBuffer(len={}, ptr={:?})", self.len, self.ptr)
    }
}
