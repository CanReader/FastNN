use std::sync::Once;

static CUDA_INIT: Once = Once::new();

extern "C" {
    fn fastdl_cuda_init(device_id: i32) -> i32;
    fn fastdl_cuda_device_count() -> i32;
    fn fastdl_cuda_synchronize() -> i32;
    fn fastdl_cuda_get_memory_info(free: *mut usize, total: *mut usize);
}

/// Manages CUDA device context and initialization.
pub struct CudaContext {
    device_id: i32,
}

impl CudaContext {
    /// Initialize CUDA on the given device. Only performs actual init once.
    pub fn new(device_id: i32) -> Result<Self, String> {
        let mut result = Ok(());
        CUDA_INIT.call_once(|| {
            let ret = unsafe { fastdl_cuda_init(device_id) };
            if ret != 0 {
                result = Err(format!("Failed to initialize CUDA device {}", device_id));
            }
        });
        result?;
        Ok(CudaContext { device_id })
    }

    pub fn device_id(&self) -> i32 {
        self.device_id
    }

    pub fn device_count() -> i32 {
        unsafe { fastdl_cuda_device_count() }
    }

    pub fn synchronize() -> Result<(), String> {
        let ret = unsafe { fastdl_cuda_synchronize() };
        if ret != 0 {
            Err("CUDA synchronize failed".to_string())
        } else {
            Ok(())
        }
    }

    pub fn memory_info() -> (usize, usize) {
        let mut free = 0usize;
        let mut total = 0usize;
        unsafe { fastdl_cuda_get_memory_info(&mut free, &mut total) };
        (free, total)
    }
}
