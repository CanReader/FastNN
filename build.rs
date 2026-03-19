use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=cuda/");

    if cfg!(feature = "cuda") {
        build_cuda();
    }
}

fn build_cuda() {
    // Find CUDA toolkit
    let cuda_path = env::var("CUDA_PATH")
        .or_else(|_| env::var("CUDA_HOME"))
        .unwrap_or_else(|_| {
            if cfg!(target_os = "windows") {
                "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.0".to_string()
            } else {
                "/usr/local/cuda".to_string()
            }
        });

    let cuda_include = PathBuf::from(&cuda_path).join("include");
    let cuda_lib = if cfg!(target_os = "windows") {
        PathBuf::from(&cuda_path).join("lib/x64")
    } else {
        PathBuf::from(&cuda_path).join("lib64")
    };

    // Compile CUDA kernels using cc with nvcc
    cc::Build::new()
        .cuda(true)
        .cudart("shared")
        .flag("-gencode=arch=compute_70,code=sm_70") // Volta
        .flag("-gencode=arch=compute_75,code=sm_75") // Turing
        .flag("-gencode=arch=compute_80,code=sm_80") // Ampere
        .flag("-gencode=arch=compute_86,code=sm_86") // Ampere (RTX 30xx)
        .flag("-gencode=arch=compute_89,code=sm_89") // Ada Lovelace
        .flag("-gencode=arch=compute_90,code=sm_90") // Hopper
        .flag("--use_fast_math")
        .flag("-O3")
        .include("cuda/include")
        .include(&cuda_include)
        .file("cuda/kernels.cu")
        .compile("fastdl_cuda_kernels");

    // Link CUDA runtime and libraries
    println!("cargo:rustc-link-search=native={}", cuda_lib.display());
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=cublas");
    println!("cargo:rustc-link-lib=dylib=curand");
}
