//! CUDA-vs-CPU parity tests.
//!
//! Several ops (matmul_nt/tn, permute_nd, sum_axis, LayerNorm) have a custom
//! CUDA kernel and a *different* CPU code path. The finite-difference gradcheck
//! in `gradcheck.rs` only exercises the CPU path, so a bug in a GPU kernel
//! (e.g. a wrong cuBLAS transpose flag or leading dimension) would pass there
//! while silently producing wrong results on the GPU.
//!
//! These tests run the exact same op on CPU and CUDA — both forward and
//! backward — and assert the results agree. They only build/run with the
//! `cuda` feature.
//!
//! Run with:  cargo test --test cuda_parity -- --test-threads=1
//!
//! NOTE: the cuBLAS handle is a global (not thread-safe), so a process-wide
//! lock serialises GPU access regardless of the test harness thread count.

#![cfg(feature = "cuda")]

use std::sync::Mutex;
use fastnn::tensor::{Tensor, Device};
use fastnn::autograd::graph;
use fastnn::cuda::CudaContext;
use fastnn::nn::{Module, LayerNorm};

static GPU_LOCK: Mutex<()> = Mutex::new(());

// Deterministic PRNG (SplitMix64) — identical to gradcheck.rs.
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self { Rng(seed.wrapping_add(0x9E3779B97F4A7C15)) }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn unit(&mut self) -> f32 { (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0 }
}
fn rand_data(n: usize, seed: u64) -> Vec<f32> {
    let mut r = Rng::new(seed); (0..n).map(|_| r.unit()).collect()
}

/// cuBLAS GEMM uses TF32 tensor cores (set in fastnn_cuda_init), giving ~1e-3
/// relative error vs f32. Elementwise/gather kernels are exact f32.
fn assert_close(name: &str, what: &str, cpu: &[f32], gpu: &[f32], rtol: f32, atol: f32) {
    assert_eq!(cpu.len(), gpu.len(), "{name}: {what} length mismatch {} vs {}", cpu.len(), gpu.len());
    let mut max_rel = 0.0f32;
    for (i, (&c, &g)) in cpu.iter().zip(gpu.iter()).enumerate() {
        let denom = c.abs().max(g.abs()).max(1.0);
        let rel = (c - g).abs() / denom;
        if rel > max_rel { max_rel = rel; }
        assert!(
            (c - g).abs() <= atol || rel <= rtol,
            "{name}: {what} mismatch at [{i}]\n  cpu = {c:.6}\n  gpu = {g:.6}\n  abs = {:.6}  rel = {:.6}",
            (c - g).abs(), rel
        );
    }
    eprintln!("  ✓ {name:24} {what:8} max_rel = {max_rel:.2e}");
}

/// Run `f` on CPU and on CUDA; compare forward output and all input gradients.
fn check_parity<F>(name: &str, datas: Vec<(Vec<f32>, Vec<usize>)>, rtol: f32, atol: f32, f: F)
where F: Fn(&[Tensor]) -> Tensor {
    // Tolerate poisoning so one failing test doesn't cascade into the rest.
    let _guard = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    CudaContext::new(0).expect("CUDA init");

    // Fixed upstream weights — need output shape first (CPU dry run, no grad).
    graph::disable_grad();
    let dry: Vec<Tensor> = datas.iter().map(|(d, s)| Tensor::from_vec(d.clone(), s)).collect();
    let out_shape = f(&dry).shape().to_vec();
    let w = rand_data(out_shape.iter().product(), 0xBADC0DE ^ name.len() as u64);

    // ── CPU forward + backward ──────────────────────────────────────────────
    graph::enable_grad();
    let cpu_in: Vec<Tensor> = datas.iter().map(|(d, s)| {
        let mut t = Tensor::from_vec(d.clone(), s); t.set_requires_grad(true); t
    }).collect();
    let cpu_out = f(&cpu_in);
    let cpu_out_vec = cpu_out.to_vec();
    let cpu_loss = cpu_out.mul(&Tensor::from_vec(w.clone(), &out_shape)).sum();
    cpu_loss.backward();
    let cpu_grads: Vec<Vec<f32>> = cpu_in.iter().map(|t| t.grad().unwrap().to_vec()).collect();
    graph::disable_grad();

    // ── CUDA forward + backward ───────────────────────────────────────────────
    graph::enable_grad();
    let gpu_in: Vec<Tensor> = datas.iter().map(|(d, s)| {
        let mut t = Tensor::from_vec(d.clone(), s).to_device(Device::Cuda(0));
        t.set_requires_grad(true); t
    }).collect();
    let gpu_out = f(&gpu_in);
    let gpu_out_vec = gpu_out.to_vec();
    let w_gpu = Tensor::from_vec(w.clone(), &out_shape).to_device(Device::Cuda(0));
    let gpu_loss = gpu_out.mul(&w_gpu).sum();
    gpu_loss.backward();
    let gpu_grads: Vec<Vec<f32>> = gpu_in.iter().map(|t| t.grad().unwrap().to_vec()).collect();
    graph::disable_grad();

    assert_close(name, "forward", &cpu_out_vec, &gpu_out_vec, rtol, atol);
    for (k, (cg, gg)) in cpu_grads.iter().zip(gpu_grads.iter()).enumerate() {
        assert_close(name, &format!("grad{k}"), cg, gg, rtol, atol);
    }
}

// TF32 tolerance for GEMM-based ops; tight tolerance for exact f32 kernels.
const TF32_RTOL: f32 = 2e-2;
const EXACT_RTOL: f32 = 1e-4;
const ATOL: f32 = 2e-3;

// ── Matmul family (cuBLAS, TF32) ──────────────────────────────────────────────

#[test]
fn parity_matmul_2d() {
    check_parity("matmul_2d", vec![
        (rand_data(12, 1), vec![3, 4]),
        (rand_data(20, 2), vec![4, 5]),
    ], TF32_RTOL, ATOL, |x| x[0].matmul(&x[1]));
}

#[test]
fn parity_matmul_nt_2d() {
    // The kernel I wrote: C[3,5] = A[3,4] @ B^T, B is [5,4].
    check_parity("matmul_nt_2d", vec![
        (rand_data(12, 1), vec![3, 4]),
        (rand_data(20, 2), vec![5, 4]),
    ], TF32_RTOL, ATOL, |x| x[0].matmul_nt(&x[1]));
}

#[test]
fn parity_matmul_tn_2d() {
    // C[4,5] = A^T @ B, A is [3,4], B is [3,5].
    check_parity("matmul_tn_2d", vec![
        (rand_data(12, 1), vec![3, 4]),
        (rand_data(15, 2), vec![3, 5]),
    ], TF32_RTOL, ATOL, |x| x[0].matmul_tn(&x[1]));
}

#[test]
fn parity_matmul_batched() {
    // [2, 3, 4] @ [2, 4, 5] — the attention-style batched path.
    check_parity("matmul_batched", vec![
        (rand_data(24, 1), vec![2, 3, 4]),
        (rand_data(40, 2), vec![2, 4, 5]),
    ], TF32_RTOL, ATOL, |x| x[0].matmul(&x[1]));
}

#[test]
fn parity_matmul_nt_batched() {
    // [2,3,4] @ B^T where B is [2,5,4] → [2,3,5]. Used in attention backward.
    check_parity("matmul_nt_batch", vec![
        (rand_data(24, 1), vec![2, 3, 4]),
        (rand_data(40, 2), vec![2, 5, 4]),
    ], TF32_RTOL, ATOL, |x| x[0].matmul_nt(&x[1]));
}

#[test]
fn parity_matmul_tn_batched() {
    // A^T where A is [2,4,3], @ B[2,4,5] → [2,3,5]. Used in attention backward.
    check_parity("matmul_tn_batch", vec![
        (rand_data(24, 1), vec![2, 4, 3]),
        (rand_data(40, 2), vec![2, 4, 5]),
    ], TF32_RTOL, ATOL, |x| x[0].matmul_tn(&x[1]));
}

// ── Exact-f32 kernels (permute, expand, sum_axis, transpose, elementwise) ─────

#[test]
fn parity_permute_3d() {
    check_parity("permute_3d", vec![(rand_data(24, 1), vec![2, 3, 4])],
        EXACT_RTOL, ATOL, |x| x[0].permute(&[1, 2, 0]));
}

#[test]
fn parity_permute_4d() {
    // The exact attention permutation [0,2,1,3].
    check_parity("permute_4d", vec![(rand_data(48, 1), vec![2, 4, 3, 2])],
        EXACT_RTOL, ATOL, |x| x[0].permute(&[0, 2, 1, 3]));
}

#[test]
fn parity_expand() {
    check_parity("expand", vec![(rand_data(4, 1), vec![1, 4])],
        EXACT_RTOL, ATOL, |x| x[0].expand(&[3, 4]));
}

#[test]
fn parity_sum_axis() {
    check_parity("sum_axis0", vec![(rand_data(12, 1), vec![3, 4])],
        EXACT_RTOL, ATOL, |x| x[0].sum_axis(0));
    check_parity("sum_axis1", vec![(rand_data(12, 1), vec![3, 4])],
        EXACT_RTOL, ATOL, |x| x[0].sum_axis(1));
}

#[test]
fn parity_transpose() {
    check_parity("transpose", vec![(rand_data(12, 1), vec![3, 4])],
        EXACT_RTOL, ATOL, |x| x[0].transpose());
}

#[test]
fn parity_elementwise() {
    check_parity("add", vec![(rand_data(12,1), vec![3,4]), (rand_data(12,2), vec![3,4])],
        EXACT_RTOL, ATOL, |x| x[0].add(&x[1]));
    check_parity("mul", vec![(rand_data(12,1), vec![3,4]), (rand_data(12,2), vec![3,4])],
        EXACT_RTOL, ATOL, |x| x[0].mul(&x[1]));
}

// ── Activations with CUDA backward fast paths ─────────────────────────────────

#[test]
fn parity_activations() {
    check_parity("relu", vec![(rand_data(12, 1), vec![3, 4])], EXACT_RTOL, ATOL, |x| x[0].relu());
    check_parity("gelu", vec![(rand_data(12, 2), vec![3, 4])], 1e-3, ATOL, |x| x[0].gelu());
    check_parity("sigmoid", vec![(rand_data(12, 3), vec![3, 4])], 1e-3, ATOL, |x| x[0].sigmoid());
    check_parity("softmax", vec![(rand_data(12, 4), vec![3, 4])], 1e-3, ATOL, |x| x[0].softmax());
}

// ── LayerNorm: LayerNormCudaBackward (GPU) vs LayerNormBackward (CPU) ─────────
// Note: this verifies the dL/dx path matches. The CPU path does not compute
// gamma/beta grads, so we only check the input gradient here.

#[test]
fn parity_layernorm_input() {
    check_parity("layernorm", vec![(rand_data(12, 1), vec![3, 4])],
        1e-3, ATOL, |x| LayerNorm::new(&[4]).forward(&x[0]));
}
