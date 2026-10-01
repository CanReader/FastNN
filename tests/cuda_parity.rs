//! CPU/GPU equivalence.
//!
//! Every op with a hand-written CUDA kernel is run twice — once on each device —
//! and the results compared. `tests/gradcheck.rs` proves the CPU maths is right;
//! this proves the GPU agrees with it, forward and backward.
//!
//!     cargo test --test cuda_parity
//!
//! Skips itself with a note when no GPU is present, so a CPU-only machine still
//! gets a green run.

// Test inputs are always passed as a slice. `&[x.clone()]` for a single input
// costs one clone of a refcounted handle and keeps every call site reading the
// same way as the multi-input ones.
#![allow(clippy::cloned_ref_to_slice_refs)]

use std::sync::Mutex;

use fastnn::nn::Param;
use fastnn::prelude::*;

/// Difference a single op may show. The kernels are built with
/// `--use_fast_math` and reduce in a different order than the host loops, so
/// exact equality is not on offer; anything past this is a real disagreement.
const OP_TOLERANCE: f32 = 2e-4;

/// Difference a whole block may show. Dozens of ops compound their rounding, so
/// end-to-end comparisons get more room than single ops.
const MODEL_TOLERANCE: f32 = 5e-3;

/// Tests share one GPU and one cuBLAS handle, so they run one at a time.
static GPU: Mutex<()> = Mutex::new(());

/// Take the GPU lock, ignoring poison.
///
/// A failing assertion poisons the mutex, and without this every later test
/// reports that poison instead of its own result — hiding the failure that
/// actually matters behind a wall of noise.
fn lock_gpu() -> std::sync::MutexGuard<'static, ()> {
    GPU.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// The GPU, or `None` on a machine or build without one.
fn gpu() -> Option<Device> {
    match Device::cuda(0) {
        Ok(device) => Some(device),
        Err(_) => {
            eprintln!("no CUDA device — skipping parity test");
            None
        }
    }
}

fn spread(n: usize, low: f32, high: f32) -> Vec<f32> {
    const GOLDEN: f32 = 0.618_034;
    (0..n)
        .map(|i| low + (high - low) * ((i as f32 + 1.0) * GOLDEN).fract())
        .collect()
}

fn sample(shape: &[usize]) -> Tensor {
    Tensor::from_vec(spread(shape.iter().product(), -0.9, 0.9), shape)
}

#[track_caller]
fn assert_close(name: &str, cpu: &[f32], cuda: &[f32], tolerance: f32) {
    assert_eq!(
        cpu.len(),
        cuda.len(),
        "{name}: {} values on CPU, {} on GPU",
        cpu.len(),
        cuda.len()
    );
    for (i, (&want, &got)) in cpu.iter().zip(cuda).enumerate() {
        let scale = 1.0f32.max(want.abs());
        assert!(
            (want - got).abs() / scale < tolerance,
            "{name}: element {i} is {want} on CPU but {got} on GPU"
        );
    }
}

/// Run `f` on both devices and compare its output.
#[track_caller]
fn check_forward(name: &str, inputs: &[Tensor], f: impl Fn(&[Tensor]) -> Tensor) {
    let Some(device) = gpu() else { return };
    let _guard = lock_gpu();

    let on_cpu = f(inputs).to_vec();
    let on_gpu: Vec<Tensor> = inputs.iter().map(|t| t.to(device)).collect();
    let on_gpu = f(&on_gpu).cpu().to_vec();

    assert_close(name, &on_cpu, &on_gpu, OP_TOLERANCE);
}

/// Run `f` on both devices and compare every input gradient.
#[track_caller]
fn check_backward(name: &str, inputs: &[Tensor], f: impl Fn(&[Tensor]) -> Tensor) {
    let Some(device) = gpu() else { return };
    let _guard = lock_gpu();

    let gradients = |placed: Vec<Tensor>| -> Vec<Vec<f32>> {
        let params: Vec<Param> = placed.into_iter().map(Param::new).collect();
        let loss = f(&params.iter().map(|p| p.tensor()).collect::<Vec<_>>());
        loss.backward();
        params
            .iter()
            .map(|p| p.grad().expect("input received no gradient").cpu().to_vec())
            .collect()
    };

    let on_cpu = gradients(inputs.to_vec());
    let on_gpu = gradients(inputs.iter().map(|t| t.to(device)).collect());

    for (index, (cpu, cuda)) in on_cpu.iter().zip(&on_gpu).enumerate() {
        assert_close(&format!("{name} grad[{index}]"), cpu, cuda, OP_TOLERANCE);
    }
}

// ── Element-wise and activations ─────────────────────────────────────────────

#[test]
fn elementwise() {
    let a = sample(&[4, 16]);
    let b = sample(&[4, 16]);

    check_forward("add", &[a.clone(), b.clone()], |x| x[0].add(&x[1]));
    check_forward("mul", &[a.clone(), b.clone()], |x| x[0].mul(&x[1]));
    check_forward("exp", &[a.clone()], |x| x[0].exp());
    check_forward("clamp", &[a.clone()], |x| x[0].clamp(-0.5, 0.5));
    check_backward("mul", &[a, b], |x| x[0].mul(&x[1]).sum());
}

#[test]
fn broadcasting_stays_on_device() {
    // The GPU path materialises the expansion rather than falling back to the
    // host; the result must still match the CPU's stride walk.
    check_forward("broadcast add", &[sample(&[4, 8]), sample(&[1, 8])], |x| {
        x[0].add(&x[1])
    });
    check_forward("broadcast rank", &[sample(&[4, 8]), sample(&[8])], |x| {
        x[0].mul(&x[1])
    });
}

#[test]
fn activations() {
    let x = sample(&[8, 32]);

    for (name, f) in activation_cases() {
        check_forward(name, &[x.clone()], f);
        check_backward(name, &[x.clone()], |i| f(i).sum());
    }
}

/// A named activation, as a plain function pointer so the list can be iterated.
type Activation = fn(&[Tensor]) -> Tensor;

fn activation_cases() -> Vec<(&'static str, Activation)> {
    vec![
        ("relu", |x| x[0].relu()),
        ("sigmoid", |x| x[0].sigmoid()),
        ("tanh", |x| x[0].tanh()),
        ("gelu", |x| x[0].gelu()),
        ("silu", |x| x[0].silu()),
        ("leaky_relu", |x| x[0].leaky_relu(0.1)),
    ]
}

#[test]
fn softmax() {
    let x = sample(&[8, 16]);
    let weights = sample(&[8, 16]);

    check_forward("softmax", &[x.clone()], |x| x[0].softmax());
    check_forward("log_softmax", &[x.clone()], |x| x[0].log_softmax());

    // Weighted so the gradient is non-trivial; a plain sum over softmax is
    // constant and its gradient is zero on both devices whatever the kernel does.
    check_backward("softmax", &[x, weights], |x| {
        x[0].softmax().mul(&x[1]).sum()
    });
}

// ── Matrix multiplication ────────────────────────────────────────────────────

#[test]
fn gemm_layouts() {
    check_forward("matmul", &[sample(&[8, 12]), sample(&[12, 6])], |x| {
        x[0].matmul(&x[1])
    });
    check_forward("matmul_nt", &[sample(&[8, 12]), sample(&[6, 12])], |x| {
        x[0].matmul_nt(&x[1])
    });
    check_forward("matmul_tn", &[sample(&[12, 8]), sample(&[12, 6])], |x| {
        x[0].matmul_tn(&x[1])
    });

    check_backward("matmul", &[sample(&[8, 12]), sample(&[12, 6])], |x| {
        x[0].matmul(&x[1]).sum()
    });
    check_backward("matmul_nt", &[sample(&[8, 12]), sample(&[6, 12])], |x| {
        x[0].matmul_nt(&x[1]).sum()
    });
}

#[test]
fn gemm_batched() {
    check_forward(
        "batched",
        &[sample(&[4, 8, 12]), sample(&[4, 12, 6])],
        |x| x[0].matmul(&x[1]),
    );
    check_forward(
        "batched nt",
        &[sample(&[4, 8, 12]), sample(&[4, 6, 12])],
        |x| x[0].matmul_nt(&x[1]),
    );
    // One side batched, the other shared — how a weight meets a batch of inputs.
    check_forward(
        "shared weight",
        &[sample(&[8, 12]), sample(&[4, 12, 6])],
        |x| x[0].matmul(&x[1]),
    );
}

// ── Layout ───────────────────────────────────────────────────────────────────

#[test]
fn layout_ops() {
    let x = sample(&[2, 3, 4]);
    let weights = sample(&[4, 3, 2]);

    check_forward("permute", &[x.clone()], |x| x[0].permute(&[2, 1, 0]));
    check_forward("transpose", &[x.clone()], |x| x[0].transpose());
    check_forward("expand", &[sample(&[1, 6])], |x| x[0].expand(&[4, 6]));
    check_backward("permute", &[x, weights], |x| {
        x[0].permute(&[2, 1, 0]).mul(&x[1]).sum()
    });
}

// ── Reductions ───────────────────────────────────────────────────────────────

#[test]
fn reductions() {
    let x = sample(&[6, 10]);

    check_forward("sum", &[x.clone()], |x| x[0].sum());
    check_forward("mean", &[x.clone()], |x| x[0].mean());
    check_forward("sum_axis 0", &[x.clone()], |x| x[0].sum_axis(0));
    check_forward("sum_axis 1", &[x.clone()], |x| x[0].sum_axis(1));
    check_backward("sum_axis", &[x, sample(&[10])], |x| {
        x[0].sum_axis(0).mul(&x[1]).sum()
    });
}

// ── Fused layers ─────────────────────────────────────────────────────────────

#[test]
fn layer_norm() {
    let x = sample(&[6, 16]);
    let gamma = Tensor::full(&[16], 1.1);
    let beta = Tensor::full(&[16], -0.2);

    check_forward(
        "layer_norm",
        &[x.clone(), gamma.clone(), beta.clone()],
        |x| x[0].layer_norm(&x[1], &x[2], 1e-5),
    );
    // The GPU kernel produces all three gradients in one pass; the CPU path
    // computes them separately. They must agree.
    //
    // The weights come in as a fourth input rather than being built inside the
    // closure: a tensor made in there would always be on the host and would not
    // survive the GPU run.
    check_backward("layer_norm", &[x, gamma, beta, sample(&[6, 16])], |x| {
        x[0].layer_norm(&x[1], &x[2], 1e-5).mul(&x[3]).sum()
    });
}

#[test]
fn row_reductions_handle_awkward_widths() {
    // The block reductions behind softmax and layer norm halve the thread count
    // each round, so a width that is not a power of two used to lose its tail.
    // 5 and 10 both exercise that; 10 is MNIST's class count.
    for width in [3usize, 5, 7, 10, 17, 31, 100, 257] {
        let x = sample(&[4, width]);
        check_forward(&format!("softmax width {width}"), &[x.clone()], |x| {
            x[0].softmax()
        });
        check_forward(&format!("log_softmax width {width}"), &[x.clone()], |x| {
            x[0].log_softmax()
        });
        check_forward(
            &format!("layer_norm width {width}"),
            &[x, Tensor::full(&[width], 1.1), Tensor::full(&[width], -0.2)],
            |x| x[0].layer_norm(&x[1], &x[2], 1e-5),
        );
    }
}

#[test]
fn attention_matches_across_sequence_lengths() {
    let Some(device) = gpu() else { return };
    let _guard = lock_gpu();

    // Attention softmaxes over the key axis, so the sequence length *is* the
    // reduction width — the exact place the power-of-two bug showed up.
    for length in [1usize, 3, 5, 8, 13] {
        manual_seed(5);
        let attention = MultiHeadAttention::new(16, 4, 0.0);
        let x = sample(&[2, length, 16]);

        let on_cpu = attention.attend(&x, &x, &x, true).to_vec();
        attention.to_device(device);
        let moved = x.to(device);
        let on_gpu = attention
            .attend(&moved, &moved, &moved, true)
            .cpu()
            .to_vec();

        assert_close(
            &format!("attention over {length}"),
            &on_cpu,
            &on_gpu,
            MODEL_TOLERANCE,
        );
    }
}

#[test]
fn embedding_lookup() {
    let table = sample(&[10, 8]);
    let weights = sample(&[5, 8]);

    check_forward("index_select", &[table.clone()], |x| {
        x[0].index_select(&[0, 3, 3, 9, 1])
    });
    // Row 3 twice: the GPU scatter-add must accumulate, not overwrite.
    check_backward("index_select", &[table, weights], |x| {
        x[0].index_select(&[0, 3, 3, 9, 1]).mul(&x[1]).sum()
    });
}

// ── End to end ───────────────────────────────────────────────────────────────

#[test]
fn a_transformer_block_matches() {
    let Some(device) = gpu() else { return };
    let _guard = lock_gpu();

    manual_seed(11);
    let block = TransformerBlock::causal(16, 4, 32, 0.0);
    block.eval();
    let x = sample(&[2, 5, 16]);

    let on_cpu = block.forward(&x).to_vec();
    block.to_device(device);
    let on_gpu = block.forward(&x.to(device)).cpu().to_vec();

    assert_close("transformer block", &on_cpu, &on_gpu, MODEL_TOLERANCE);
}

#[test]
fn a_model_trains_identically_on_both_devices() {
    let Some(device) = gpu() else { return };
    let _guard = lock_gpu();

    let train = |device: Device| {
        manual_seed(3);
        let model = Sequential::new()
            .add(Linear::new(8, 12))
            .add(ReLU)
            .add(Linear::new(12, 4));
        model.to_device(device);

        let mut opt = SGD::new(model.parameters(), 0.1);
        let inputs = sample(&[6, 8]).to(device);
        let targets = [0usize, 1, 2, 3, 1, 0];

        for _ in 0..10 {
            let loss = cross_entropy(&model.forward(&inputs), &targets);
            opt.zero_grad();
            loss.backward();
            opt.step();
        }
        cross_entropy(&model.forward(&inputs), &targets).item()
    };

    let on_cpu = train(Device::Cpu);
    let on_gpu = train(device);
    assert_close("final loss", &[on_cpu], &[on_gpu], MODEL_TOLERANCE);
}

#[test]
fn convolution_lowering() {
    // Stride 2 with padding 1 exercises both the padding zeros and the
    // windows-overlap accumulation in col2im.
    let x = sample(&[2, 3, 6, 5]);
    let window = Window::square(3, 2, 1);

    check_forward("im2col", &[x.clone()], |v| v[0].im2col(window));
    check_backward("im2col", &[x.clone()], |v| {
        let cols = v[0].im2col(window);
        let weight =
            Tensor::from_vec(spread(cols.numel(), -0.9, 0.9), cols.shape()).to(cols.device());
        cols.mul(&weight).sum()
    });
}

#[test]
fn dilated_convolution_lowering() {
    let x = sample(&[1, 2, 6, 6]);
    let window = Window::square(2, 1, 1).dilated(2);

    check_forward("dilated im2col", &[x.clone()], |v| v[0].im2col(window));
    check_backward("dilated col2im op", &[sample(&[1, 8, 4])], |v| {
        let folded = v[0].col2im(Window::square(2, 2, 0), (4, 4));
        let weight =
            Tensor::from_vec(spread(folded.numel(), -0.9, 0.9), folded.shape()).to(folded.device());
        folded.mul(&weight).sum()
    });
}
