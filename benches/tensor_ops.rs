//! Throughput benchmarks for the hot paths.
//!
//!     cargo bench                         # CPU
//!     cargo bench --features cuda         # CUDA, if the toolkit is present
//!
//! Benchmarks run on whichever device is available, so the same command measures
//! either backend. GPU timings include the launch, not just the kernel.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use fastnn::prelude::*;

/// The device under test, named for the report.
fn device() -> (Device, &'static str) {
    match Device::cuda(0) {
        Ok(device) => (device, "cuda"),
        Err(_) => (Device::Cpu, "cpu"),
    }
}

fn matmul(c: &mut Criterion) {
    let (device, name) = device();
    let mut group = c.benchmark_group(format!("matmul/{name}"));

    for size in [128usize, 512, 1024] {
        let a = Tensor::randn(&[size, size]).to(device);
        let b = Tensor::randn(&[size, size]).to(device);

        // Two flops per multiply-accumulate.
        group.throughput(Throughput::Elements(2 * (size * size * size) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |bench, _| {
            bench.iter(|| a.matmul(&b))
        });
    }
    group.finish();
}

/// The transpose-fused forms, which the backward pass of every linear layer
/// leans on. They should track plain matmul closely — if they do not, a
/// transposed copy is being materialised somewhere it should not be.
fn matmul_fused(c: &mut Criterion) {
    let (device, name) = device();
    let mut group = c.benchmark_group(format!("matmul_fused/{name}"));

    let a = Tensor::randn(&[512, 512]).to(device);
    let b = Tensor::randn(&[512, 512]).to(device);

    group.bench_function("plain", |bench| bench.iter(|| a.matmul(&b)));
    group.bench_function("nt", |bench| bench.iter(|| a.matmul_nt(&b)));
    group.bench_function("tn", |bench| bench.iter(|| a.matmul_tn(&b)));
    group.bench_function("transpose_then_matmul", |bench| {
        bench.iter(|| a.matmul(&b.transpose()))
    });
    group.finish();
}

fn elementwise(c: &mut Criterion) {
    let (device, name) = device();
    let mut group = c.benchmark_group(format!("elementwise/{name}"));

    let a = Tensor::randn(&[1024, 1024]).to(device);
    let b = Tensor::randn(&[1024, 1024]).to(device);
    group.throughput(Throughput::Elements(1024 * 1024));

    group.bench_function("add", |bench| bench.iter(|| a.add(&b)));
    group.bench_function("mul", |bench| bench.iter(|| a.mul(&b)));
    group.bench_function("relu", |bench| bench.iter(|| a.relu()));
    group.bench_function("gelu", |bench| bench.iter(|| a.gelu()));
    group.finish();
}

fn reductions(c: &mut Criterion) {
    let (device, name) = device();
    let mut group = c.benchmark_group(format!("reduce/{name}"));

    let x = Tensor::randn(&[1024, 1024]).to(device);
    group.throughput(Throughput::Elements(1024 * 1024));

    group.bench_function("sum", |bench| bench.iter(|| x.sum()));
    group.bench_function("sum_axis_0", |bench| bench.iter(|| x.sum_axis(0)));
    group.bench_function("sum_axis_1", |bench| bench.iter(|| x.sum_axis(1)));
    group.bench_function("softmax", |bench| bench.iter(|| x.softmax()));
    group.finish();
}

/// Rearranging is pure memory traffic, and it sits in every attention forward.
fn layout(c: &mut Criterion) {
    let (device, name) = device();
    let mut group = c.benchmark_group(format!("layout/{name}"));

    let heads = Tensor::randn(&[32, 8, 128, 64]).to(device);
    group.bench_function("permute_heads", |bench| {
        bench.iter(|| heads.permute(&[0, 2, 1, 3]))
    });

    let row = Tensor::randn(&[1, 512]).to(device);
    group.bench_function("expand", |bench| bench.iter(|| row.expand(&[1024, 512])));
    group.finish();
}

/// A whole training step, which is what actually matters.
fn training_step(c: &mut Criterion) {
    let (device, name) = device();
    let mut group = c.benchmark_group(format!("step/{name}"));

    let model = Sequential::new()
        .add(Linear::new(784, 256))
        .add(ReLU)
        .add(Linear::new(256, 10));
    model.to_device(device);

    let mut opt = Adam::new(model.parameters(), 1e-3);
    let inputs = Tensor::randn(&[128, 784]).to(device);
    let targets: Vec<usize> = (0..128).map(|i| i % 10).collect();

    group.bench_function("mlp_batch128", |bench| {
        bench.iter(|| {
            let loss = cross_entropy(&model.forward(&inputs), &targets);
            opt.zero_grad();
            loss.backward();
            opt.step();
        })
    });

    let block = TransformerBlock::causal(256, 8, 1024, 0.0);
    block.to_device(device);
    block.eval();
    let sequence = Tensor::randn(&[8, 128, 256]).to(device);

    group.bench_function("transformer_block_forward", |bench| {
        bench.iter(|| no_grad(|| block.forward(&sequence)))
    });
    group.finish();
}

criterion_group!(
    benches,
    matmul,
    matmul_fused,
    elementwise,
    reductions,
    layout,
    training_step
);
criterion_main!(benches);
