use criterion::{criterion_group, criterion_main, Criterion};
use fastnn::tensor::Tensor;

fn bench_matmul(c: &mut Criterion) {
    let a = Tensor::randn(&[128, 256]);
    let b = Tensor::randn(&[256, 128]);

    c.bench_function("matmul_128x256_x_256x128", |bencher| {
        bencher.iter(|| a.matmul(&b))
    });
}

fn bench_matmul_large(c: &mut Criterion) {
    let a = Tensor::randn(&[512, 512]);
    let b = Tensor::randn(&[512, 512]);

    c.bench_function("matmul_512x512", |bencher| {
        bencher.iter(|| a.matmul(&b))
    });
}

fn bench_relu(c: &mut Criterion) {
    let a = Tensor::randn(&[1024, 1024]);

    c.bench_function("relu_1M_elements", |bencher| {
        bencher.iter(|| a.relu())
    });
}

fn bench_softmax(c: &mut Criterion) {
    let a = Tensor::randn(&[64, 1000]);

    c.bench_function("softmax_64x1000", |bencher| {
        bencher.iter(|| a.softmax())
    });
}

fn bench_elementwise(c: &mut Criterion) {
    let a = Tensor::randn(&[1024, 1024]);
    let b = Tensor::randn(&[1024, 1024]);

    c.bench_function("add_1M_elements", |bencher| {
        bencher.iter(|| a.add(&b))
    });

    c.bench_function("mul_1M_elements", |bencher| {
        bencher.iter(|| a.mul(&b))
    });
}

criterion_group!(benches, bench_matmul, bench_matmul_large, bench_relu, bench_softmax, bench_elementwise);
criterion_main!(benches);
