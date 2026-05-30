//! Finite-difference gradient checks for the autograd engine (CPU backend).
//!
//! For a scalar loss `L = sum(f(x) * w)` with fixed random weights `w`, the
//! analytic gradient produced by `backward()` must match the central finite
//! difference  `(L(x_i + h) - L(x_i - h)) / 2h`  for every input element `x_i`.
//!
//! Using a *random* weight vector `w` (rather than `f(x).sum()`, which gives a
//! uniform upstream gradient of 1) exercises each `GradFn` with a non-trivial
//! upstream gradient, catching transpose/stride/scale bugs that a uniform
//! gradient would mask.
//!
//! Run with:  cargo test --no-default-features --test gradcheck

use fastnn::tensor::Tensor;
use fastnn::autograd::graph;
use fastnn::nn::{Module, LayerNorm, CrossEntropyLoss, MSELoss};

// ── Deterministic pseudo-random data ────────────────────────────────────────
// A tiny SplitMix64-style generator so tests are reproducible without depending
// on the `rand` crate's behaviour or the library's global RNG state.
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
    /// Uniform f32 in [-1, 1).
    fn unit(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
    }
}

fn rand_data(n: usize, seed: u64) -> Vec<f32> {
    let mut r = Rng::new(seed);
    (0..n).map(|_| r.unit()).collect()
}

/// Strictly-positive data in [0.5, 1.5) — for ops needing positive inputs
/// (log, sqrt, div denominator, non-integer pow).
fn rand_pos(n: usize, seed: u64) -> Vec<f32> {
    let mut r = Rng::new(seed);
    (0..n).map(|_| r.unit() * 0.5 + 1.0).collect()
}

/// Data guaranteed to stay away from 0 by at least `gap`, so finite differences
/// across a kink (relu/abs/leaky_relu) never straddle the non-differentiable
/// point at the chosen step size.
fn rand_away_from_zero(n: usize, seed: u64, gap: f32) -> Vec<f32> {
    let mut r = Rng::new(seed);
    (0..n).map(|_| {
        let v = r.unit();
        if v >= 0.0 { v + gap } else { v - gap }
    }).collect()
}

// ── The harness ──────────────────────────────────────────────────────────────

const H: f32 = 5e-3;     // central-difference step (near-optimal for f32)
const RTOL: f32 = 3e-2;  // relative tolerance
const ATOL: f32 = 1e-2;  // absolute tolerance

/// Run `f` on freshly built input tensors and return the scalar loss
/// `sum(f(inputs) * w)`. Caller controls grad tracking via the global tape.
fn forward_loss<F>(f: &F, datas: &[(Vec<f32>, Vec<usize>)], w: &[f32], req_grad: bool) -> (f32, Vec<Tensor>)
where F: Fn(&[Tensor]) -> Tensor {
    let inputs: Vec<Tensor> = datas.iter().map(|(d, s)| {
        let mut t = Tensor::from_vec(d.clone(), s);
        if req_grad { t.set_requires_grad(true); }
        t
    }).collect();
    let out = f(&inputs);
    let wt = Tensor::from_vec(w.to_vec(), out.shape());
    let loss = out.mul(&wt).sum();
    (loss.item(), { let mut v = inputs; v.push(loss); v })
}

/// Gradient-check `f` at the given inputs. Panics with a detailed message on
/// the first element whose analytic and numerical gradients disagree.
fn gradcheck<F>(name: &str, datas: Vec<(Vec<f32>, Vec<usize>)>, f: F)
where F: Fn(&[Tensor]) -> Tensor {
    // 1) Dry run (no grad) to learn the output shape, then build fixed weights.
    graph::disable_grad();
    let dry_inputs: Vec<Tensor> = datas.iter().map(|(d, s)| Tensor::from_vec(d.clone(), s)).collect();
    let out_shape = f(&dry_inputs).shape().to_vec();
    let out_numel: usize = out_shape.iter().product();
    let w = rand_data(out_numel, 0xC0FFEE ^ name.len() as u64);

    // 2) Analytic gradients via one backward pass.
    graph::enable_grad();
    let inputs: Vec<Tensor> = datas.iter().map(|(d, s)| {
        let mut t = Tensor::from_vec(d.clone(), s);
        t.set_requires_grad(true);
        t
    }).collect();
    let out = f(&inputs);
    let wt = Tensor::from_vec(w.clone(), out.shape());
    let loss = out.mul(&wt).sum();
    loss.backward();
    let analytic: Vec<Vec<f32>> = inputs.iter().enumerate().map(|(k, t)| {
        t.grad().unwrap_or_else(|| panic!("{name}: input {k} has no gradient after backward"))
            .to_vec()
    }).collect();
    graph::disable_grad();

    // 3) Numerical gradient element-by-element, comparing as we go.
    let mut max_rel = 0.0f32;
    for k in 0..datas.len() {
        let (data_k, _shape_k) = &datas[k];
        for i in 0..data_k.len() {
            let mut plus = datas.clone();
            let mut minus = datas.clone();
            plus[k].0[i] += H;
            minus[k].0[i] -= H;
            let (lp, _) = forward_loss(&f, &plus, &w, false);
            let (lm, _) = forward_loss(&f, &minus, &w, false);
            let num = (lp - lm) / (2.0 * H);
            let ana = analytic[k][i];

            let denom = ana.abs().max(num.abs()).max(1.0);
            let rel = (ana - num).abs() / denom;
            if rel > max_rel { max_rel = rel; }

            let abs_ok = (ana - num).abs() <= ATOL;
            let rel_ok = rel <= RTOL;
            assert!(
                abs_ok || rel_ok,
                "{name}: grad mismatch at input {k} elem {i}\n  analytic = {ana:.6}\n  numerical= {num:.6}\n  abs_err  = {:.6}\n  rel_err  = {:.6}",
                (ana - num).abs(), rel
            );
        }
    }
    eprintln!("  ✓ {name:28} max_rel_err = {max_rel:.2e}");
}

// ── Elementwise & scalar ops ──────────────────────────────────────────────────

#[test]
fn check_add() {
    gradcheck("add", vec![
        (rand_data(12, 1), vec![3, 4]),
        (rand_data(12, 2), vec![3, 4]),
    ], |x| x[0].add(&x[1]));
}

#[test]
fn check_add_broadcast() {
    // [3,4] + [1,4] — bias-style broadcasting (exercises ExpandBackward reduce).
    gradcheck("add_broadcast", vec![
        (rand_data(12, 1), vec![3, 4]),
        (rand_data(4, 2), vec![1, 4]),
    ], |x| x[0].add(&x[1].expand(&[3, 4])));
}

#[test]
fn check_sub() {
    gradcheck("sub", vec![
        (rand_data(12, 1), vec![3, 4]),
        (rand_data(12, 2), vec![3, 4]),
    ], |x| x[0].sub(&x[1]));
}

#[test]
fn check_mul() {
    gradcheck("mul", vec![
        (rand_data(12, 1), vec![3, 4]),
        (rand_data(12, 2), vec![3, 4]),
    ], |x| x[0].mul(&x[1]));
}

#[test]
fn check_div() {
    gradcheck("div", vec![
        (rand_data(12, 1), vec![3, 4]),
        (rand_pos(12, 2), vec![3, 4]),  // denominator strictly positive
    ], |x| x[0].div(&x[1]));
}

#[test]
fn check_add_scalar() {
    gradcheck("add_scalar", vec![(rand_data(12, 1), vec![3, 4])], |x| x[0].add_scalar(0.7));
}

#[test]
fn check_mul_scalar() {
    gradcheck("mul_scalar", vec![(rand_data(12, 1), vec![3, 4])], |x| x[0].mul_scalar(-1.5));
}

#[test]
fn check_pow_scalar() {
    gradcheck("pow_scalar", vec![(rand_pos(12, 1), vec![3, 4])], |x| x[0].pow_scalar(2.5));
}

#[test]
fn check_neg() {
    gradcheck("neg", vec![(rand_data(12, 1), vec![3, 4])], |x| x[0].neg());
}

#[test]
fn check_exp() {
    gradcheck("exp", vec![(rand_data(12, 1), vec![3, 4])], |x| x[0].exp());
}

#[test]
fn check_log() {
    gradcheck("log", vec![(rand_pos(12, 1), vec![3, 4])], |x| x[0].log());
}

#[test]
fn check_sqrt() {
    gradcheck("sqrt", vec![(rand_pos(12, 1), vec![3, 4])], |x| x[0].sqrt());
}

#[test]
fn check_abs() {
    gradcheck("abs", vec![(rand_away_from_zero(12, 1, 0.2), vec![3, 4])], |x| x[0].abs());
}

#[test]
fn check_clamp() {
    // Mix of values clearly inside (-0.5,0.5) and clearly outside, all kept
    // away from the boundaries so finite diff never straddles a kink.
    let data = vec![-0.9, -0.7, -0.3, -0.1, 0.0, 0.1, 0.3, 0.2, 0.7, 0.9, 1.2, -1.3];
    gradcheck("clamp", vec![(data, vec![3, 4])], |x| x[0].clamp(-0.5, 0.5));
}

// ── Activations ───────────────────────────────────────────────────────────────

#[test]
fn check_relu() {
    gradcheck("relu", vec![(rand_away_from_zero(12, 1, 0.2), vec![3, 4])], |x| x[0].relu());
}

#[test]
fn check_leaky_relu() {
    gradcheck("leaky_relu", vec![(rand_away_from_zero(12, 1, 0.2), vec![3, 4])], |x| x[0].leaky_relu(0.1));
}

#[test]
fn check_sigmoid() {
    gradcheck("sigmoid", vec![(rand_data(12, 1), vec![3, 4])], |x| x[0].sigmoid());
}

#[test]
fn check_tanh() {
    gradcheck("tanh", vec![(rand_data(12, 1), vec![3, 4])], |x| x[0].tanh_act());
}

#[test]
fn check_gelu() {
    gradcheck("gelu", vec![(rand_data(12, 1), vec![3, 4])], |x| x[0].gelu());
}

#[test]
fn check_silu() {
    gradcheck("silu", vec![(rand_data(12, 1), vec![3, 4])], |x| x[0].silu());
}

// ── Softmax family ────────────────────────────────────────────────────────────

#[test]
fn check_softmax() {
    gradcheck("softmax", vec![(rand_data(12, 1), vec![3, 4])], |x| x[0].softmax());
}

#[test]
fn check_log_softmax() {
    gradcheck("log_softmax", vec![(rand_data(12, 1), vec![3, 4])], |x| x[0].log_softmax());
}

// ── Matmul & transpose ────────────────────────────────────────────────────────

#[test]
fn check_matmul() {
    gradcheck("matmul", vec![
        (rand_data(6, 1), vec![2, 3]),
        (rand_data(15, 2), vec![3, 5]),
    ], |x| x[0].matmul(&x[1]));
}

#[test]
fn check_matmul_nt() {
    // C[2,5] = A[2,3] @ B^T  where B is [5,3].
    gradcheck("matmul_nt", vec![
        (rand_data(6, 1), vec![2, 3]),
        (rand_data(15, 2), vec![5, 3]),
    ], |x| x[0].matmul_nt(&x[1]));
}

#[test]
fn check_matmul_tn() {
    // C[3,5] = A^T @ B  where A is [2,3], B is [2,5].
    gradcheck("matmul_tn", vec![
        (rand_data(6, 1), vec![2, 3]),
        (rand_data(10, 2), vec![2, 5]),
    ], |x| x[0].matmul_tn(&x[1]));
}

#[test]
fn check_transpose() {
    gradcheck("transpose", vec![(rand_data(6, 1), vec![2, 3])], |x| x[0].transpose());
}

#[test]
fn check_matmul_nt_batched() {
    // C[2,3,5] = A[2,3,4] @ B^T, B is [2,5,4]. Ground-truth for the CPU path.
    gradcheck("matmul_nt_batched", vec![
        (rand_data(24, 1), vec![2, 3, 4]),
        (rand_data(40, 2), vec![2, 5, 4]),
    ], |x| x[0].matmul_nt(&x[1]));
}

#[test]
fn check_matmul_tn_batched() {
    // C[2,3,5] = A^T @ B, A is [2,4,3], B is [2,4,5].
    gradcheck("matmul_tn_batched", vec![
        (rand_data(24, 1), vec![2, 4, 3]),
        (rand_data(40, 2), vec![2, 4, 5]),
    ], |x| x[0].matmul_tn(&x[1]));
}

#[test]
fn check_matmul_batched() {
    gradcheck("matmul_batched", vec![
        (rand_data(24, 1), vec![2, 3, 4]),
        (rand_data(40, 2), vec![2, 4, 5]),
    ], |x| x[0].matmul(&x[1]));
}

// ── Reductions ────────────────────────────────────────────────────────────────

#[test]
fn check_sum() {
    gradcheck("sum", vec![(rand_data(12, 1), vec![3, 4])], |x| x[0].sum());
}

#[test]
fn check_mean() {
    gradcheck("mean", vec![(rand_data(12, 1), vec![3, 4])], |x| x[0].mean());
}

#[test]
fn check_sum_axis0() {
    gradcheck("sum_axis0", vec![(rand_data(12, 1), vec![3, 4])], |x| x[0].sum_axis(0));
}

#[test]
fn check_sum_axis1() {
    gradcheck("sum_axis1", vec![(rand_data(12, 1), vec![3, 4])], |x| x[0].sum_axis(1));
}

#[test]
fn check_mean_axis() {
    gradcheck("mean_axis", vec![(rand_data(12, 1), vec![3, 4])], |x| x[0].mean_axis(1));
}

// ── Shape ops ─────────────────────────────────────────────────────────────────

#[test]
fn check_reshape() {
    gradcheck("reshape", vec![(rand_data(12, 1), vec![3, 4])], |x| x[0].reshape(&[4, 3]));
}

#[test]
fn check_permute() {
    gradcheck("permute", vec![(rand_data(24, 1), vec![2, 3, 4])], |x| x[0].permute(&[1, 2, 0]));
}

#[test]
fn check_expand() {
    gradcheck("expand", vec![(rand_data(4, 1), vec![1, 4])], |x| x[0].expand(&[3, 4]));
}

// ── NN-level fused ops ────────────────────────────────────────────────────────

#[test]
fn check_cross_entropy() {
    // Loss is already scalar; harness weights it by a scalar (still valid).
    // [3 batch, 4 classes], one target class per row.
    gradcheck("cross_entropy", vec![(rand_data(12, 1), vec![3, 4])],
        |x| CrossEntropyLoss::new().forward(&x[0], &[0usize, 2, 1]));
}

#[test]
fn check_mse() {
    // Both prediction and target carry gradient here; we only check x[0]'s.
    gradcheck("mse", vec![
        (rand_data(12, 1), vec![3, 4]),
        (rand_data(12, 2), vec![3, 4]),
    ], |x| MSELoss::new().forward(&x[0], &x[1]));
}

#[test]
fn check_layernorm_input() {
    // LayerNorm::new initialises gamma=1, beta=0 deterministically, so forward
    // is a pure normalisation. Checks the dL/dx path (the load-bearing one).
    gradcheck("layernorm_input", vec![(rand_data(12, 1), vec![3, 4])],
        |x| LayerNorm::new(&[4]).forward(&x[0]));
}

// ── A small composite chain (catches accumulation ordering bugs) ──────────────

#[test]
fn check_composite_mlp() {
    // y = gelu(x @ W1) @ W2 ; loss = sum(y * w)
    gradcheck("composite_mlp", vec![
        (rand_data(8, 1), vec![2, 4]),   // x  [2,4]
        (rand_data(12, 2), vec![4, 3]),  // W1 [4,3]
        (rand_data(15, 3), vec![3, 5]),  // W2 [3,5]
    ], |x| x[0].matmul(&x[1]).gelu().matmul(&x[2]));
}
