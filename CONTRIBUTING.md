# Contributing

Thanks for taking a look. FastNN is meant to be read and edited, so every file
is one idea and none of them are long. Most of what follows is about keeping it
that way.

If you want to work on something bigger than a bug fix, open an issue first so
we can agree on the shape before you write it. Issues labelled
`good first issue` are a decent place to start.

## Setup

You need Rust 1.87 or newer. The default build is CPU only and needs nothing
else.

```bash
cargo test                                   # everything, including doctests
cargo clippy --all-targets -- -D warnings
cargo fmt
```

For the GPU path you also need the CUDA toolkit:

```bash
cargo build --features cuda
cargo test --features cuda --test cuda_parity -- --test-threads=1
```

If `nvcc` rejects your system compiler, set `FASTNN_NVCC_CCBIN` to one it
accepts. Use `--release` for anything that trains, debug is around 50x slower.

## Pull requests

1. Fork, then branch off `master`. Name the branch after what it does,
   `fix/batchnorm-variance` or `feat/conv3d`.
2. Keep one change per PR. A bug fix and a refactor are two PRs.
3. Fill in the template. "How it was verified" matters more than anything else
   in there, so be specific.
4. CI has to be green before merge. PRs are squash merged, so the PR title
   becomes the commit on `master`. Keep it short, lowercase, and imperative:
   `fix softmax for widths that are not a power of two`.

`master` is protected. Nothing lands on it without a PR and a passing CI run,
including my own changes.

## What CI checks

| Job | What it runs |
|---|---|
| Format | `cargo fmt --check` |
| Clippy | `cargo clippy --all-targets -- -D warnings` |
| Test | `cargo test` on Linux, macOS, and Windows |
| MSRV | `cargo check` on Rust 1.87 |
| Docs | `cargo doc` with warnings as errors |
| Package | `cargo package`, so the published crate builds |
| CUDA build | compiles the kernels with nvcc and runs clippy with `--features cuda` |

Hosted runners have no GPU. The CUDA job proves the kernels compile and link,
but nothing in CI runs them. If you touch `cuda/`, run `cuda_parity` locally
and paste the result in the PR. If you have no GPU, say so and I'll run it.

A nightly job reruns the suite in release on stable and beta and trains the
small examples, so problems that only show up with optimizations still get
caught.

## Code conventions

**Comments.** `///` on public items says what the thing is for. Inline comments
explain *why*, never *what*. If a comment restates the code, delete it.

**Module docs.** Every `mod.rs` opens with a `//!` block telling a new reader
what lives there and how the pieces relate.

**Errors.** Shape and device mistakes panic, with a message naming both shapes.
`Result` is for I/O, downloads, and CUDA. Don't add `Result` to ordinary tensor
ops; the fallible surface is `tensor/checked.rs`, which validates and then calls
the panicking op rather than reimplementing it.

**Naming.** Methods say what they produce (`matmul_nt`, `sum_axis_keep`), not
how. Backward rules are `<Op>Backward`.

**Unsafe.** Raw pointers live in `cuda/ffi.rs` and `cuda/kernels.rs` and nowhere
else.

## Adding an op

`tensor/ops/` and `autograd/ops/` mirror each other file for file. The forward
for an activation is in `tensor/ops/activation.rs` and its derivative is in
`autograd/ops/activation.rs`.

1. Forward in `tensor/ops/<category>.rs`, attached with `with_grad`.
2. Rule in `autograd/ops/<category>.rs`, returning gradients in input order.
3. Save **detached** tensors in the rule. A saved value that still carries its
   own node pins the graph that produced it.
4. Add a case to `tests/gradcheck.rs`. A rule without one is not finished.
5. If it has a CUDA kernel, declare it in `cuda/ffi.rs`, wrap it in
   `cuda/kernels.rs`, and add a case to `tests/cuda_parity.rs`.

The README has a full worked example under "Extending it".

## Writing a gradcheck case

Gradcheck compares each backward rule against central finite differences of its
own forward. Two things trip people up:

* **Keep values bounded.** Finite differences subtract two nearby losses. If the
  loss is in the thousands, `f32` loses most of its digits to cancellation and
  the test fails on its own arithmetic. The `spread` helper keeps inputs in
  `(-0.9, 0.9)` for this reason.
* **Weight the output when a plain sum is blind.** `softmax(x).sum()` is 1 for
  any `x`, so its gradient is zero and the check passes against any rule at all.
  Permutations have the same problem. Multiply by a fixed weight tensor first.

## CUDA notes

Kernels build with `--use_fast_math`, so GPU and CPU differ in the last few
digits. Parity tolerance is `2e-4` for a single op and `5e-3` for a whole block.

Block reductions need a power-of-two thread count, because the tree halves
`blockDim.x` every round. Use `reduction_threads()`, which rounds up. Getting
this wrong silently drops the tail of every row, and it already happened once
with softmax on MNIST's ten classes.

## Releases

Releases are cut from `master` by the maintainer. Bump the version in
`Cargo.toml` through a normal PR, then push a matching tag:

```bash
git tag v0.4.0
git push origin v0.4.0
```

The release workflow checks the tag against `Cargo.toml`, runs the suite in
release, builds the example programs for Linux (x86_64, ARM64, and x86_64 with
CUDA), macOS (Intel and Apple Silicon), and Windows, publishes to crates.io, and
creates the GitHub release with the archives, a `SHA256SUMS` file, and notes
from the merged PRs. Release notes are grouped by PR label, so label your PRs.

Any PR that touches `release.yml` runs the whole thing except the publish
steps, so the pipeline gets tested before a real tag depends on it.

## License

By contributing you agree that your work is released under the MIT license,
same as the rest of the project.
