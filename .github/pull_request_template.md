## What this changes

<!-- One or two sentences. What is different after this merges? -->

## Why

<!-- The problem it solves. Link the issue if there is one: Fixes #123 -->

## How it was verified

<!-- Delete what does not apply, keep it honest. -->

- [ ] `cargo test` passes
- [ ] `cargo clippy --all-targets -- -D warnings` is clean
- [ ] Ran `cuda_parity` on a GPU
- [ ] Trained an example in `--release`
- [ ] No GPU here, CUDA side not runtime tested

## Checklist

- [ ] Every new or changed backward rule has a gradcheck case
- [ ] Every new kernel has a cuda_parity case
- [ ] `tensor/ops` and `autograd/ops` still mirror each other
- [ ] Public items have `///` docs
