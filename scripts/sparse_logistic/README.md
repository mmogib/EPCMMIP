# Sparse Logistic Benchmark

This folder follows the same benchmark-script pattern used in
`scripts/compressed_sensing`, but for the sparse `l1`-regularized logistic
regression problem.

## Canonical workflow

- `s00_problem_smoke.jl`
  - lightweight problem-layer smoke test
  - no DB writes, no logs
- `s01_smoke_test.jl`
  - quick DB-backed benchmark smoke test
  - one case, one dataset, two methods (`AEFBFP`, `IFRAB`)
  - writes only non-production rows
- `s30_benchmark.jl`
  - main sparse-logistic benchmark
  - writes to `results/sparse_logistic/experiments.db`
  - stores representative history rows for dataset 1, init 1

## Model

Each case solves the monotone inclusion associated with

- `min_x sum_i log(1 + exp(-b_i a_i' x)) + rho * ||x||_1`

with

- `A = partial (rho * ||x||_1)`
- `B(x) = K' * sigmoid(Kx)`, where `K = -Diag(b) * X`

The native stopping quantity used here is the fixed-point residual

- `||x - J^A_1(x - B(x))||`

## Cases

Default benchmark cases are:

- `m128_n64`
- `m256_n128`
- `m512_n256`
- `m1024_n512`

where `m` is the feature dimension and `N = m/2` is the sample count.

## Results layout

The scripts write to:

- `results/sparse_logistic/experiments.db`
- `results/sparse_logistic/logs/`
- `results/sparse_logistic/figures/`
