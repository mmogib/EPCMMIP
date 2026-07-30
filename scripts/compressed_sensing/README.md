# Compressed Sensing Benchmark

This folder now follows the same script structure used in `scripts/optimal_control`.

## Canonical workflow

- `s01_smoke_test.jl`
  - quick sanity check
  - one case, one dataset, two methods
  - writes only non-production rows
- `s20_aefbfp_parameter_search.jl`
  - tunes one AEFBFP parameter set for the full compressed-sensing family
  - writes the winning configuration into the local `tuned_winners` table
- `s30_benchmark.jl`
  - main compressed-sensing benchmark
  - writes to `results/compressed_sensing/experiments.db`
  - stores representative history rows for later figures
- `s70_figures_tables.jl`
  - writes `results/compressed_sensing/figures/tables.tex`
  - can also generate selected figures through `--figures=...`

## Benchmark policy

Compressed sensing uses the same method family requested for the optimal-control comparison:

- `AEFBFP`
- `IFRAB`
- `VAFBS`
- `IMTTM`
- `MTTM`
- `SFRBM`

Full source citations for the five baseline competitors:

- `IFRAB`: Chinedu Izuchukwu, Simeon Reich, Yekini Shehu, and Adeolu Taiwo, "Strong Convergence of Forward-Reflected-Backward Splitting Methods for Solving Monotone Inclusions with Applications to Image Restoration and Optimal Control," `Journal of Scientific Computing` 94(3), Article 73 (2023). https://doi.org/10.1007/s10915-023-02132-6
- `VAFBS`: Duong Viet Thong and Prasit Cholamjiak, "Strong convergence of a forward-backward splitting method with a new step size for solving monotone inclusions," `Computational and Applied Mathematics` 38, Article 94 (2019). https://doi.org/10.1007/s40314-019-0855-z
- `IMTTM`: Bing Tan and Sun Young Cho, "Strong convergence of inertial forward-backward methods for solving monotone inclusions," `Applicable Analysis` 101(15), 5386-5414 (2022). https://doi.org/10.1080/00036811.2021.1892080
- `MTTM`: Aviv Gibali and Duong Viet Thong, "Tseng type methods for solving inclusion problems and its applications," `Calcolo` 55, Article 49 (2018). https://doi.org/10.1007/s10092-018-0292-1
- `SFRBM`: Yonghong Yao, Abubakar Adamu, and Yekini Shehu, "Forward-Reflected-Backward Splitting Algorithms with Momentum: Weak, Linear and Strong Convergence Results," `Journal of Optimization Theory and Applications` 201(3), 1364-1397 (2024). https://doi.org/10.1007/s10957-024-02410-9

Preset policy:

- `AEFBFP` uses the locally tuned compressed-sensing winner written by `s20_aefbfp_parameter_search.jl`
- `IFRAB`, `VAFBS`, `IMTTM`, `MTTM`, `SFRBM` use their source-paper `:paper` presets

## Cases

The benchmark cases are:

- `(M, N, k) = (256, 512, 30)`
- `(256, 512, 50)`
- `(512, 1024, 50)`
- `(512, 1024, 80)`

These are stored in the DB under the case IDs:

- `M256_N512_k30`
- `M256_N512_k50`
- `M512_N1024_k50`
- `M512_N1024_k80`

## Model and stopping rule

Each case solves the LASSO-type inclusion

- `0 in A(x) + B(x)`

with

- `A = gamma * partial ||x||_1`
- `B(x) = C' * (C*x - y)`

All methods use the common (1/L)-scaled LASSO fixed-point residual

- `||x - prox_{(gamma/L)*||.||_1}(x - (1/L)*C' * (C*x - y))||`

for stopping and final-accuracy reporting, with (L=||C||_2^2). This is a
measurement/stopping convention; AEFBFP itself does not receive (L).

The default benchmark tolerance is `1e-5`, with `consec=2`.

## Results layout

The new scripts write to:

- `results/compressed_sensing/experiments.db`
- `results/compressed_sensing/logs/`
- `results/compressed_sensing/figures/`
