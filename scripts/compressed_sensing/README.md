# Compressed-sensing benchmark

Definitive manuscript regeneration for four LASSO recovery cases:

| Case | Measurements `M` | Signal `N` | Sparsity `k` |
|---|---:|---:|---:|
| 1 | 256 | 512 | 30 |
| 2 | 256 | 512 | 50 |
| 3 | 512 | 1024 | 50 |
| 4 | 512 | 1024 | 80 |

For each case, a Gaussian `N x M` draw is orthonormalized by thin QR and transposed to form `C`, giving `C*C' = I_M`. The planted support and signs, Gaussian noise with standard deviation `0.01`, and each of ten `0.1*randn` starts use independent published seeds. The LASSO weight is `lambda=10^-3`.

The manuscript stopping rule is successive displacement `norm(x_n-x_{n-1}) <= 10^-5` at two consecutive iterations, with a cap of 5000. The fixed-point residual in the accuracy table is a common terminal diagnostic, not the stopping rule.

## Run

From `jcode/`, with Julia and BLAS restricted to one thread:

```powershell
julia --project=. scripts/compressed_sensing/s01_smoke_test.jl
julia --project=. scripts/compressed_sensing/s30_benchmark.jl
julia --project=. scripts/compressed_sensing/s70_tables.jl
```

`s30` performs two warm-ups and three measured repetitions per start and asserts matching flag/iteration/evaluation signatures. `s70_tables.jl` writes the two performance tables and the separate accuracy table without loading a plotting package.

Mohammed-only plotting entry point (do not use in numerical validation):

```powershell
julia --project=. scripts/compressed_sensing/s70_figures_tables.jl
```

Results are stored under `results/compressed_sensing/`: `experiments.db`, `logs/`, `manifests/`, `tables/`, and `figures/`.
