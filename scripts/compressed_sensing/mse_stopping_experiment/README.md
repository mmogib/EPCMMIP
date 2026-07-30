# Compressed sensing with MSE stopping

This is an independent experiment.  It does not read from or write to the
existing `compressed_sensing` experiment database or figures.

For every case `(M, N, K)`, the planted signal has exactly `K` nonzero entries
drawn independently from `Uniform[-2, 2]`.  The sensing matrix has independent
entries `Normal(0, 1)` and observations are

```
y = C * x_star + epsilon,     epsilon ~ Normal(0, 1e-4 I_M).
```

The LASSO model is `min_x 0.5 * ||C*x-y||^2 + gamma * ||x||_1`.  A solve is
declared converged when `MSE(x, x_star) = ||x-x_star||^2 / N` is below the MSE
tolerance for two consecutive iterations.  The maximum-iteration and NaN/Inf
stoppers remain active safeguards.

## Run

From the repository root:

```powershell
julia --project=. scripts/compressed_sensing/mse_stopping_experiment/s30_benchmark.jl --quick
julia --project=. scripts/compressed_sensing/mse_stopping_experiment/s30_benchmark.jl
julia --project=. scripts/compressed_sensing/mse_stopping_experiment/s70_mse_figures.jl --png
```

Results, logs, the SQLite database, and figures are all created under this
directory's `results/` folder.
