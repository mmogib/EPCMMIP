# Optimal-control benchmarks

The two self-contained studies are:

- `double_integrator_control/`
- `harmonic_oscillator/`

Each uses mesh sizes `K=50,100,200` and ten independently seeded starts. The stopping quantity is

`R_n = 0.5 * norm(z_n - J^A_1(z_n - B(z_n)))^2`,

with tolerance `10^-5`, two consecutive hits, and a cap of 4000 iterations. Every production cell has two warm-ups and three measured repetitions; the runner asserts identical flag/iteration/evaluation signatures and records median CPU time.

## Run

From `jcode/`, with Julia and BLAS restricted to one thread:

```powershell
julia --project=. scripts/optimal_control/double_integrator_control/s01_smoke_test.jl
julia --project=. scripts/optimal_control/double_integrator_control/s30_benchmark.jl
julia --project=. scripts/optimal_control/double_integrator_control/s70_tables.jl

julia --project=. scripts/optimal_control/harmonic_oscillator/s01_smoke_test.jl
julia --project=. scripts/optimal_control/harmonic_oscillator/s30_benchmark.jl
julia --project=. scripts/optimal_control/harmonic_oscillator/s70_tables.jl
```

The `s70_tables.jl` files are pure table generators. Mohammed-only plotting entry points are:

```powershell
julia --project=. scripts/optimal_control/double_integrator_control/s70_figures_tables.jl
julia --project=. scripts/optimal_control/harmonic_oscillator/s70_figures_tables.jl
```

Each problem writes `experiments.db`, logs, JSON manifests, tables, and figure destinations below `results/optimal_control/<problem>/`.
