# EPCMMIP numerical program

Reproducible Julia 1.12.6 experiments for the manuscript's compressed-sensing, optimal-control, and matrix saddle-point studies. The implementation compares AEFBFP with VAFBS, MDITSM, RFBSM, IRFBSM, and IFRAB; the game study also includes a local Halpern forward-backward diagnostic.

## Reproducibility contract

- one Julia thread and one BLAS thread;
- independent named seeds for every random artifact;
- SHA-256 hashes of realized arrays;
- two warm-ups followed by three measured repetitions;
- identical flag/iteration/evaluation signatures required across repetitions;
- content-addressed SQLite rows and JSON run manifests under `results/`.

Set up and test from this directory:

```powershell
julia --project=. -e 'import Pkg; Pkg.instantiate()'
$env:JULIA_NUM_THREADS='1'
julia --project=. test/runtests.jl
julia --project=. test/test_cs_protocol.jl
julia --project=. test/test_oc_protocol.jl
julia --project=. test/test_saddle_point.jl
julia --project=. test/test_saddle_runner.jl
```

## Production workflows

```powershell
# Compressed sensing
julia --project=. scripts/compressed_sensing/s30_benchmark.jl
julia --project=. scripts/compressed_sensing/s70_tables.jl

# Optimal control
julia --project=. scripts/optimal_control/double_integrator_control/s30_benchmark.jl
julia --project=. scripts/optimal_control/double_integrator_control/s70_tables.jl
julia --project=. scripts/optimal_control/harmonic_oscillator/s30_benchmark.jl
julia --project=. scripts/optimal_control/harmonic_oscillator/s70_tables.jl

# Matrix games
julia --project=. scripts/saddle_point/s30_benchmark.jl
julia --project=. scripts/saddle_point/s60_recompute_gaps.jl
julia --project=. scripts/saddle_point/s70_tables.jl
```

The `s70_tables.jl` scripts are plot-free. Plotting dependencies are intentionally imported only by the separate Mohammed-run figure entry points:

- `scripts/compressed_sensing/s70_figures_tables.jl`
- `scripts/optimal_control/double_integrator_control/s70_figures_tables.jl`
- `scripts/optimal_control/harmonic_oscillator/s70_figures_tables.jl`
- `scripts/saddle_point/s71_figures.jl`

## Layout

- `src/`: algorithms, callbacks, projections, database helpers, and reproducibility utilities.
- `scripts/compressed_sensing/`: four row-orthonormal LASSO recovery cases.
- `scripts/optimal_control/`: double-integrator and harmonic-oscillator control problems.
- `scripts/saddle_point/`: three random zero-sum games and one duplicated-identity game with exact anchor diagnostics.
- `test/`: core, protocol, geometry, and runner tests.
- `results/`: runtime databases, logs, manifests, tables, and figure destinations (gitignored).

See `CLAUDE.md` and each script-family README for the exact stopping rules and protocol details.
