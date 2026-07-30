# Optimal Control Benchmarks

This folder now mirrors the compressed-sensing workflow.

## Per-problem folders

- `double_integrator_control/`
- `harmonic_oscillator/`

Each folder is self-contained and contains its own:

- problem builder
- local benchmark/report helpers
- local benchmark DB helpers
- AEFBFP tuning support
- benchmark runner
- figure and table builder
- `s01_smoke_test.jl`
- `s20_aefbfp_parameter_search.jl`
- `s30_benchmark.jl`
- `s70_figures_tables.jl`

Current layout:

- `double_integrator_control/`: standalone `s01`, `s20`, `s30`, and `s70` files
- `harmonic_oscillator/`: standalone `s01`, `s20`, `s30`, and `s70` files

The `s20` files are informational only in the optimal-control folders:

- no local AEFBFP tuning is run here
- AEFBFP uses the fixed preset chosen for the optimal-control benchmarks
- the actual paper workflow is `s01 -> s30 -> s70`

## Canonical benchmark design

- methods: `AEFBFP`, `VAFBS`, `MDITSM`, `RFBSM`, `IRFBSM`, `IFRAB`
- `AEFBFP` parameters: taken from the fixed optimal-control preset
- mesh sizes: `K = 50, 100, 200`
- starts per mesh: `10`
- stopping quantity: `R_n = 0.5 * ||z_n - J_A(z_n - B(z_n))||^2`
- tolerance: `1e-5`
- consecutive hits: `2`
- max iterations: `4000`
- representative figure run: `K = 100`, `seed 1`

## Outputs

Each problem now writes under the shared optimal-control result root:

- `results/optimal_control/<problem>/experiments.db`
- `results/optimal_control/<problem>/logs/`
- `results/optimal_control/<problem>/figures/`
