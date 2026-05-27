# EPCMMIP

Julia implementation of the **Extrapolated Projection--Contraction Method** (EPCM)
for monotone variational inclusion problems
$$0 \in A(x) + B(x),$$
where $A$ is maximal monotone (set-valued) and $B$ is monotone and Lipschitz
(single-valued).

This repository accompanies the manuscript

> M. Uddin, M. Alshahrani, Q. H. Ansari,
> *Strong Convergence of an Extrapolated Projection--Contraction Algorithm for Variational Inclusion Problems*.

## Benchmarks
The repository compares EPCM against four reference methods on three problem
classes (box-constrained VIP, $\ell_1$ + quadratic minimization, optimal-control
discretization):

| Acronym | Reference | Algorithm |
|---------|-----------|-----------|
| EPCM    | This paper | Proposed |
| MTTM    | Gibali--Thong, 2018 | Alg. 1 |
| IMTTM   | Tan--Cho, 2022 | Alg. 3.3 |
| IFRAB   | Izuchukwu et al., 2023 | Alg. 4.5 |
| SFRBM   | Yao--Adamu--Shehu, 2024 | Alg. 2 |

## Requirements
- Julia 1.10+ (developed on 1.12.6)

## Setup
```bash
julia --project=. -e 'import Pkg; Pkg.instantiate()'
```

## Running
```bash
julia --project=. scripts/s01_smoke_test.jl
```

## Layout
- `src/` — solver and problem code
- `scripts/` — numbered experiment scripts (`s01_…`, `s30_…`, `s70_…`)
- `results/` — SQLite database, logs, figures (not tracked)

## Status
Scaffolding. Algorithm and problem implementations pending.

## License
TBD.
