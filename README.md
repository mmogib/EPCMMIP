# EPCMMIP

Julia implementation and benchmark study of the **Extrapolated Projection–Contraction
Method** (EPCM) for monotone variational inclusion problems
$$0 \in A(x) + B(x),$$
where $A$ is maximal monotone (set-valued) and $B$ is monotone and $L$-Lipschitz
(single-valued, **not** assumed cocoercive).

EPCM is a Halpern-anchored, forward-reflected–backward splitting method with an
**$L$-free self-adaptive step size** and an additional projection–contraction (PC)
correction. It is **single-call** (one forward evaluation of $B$ and one resolvent of
$A$ per iteration) and **strongly convergent** to the projection of the anchor onto the
solution set, without cocoercivity or compactness.

This repository accompanies an exploratory study by M. Uddin, M. Alshahrani, and
Q. H. Ansari (KFUPM Department of Mathematics / IRC-MOIN).

## What this repository contains
- A source-verified implementation of EPCM (proposed) and four reference solvers.
- Four monotone-inclusion test problems drawn from the recent splitting literature.
- A reproducible benchmark pipeline (SQLite-backed experiment database, parameter
  tuning via Latin-hypercube search, source-paper validation, and a full sweep).

## Methods

| Acronym | Reference | Algorithm |
|---------|-----------|-----------|
| EPCM    | This study | Proposed |
| MTTM    | Gibali–Thong, 2018 | Alg. 1 |
| IMTTM   | Tan–Cho, 2022 | Alg. 3.3 |
| IFRAB   | Izuchukwu et al., 2023 | Alg. 4.5 (inertial) |
| SFRBM   | Yao–Adamu–Shehu, 2024 | Alg. 2 |

## Test problems

| Problem | Source | Structure |
|---------|--------|-----------|
| P1 | Tan–Cho, 2022 (§5.2) | Volterra split-feasibility, $L=(2/\pi)^2$ |
| P2 | Yao–Adamu–Shehu, 2024 (§4.2) | $\ell_1$ + quadratic, strongly monotone, $L=2$ |
| P3 | Izuchukwu et al., 2023 (§6.2) | box-constrained, **rank-1** $B = 2h\,\mathbf{1}\mathbf{1}^\top$ |
| P4 | Tan–Cho, 2022 (§5.3) | LASSO, $C$ underdetermined, $L=1$ |

## Benchmark findings

The headline conclusion of this study is a **negative result for EPCM as currently
formulated**: it is **structurally slow** on well-conditioned / full-rank problems and
is **dominated by IFRAB** (Izuchukwu et al., 2023) — a published method in the same
forward-reflected family that pays for the anchoring drag with working inertia.

Full benchmark sweep (`s30`, 490 cells, 0 errors; tuned-vs-tuned, **median iterations to
reach each problem's native stopping tolerance**, with `converged / total` instances):

| Method | P1 | P2 | P3 | P4 |
|--------|----|----|----|----|
| **EPCM** | 908 (24/30) | 1012 (40/40) | **157 (20/20)** | DNC (2/20) |
| IFRAB | **290 (30/30)** | **46 (40/40)** | 400 (20/20) | DNC (0/20) |
| MTTM  | 2 (30/30)* | DNC | DNC | DNC |
| IMTTM | 10 (30/30)* | DNC | DNC | DNC |
| SFRBM | DNC | 1204 (40/40) | DNC | DNC |

`DNC` = did not converge within the iteration budget on a majority of instances.
`*` MTTM/IMTTM "converge" on P1 only through trivial feasibility of the initialization,
not as a genuine performance win.

**Why EPCM is slow** (intrinsic to the method + its convergence proof, not an
implementation artifact — the solver is independently verified):
1. **Proof-mandated small step.** The admissibility condition forces the self-adaptive
   step to settle near $\vartheta_1/L < 0.5/L$ on any full-rank operator — typically
   3–9× below the Tseng-type baselines.
2. **Anchoring drag without inertial compensation.** The Halpern term that buys strong
   convergence is a persistent deceleration; IFRAB/SFRBM carry the same guarantee *and*
   add inertia/momentum to offset it, while EPCM does not.

**Where EPCM does shine.** Its one clear, robust win is **P3 — the rank-1 structured
operator** ($\sim 2.5\times$ faster than IFRAB). EPCM is fast exactly when $B$ is
degenerate enough that its adaptive step is free to grow, pointing to
low-rank / structured operators as the regime where this method family is genuinely
competitive.

## Status

The EPCM benchmark line of work is **concluded**: on this suite EPCM offers no
demonstrated advantage over the already-published IFRAB outside the rank-1 case. The
implementation, baselines, test problems, and benchmark pipeline are retained as a
**reusable foundation** for follow-on monotone-operator-splitting research and are not
slated for deletion.

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
Numbered scripts: `s01` (smoke), `s10` (one-at-a-time sensitivity), `s11` (PC-stride
probe), `s20` (Latin-hypercube tuning), `s25` (promote best parameters), `s28`
(source-paper baseline validation), `s30` (full benchmark).

## Layout
- `src/` — solver, problem, and benchmark code (flat-include, no module wrapper)
- `scripts/` — numbered experiment scripts
- `results/` — SQLite database, logs, figures (not tracked)

## License
TBD.
