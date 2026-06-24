# EPCMMIP

Julia implementation and benchmark study of the **Extrapolation–Proximal–Contraction
Method** (EPCM) and its two reduced variants (**EFBFP**, **AEFBFP**) for the monotone
variational inclusion
$$0 \in A(x) + B(x),$$
where $A$ is maximal monotone (set-valued) and $B$ is monotone and $L$-Lipschitz
(single-valued, **not** assumed cocoercive).

EPCM is a Halpern-anchored, forward-reflected–backward splitting method with an
**$L$-free self-adaptive step size** (no knowledge of $L$ required), a single resolvent
of $A$ and **one new evaluation of $B$ per iteration** (it reuses the previous
$B$-value), and a projection–contraction correction. It is **strongly convergent** to
$P_{\mathcal S}(x_0)$ — the metric projection of the anchor onto the solution set —
without cocoercivity, strong monotonicity, or compactness. The two variants drop the
projection–contraction step (**EFBFP**) and additionally switch to a min-rule step size
(**AEFBFP**); all three are proven strongly convergent.

This repository accompanies a study by M. Uddin, M. Alshahrani (corresponding), and
Q. H. Ansari (KFUPM Department of Mathematics / IRC-MOIN).

## What this repository contains
- A source-verified implementation of EPCM, EFBFP, AEFBFP (proposed) and four reference solvers.
- Four monotone-inclusion test problems drawn from the recent splitting literature.
- A reproducible benchmark pipeline (SQLite-backed experiment database, parameter
  tuning via Latin-hypercube search, source-paper validation, a full sweep, and the
  paper's figures/tables).

## Methods

| Acronym | Reference | Algorithm |
|---------|-----------|-----------|
| EPCM    | This study | Proposed (Alg. 1) |
| EFBFP   | This study | Proposed reduced variant (Alg. 2) |
| AEFBFP  | This study | Proposed reduced variant, min-rule step (Alg. 3) |
| MTTM    | Gibali–Thong, 2018 | Alg. 1 |
| IMTTM   | Tan–Cho, 2022 | Alg. 3.3 |
| IFRAB   | Izuchukwu et al., 2023 | Alg. 4.5 (inertial) |
| SFRBM   | Yao–Adamu–Shehu, 2024 | Alg. 2 |

## Test problems
*(suite under finalization — P1/P2 may be revised)*

| Problem | Source | Structure |
|---------|--------|-----------|
| P1 | Tan–Cho, 2022 (§5.2) | Volterra split-feasibility, $L=(2/\pi)^2$ |
| P2 | Yao–Adamu–Shehu, 2024 (§4.2) | $\ell_1$ + quadratic, $L=2$ |
| P3 | Izuchukwu et al., 2023 (§6.2) | box-constrained, **rank-1** $B = 2h\,\mathbf{1}\mathbf{1}^\top$ |
| P4 | Tan–Cho, 2022 (§5.3) | LASSO, $C$ underdetermined, $L=1$ |

## Benchmark findings (current)

Full tuned-vs-tuned sweep (`s30`; uniform native tolerance $10^{-6}$, budget $6000$;
0 errors), with all proposed methods run at parameters **inside the proven admissible
region** ($\vartheta_0,\mu < 1/\sqrt6$). **Median iterations to the native stopping
tolerance** over converged seeds $\times$ dimensions (`converged / total`):

| Method | P1 | P2 | P3 | P4 |
|--------|----|----|----|----|
| EPCM   | 1802 (28/30) | 1596 (40/40) | 265 (60/60) | 4272 (20/20) |
| EFBFP  | 1431 (30/30) | 1377 (39/40) | 246 (60/60) | 3994 (20/20) |
| **AEFBFP** | 1806 (30/30) | **656 (40/40)** | **171 (60/60)** | **3266 (20/20)** |
| IFRAB  | **348 (30/30)** | **46 (40/40)** | 376 (60/60) | 3532 (20/20) |
| MTTM   | — (artifact) | 4578 (20/40) | DNC | DNC |
| IMTTM  | 10 (30/30)\* | 4573 (20/40) | DNC | DNC |
| SFRBM  | DNC | 1204 (40/40) | DNC | DNC |

`DNC` = no convergence within the budget on a majority of instances.
`*` MTTM/IMTTM are unusually fast on P1 only through near-trivial feasibility of the
initialization (MTTM/P1 is excluded as an artifact).

Highlights:
- **AEFBFP wins P3 and P4**, beating the nearest single-call competitor IFRAB on both
  (P3 by $\sim 2.2\times$; P4 by $\sim 8\%$); on P3 all three proposed methods are
  faster than every baseline.
- **EFBFP $\ge$ EPCM** on every problem — the projection–contraction step is empirically
  removable on this suite.
- On the well-conditioned **P1** the anchored methods are slow relative to IFRAB; the
  proposed family is most competitive on structured / harder problems (P3, P4).

## Status

Theory is finalized and independently verified — strong convergence of all three
methods, with admissibility $\vartheta_0,\mu < 1/\sqrt6$. The numerical study is **in
progress**: the test suite is being finalized and the manuscript's numerical section is
being written. The implementation, baselines, test problems, and benchmark pipeline are
a reusable foundation for monotone-operator-splitting research.

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
probe), `s20` (Latin-hypercube tuning + promotes the fine-tuned presets), `s28`
(source-paper baseline validation), `s30` (full benchmark), `s70` (figures & LaTeX
tables from the `s30` production data).

## Layout
- `src/` — solver, problem, and benchmark code (flat-include, no module wrapper)
- `configs/` — parameter presets (tracked): competitors' source values +
  the `s20`-generated fine-tuned winners
- `scripts/` — numbered experiment scripts
- `results/` — SQLite database, logs, figures (not tracked)

## License
TBD.
