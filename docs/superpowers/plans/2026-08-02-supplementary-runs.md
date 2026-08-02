# Supplementary Runs and Timing Protocol v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repair figure sources, add a signature-gated external timing protocol, and deliver the extended-cap, xi-matched, and expensive-B supplementary experiments commissioned by exchange 009.

**Architecture:** Preserve all `s30` rows and defaults. Add opt-in monitoring-free solver keywords, a shared external timing/SQLite layer, four thin family timing adapters, and additive family supplementary runners. New rows use distinct script/protocol hashes; every timing solve is checked against its stored production signature.

**Tech Stack:** Julia 1.12.6, SQLite.jl/DBInterface, DataFrames, SHA, JSON3, `time_ns()`, existing flat-include experiment infrastructure, Julia `Test`.

---

This plan is executed in the current shared `jcode/` tree because the adopted
numerical pipeline is uncommitted and absent from HEAD. Do not create a
worktree from HEAD, commit unrelated user changes, run plotting entry points,
or load plotting packages.

### Task 1: Repair and lock the figure sources

**Files:**
- Modify: `test/test_figure_handoff.jl`
- Modify: `scripts/compressed_sensing/s70_figures_tables.jl`
- Modify: `scripts/optimal_control/double_integrator_control/s70_figures_tables.jl`
- Modify: `scripts/optimal_control/harmonic_oscillator/s70_figures_tables.jl`
- Modify: `scripts/saddle_point/s71_figures.jl`

- [ ] **Step 1: Add failing source assertions**

Assert that the CS source contains the C/epsilon measurement label, uses `k`
for sparsity, and names start 6; assert neither OC source contains
`xlims=(0,250)` or its spaced equivalent and both name start 1; assert the game
source names start 1. These tests read source text and never include a plotting
entry point.

- [ ] **Step 2: Verify RED**

```powershell
$env:JULIA_NUM_THREADS='1'
& 'C:\Users\mmogi\.julia\juliaup\julia-1.12.6+0.x64.w64.mingw32\bin\julia.exe' --project=. test/test_figure_handoff.jl
```

Expected: existing 40 assertions remain green and the new label/x-limit/start
assertions fail for the current sources.

- [ ] **Step 3: Make the minimal source repairs**

Change only label/title/annotation text and delete the two fixed `xlims`
keywords. Preserve data selection, curves, output filenames, dimensions, and
save logic.

- [ ] **Step 4: Verify GREEN and syntax**

Run the Step-2 command, then parse each modified figure source with
`Meta.parseall(read(path,String))` in a Julia `-e` command. Do not include it.

### Task 2: Add monitoring-free solver execution

**Files:**
- Create: `test/test_timing_protocol_v2.jl`
- Modify: `src/algorithm.jl`
- Modify: `scripts/saddle_point/halpern_forward_backward.jl`
- Modify: `test/runtests.jl`

- [ ] **Step 1: Write counted-closure signature tests**

Construct a small monotone linear `TestProblem` whose B and resolvent closures
increment counters. For AEFBFP, VAFBS, MDITSM, RFBSM, IRFBSM, IFRAB, and HFB,
run the same method twice for a fixed short iteration cap: once with defaults
and once with `monitor_residual=false, record_elapsed=false`. Assert equal
`(flag,iterations,f_evals)`, equal final iterates, finite zero residual
sentinels only in monitoring-free results, and fewer diagnostic closure calls
where the existing method has a separate residual block.

- [ ] **Step 2: Verify RED**

Run `test/test_timing_protocol_v2.jl`; expected failure is an unsupported
`monitor_residual` keyword.

- [ ] **Step 3: Add opt-in keywords and guarded residual blocks**

For each relevant `solve`, add:

```julia
monitor_residual::Bool = true,
record_elapsed::Bool = true
```

Use `record_elapsed && (state.elapsed = time() - t0)`. If a residual block is
diagnostic only, guard the whole B/prox block. If its B value is required for
the algorithm/cache, keep that B call and guard only the prox/norm block. When
disabled, assign `state.residual = 0.0` and
`state.scaled_residual = 0.0`. Defaults must produce byte-for-byte-equivalent
iteration logic.

- [ ] **Step 4: Verify GREEN and regression suites**

Run the new test, `test/runtests.jl`, `test/test_cs_protocol.jl`,
`test/test_oc_protocol.jl`, `test/test_saddle_point.jl`, and
`test/test_saddle_runner.jl`.

### Task 3: Implement the shared timing-v2 engine and schema

**Files:**
- Modify: `test/test_timing_protocol_v2.jl`
- Create: `src/timing_protocol_v2.jl`
- Modify: `src/includes.jl`

- [ ] **Step 1: Add failing timing/database tests**

Tests require:

```julia
timing_signature(result) == (result.flag, result.iterations, result.f_evals)
run_timing_v2(solve_once, expected; warmups=2, repetitions=3,
              min_batch_seconds=0.1)
ensure_timing_v2_table!(db)
insert_timing_v2_repetition!(db, row)
timing_v2_summary(db, protocol_hash)
```

Use a deterministic artificial solve closure. Assert two warm-ups occur,
three repetition records are returned, every batch is nonempty, durations and
ms/solve are positive, a wrong signature throws, all repetitions survive a DB
round trip, duplicate insertion is idempotent, and summary median/Q1/Q3 are
correct.

- [ ] **Step 2: Verify RED**

Expected: the timing source/API does not exist.

- [ ] **Step 3: Implement minimal timing types and engine**

Use `time_ns()` around whole batches. Calibrate by doubling batch size until a
batch reaches the requested duration. Validate every solve's signature during
calibration and measurement. If a measured batch is materially shorter than
the target, grow the batch and repeat that repetition rather than storing a
short sample. Return raw rows; do not aggregate away repetitions.

- [ ] **Step 4: Implement additive SQLite storage**

Create `timing_v2_repetitions` with a primary key over protocol/source/cell/
repetition and columns for family, method, source config hash, problem,
dimension, init, seed, expected flag/iterations/f-evals, batch size, total ns,
ms/solve, run id, and created time. Implement resumability and Julia-side
median/quantile summary.

- [ ] **Step 5: Verify GREEN**

Run the new test twice to exercise idempotence, then the core suite.

### Task 4: Add and validate the four timing adapters

**Files:**
- Create: `scripts/compressed_sensing/s35_timing_v2.jl`
- Create: `scripts/optimal_control/double_integrator_control/s35_timing_v2.jl`
- Create: `scripts/optimal_control/harmonic_oscillator/s35_timing_v2.jl`
- Create: `scripts/saddle_point/s35_timing_v2.jl`
- Create: `test/test_timing_v2_runners.jl`
- Modify: `test/runtests.jl`
- Modify: the four plot-free `s70_tables.jl` files as required to consume timing v2

- [ ] **Step 1: Write runner source/protocol tests**

Require exact family database paths, production problem/case lists, ten starts,
six manuscript methods (seven for games), two warm-ups, three repetitions,
0.1-second minimum batches, monitoring/elapsed disabled, source-signature
lookup, signature assertions, JSON manifests, and summary/export modes. Require
all table generators to label timing as external milliseconds and source it
from retained timing-v2 repetitions.

- [ ] **Step 2: Verify RED**

Expected: all four runner files are missing.

- [ ] **Step 3: Implement thin family adapters**

Each adapter includes its existing `s30` entry point, selects the latest
definitive production config hash using existing selectors, reconstructs the
same problem/start/algorithm/stopping tuple, and calls the common engine with
`observers=()`, `monitor_residual=false`, `record_elapsed=false`. It writes
one manifest and three retained rows per start. Add `--quick`, `--summary`, and
resume-by-default behavior; production defaults require three repetitions.

- [ ] **Step 4: Run one-cell quick gates**

Run one start and one method per family with a shorter test-only batch target,
then query the databases to confirm three rows and exact source signatures.
Quick protocol hashes must differ from production timing hashes.

- [ ] **Step 5: Run production timing v2**

Run all four adapters with one Julia/BLAS thread. Resume safely after any
interruption. Generate raw and median/IQR aligned-text summaries from retained
rows; do not run plotting or LaTeX.

### Task 5: Add duplicated-identity and xi-matched supplementary runners

**Files:**
- Create: `scripts/compressed_sensing/s40_supplementary.jl`
- Create: `scripts/optimal_control/double_integrator_control/s40_supplementary.jl`
- Create: `scripts/optimal_control/harmonic_oscillator/s40_supplementary.jl`
- Create: `scripts/saddle_point/s40_supplementary.jl`
- Create: `test/test_supplementary_protocol.jl`
- Modify: `test/runtests.jl`

- [ ] **Step 1: Write exact-protocol tests**

Assert the duplicated-identity runner uses q=100, all seven methods, ten named
starts, tolerance 1e-6, two hits, cap 100000, distinct script tags, and anchor
insertion for AEFBFP/IFRAB/RFBSM. Assert each xi companion constructs
`AEFBFP(; manuscript_params..., xi_exp=2.0)` without changing any other field,
uses exactly CS Case 1, DI K=100, HO K=50/100/200, and duplicated identity cap
100000, and writes manifests.

- [ ] **Step 2: Verify RED**

Expected: runner files/API absent.

- [ ] **Step 3: Implement minimal additive runners**

Reuse local problem builders, starts, stopping tuples, signature aggregators,
metrics, and manifest utilities. Use script tags `s40_extended_cap` and
`s40_xi2_companion`; never replace `s30` rows. Check the three repetition
signatures before insertion.

- [ ] **Step 4: Verify GREEN and quick smoke cells**

Run protocol tests and one short-cap/start quick cell per runner under a quick
hash/script tag. Confirm no production row changed.

- [ ] **Step 5: Run the commissioned production cells**

Run the extended duplicated-identity study and all xi2 cells. Recompute and
query terminal natural residual, gap, objective/MSE/common residual or native
residual as applicable, stepsizes/branch fractions where stored, and anchor
diagnostics. Produce per-start rows and five-number summaries.

### Task 6: Gate and run the K=10000 expensive-B benchmark

**Files:**
- Modify: `scripts/optimal_control/double_integrator_control/s40_supplementary.jl`
- Modify: `test/test_supplementary_protocol.jl`

- [ ] **Step 1: Add pilot-gate tests**

Require K=10000, six manuscript methods, seed 1 pilot, tolerance 1e-5, two
hits, cap 4000, exact production builders, timing-v2 measurement, and a gate
that returns blocked if any pilot result is maxiter/DNC/nonfinite. Require full
mode to refuse execution without a persisted successful pilot record.

- [ ] **Step 2: Verify RED then implement gate**

The pilot writes results with `s40_expensive_b_pilot`; full rows use
`s40_expensive_b`. Project full cost from the measured six-method seed-1 pilot
without altering the protocol.

- [ ] **Step 3: Run pilot and inspect gate**

If any method fails the gate, stop Task 6 and report the evidence as BLOCKED.
If all pass, run seeds 2–10 and store external timing repetitions.

- [ ] **Step 4: Summarize**

Report per-method success, iteration/F counts, native residual, timing median
and IQR, and per-start spread. Compare elapsed time against forward-evaluation
counts without claiming causality beyond the measured cell.

### Task 7: Final verification and channel handoff

**Files:**
- Create: `../channels/codex_to_claude/009_supplementary-runs_reply.md`
- Read only: all new databases/logs/manifests and modified sources

- [ ] **Step 1: Run all non-plotting tests**

Run the six baseline suites plus the three new suites. All must pass with one
Julia/BLAS thread and no plotting dependency loaded.

- [ ] **Step 2: Audit database/manifests**

Count expected timing repetitions and supplementary rows; verify unique seeds,
protocol hashes, source signatures, run IDs, finite values, and no modification
of `s30` numerical signatures.

- [ ] **Step 3: Write the self-contained reply**

For Tasks 1–5 give commands, gates, exact data tables, per-start spreads,
manifest/log/database paths, anomalies, and any BLOCKED result. Include exact
rerender commands for Mohammed and explicitly state that Codex did not execute
plotting code.

- [ ] **Step 4: Verification-before-completion pass**

Re-read 009 and map every sentence to code, evidence, or an explicit BLOCKED
finding. Verify that the only non-`jcode/` write is the specified reply file.
