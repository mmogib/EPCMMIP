# Supplementary Runs and Timing Protocol v2 Design

**Date:** 2026-08-02

## Objective

Deliver exchange 009 without changing any production numerical signature or
executing plotting code. Repair the three figure sources, replace contaminated
internal CPU measurements with externally batched monitoring-free timings,
run the duplicated-identity extended-cap and xi-matched companions, and gate a
K=10000 double-integrator benchmark through a one-start pilot.

The existing production databases and rows remain immutable evidence. New
timing repetitions and supplementary rows are additive and content-addressed.

## Constraints and Accepted Protocol

- Julia 1.12.6, one Julia thread, one BLAS thread.
- Existing problem generators, named seeds, starts, tolerances, consecutive-hit
  rules, algorithms, and algorithmic forward-evaluation accounting are reused.
- No plotting entry point or plotting dependency is executed by Codex.
- Timing v2 has two warm-ups, at least three retained repetitions, an external
  `time_ns()` clock, and batching until each measured batch is approximately
  0.1 seconds or longer.
- Every solve in a timed batch must reproduce the stored production
  `(flag, iterations, f_evals)` signature. A mismatch aborts the cell before a
  timing row is accepted.
- No protocol parameter is altered to rescue a failed supplementary cell.

## Architecture

### Monitoring-free solver mode

Add opt-in keywords to the six manuscript solvers and the game-local HFB
solver:

```julia
monitor_residual::Bool = true
record_elapsed::Bool = true
```

Defaults preserve every existing call. Timing v2 passes both as `false`.
Algorithmically required B evaluations and cache rotations remain unchanged.
Only the universal/native reporting residual block is skipped. When it is
skipped, the state residual fields receive finite zero sentinels so the existing
`NanStopping` callback does not stop a valid timing solve solely because no
reporting residual was computed. These timing results are never inserted as
production numerical rows; only their signature and external duration are
retained. Native stopping evaluations remain active where the protocol requires
them.

### Shared timing engine and additive schema

Create `src/timing_protocol_v2.jl`, included by `src/includes.jl`. It owns:

- the exact result signature;
- warm-up and calibration;
- external batched measurement with `time_ns()`;
- signature assertions for every solve;
- a `timing_v2_repetitions` SQLite table; and
- median/IQR summaries in milliseconds per solve.

Each family writes timing rows into its existing database. The primary key
includes the timing protocol hash, source production config hash, problem,
dimension, start, and repetition, so timing is resumable and cannot overwrite
production `results`. Every row retains batch size, total nanoseconds,
milliseconds per solve, expected signature, and run metadata.

Four small family entry points adapt existing builders and problem generators
to the common engine:

- `scripts/compressed_sensing/s35_timing_v2.jl`;
- `scripts/optimal_control/double_integrator_control/s35_timing_v2.jl`;
- `scripts/optimal_control/harmonic_oscillator/s35_timing_v2.jl`; and
- `scripts/saddle_point/s35_timing_v2.jl`.

They use no observers, disable universal residual monitoring/internal elapsed
sampling, retain only required stopping callbacks, and compare against the
latest definitive `s30` production row selected by the existing config hashes.
Each writes a JSON manifest under its family's existing manifest directory.

Plot-free table generators read timing-v2 medians instead of `results.cpu_time`.
The original `cpu_time` values remain untouched for auditability. The reply
reports median and IQR for all cells, including at-budget DNC cells; manuscript
table formatting may continue to use its existing convergence convention.

### Supplementary numerical runners

Use one additive `s40_supplementary.jl` per affected family, reusing the local
`s30` builders and insertion functions with distinct script tags and config
hash inputs.

- Saddle point: all seven methods on `duplicated_identity_q100`, ten starts,
  cap 100000; recompute anchor rows for AEFBFP, IFRAB, and RFBSM. Also run the
  xi-matched AEFBFP companion at the same cap.
- Compressed sensing: AEFBFP xi-matched companion on Case 1.
- Double integrator: AEFBFP xi-matched companion at K=100; separate K=10000
  pilot/full mode for all six methods.
- Harmonic oscillator: AEFBFP xi-matched companion at K=50,100,200.

The xi companion changes only `xi_exp` from 1.11 to 2.0, realizing
`xi_k=(k+1)^(-2)` with all other manuscript parameters unchanged. New manifests
record the exact tuple, source hashes, named seeds, tolerance, cap, repetitions,
and runtime fingerprint.

### K=10000 gate

Run seed 1 for all six methods first. Report its signature, convergence status,
iterations, native residual, and external timing projection. Proceed to the
remaining nine starts only if every pilot method reaches the unchanged native
residual tolerance before the 4000 cap and all outputs are finite. If any pilot
method is DNC/maxiter/nonfinite, mark Task 5 BLOCKED and do not change the norm,
tolerance, cap, or starting point.

## Figure-source repairs

The source-only handoff test will require:

- CS measurement text `y = Cx* + epsilon` and sparsity symbol `k`;
- removal of both OC `xlims=(0,250)` clauses; and
- explicit `start 6` (CS) or `start 1` (OC and games) in the displayed title or
  annotation.

No plot is rendered. The reply gives Mohammed exact rerender commands.

## Testing and Failure Handling

Follow red-green-refactor:

1. Extend `test/test_figure_handoff.jl` and observe source-level failures.
2. Add `test/test_timing_protocol_v2.jl` and observe missing-API failures.
3. Add monitoring-off solver tests with counted B/prox closures; require the
   same numerical signature and iterate, and the expected reduction in
   monitoring calls.
4. Implement the minimal solver and timing engine changes.
5. Add family-runner protocol tests for exact cells, parameters, seeds, caps,
   script tags, and manifest fields before implementing runners.
6. Run the full pre-existing test suite plus all new tests.
7. Syntax-parse every modified figure source without including it.
8. Run smoke/quick timing cells, then production commands.

A signature mismatch, missing production row, hash mismatch, nonfinite timing,
or pilot gate failure is fatal for that task. It is reported rather than
patched by changing scientific protocol.

## Scope Boundaries

- Do not modify or replace existing production result rows.
- Do not import or run `Plots`/`LaTeXStrings`.
- Do not change algorithms' default behavior or forward-evaluation counts.
- Do not compile LaTeX or edit files outside `jcode/` except the final channel
  reply.
- Do not infer a rescue protocol for failed supplementary runs.
