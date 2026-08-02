# Compressed-sensing figure protocol repair

Date: 2026-08-01

## Problem

`scripts/compressed_sensing/s70_figures_tables.jl` is still coupled to the superseded pre-regeneration workflow. It fails before plotting because `report_main` asks `resolved_aefbfp_params` for an `s20` tuned-winner row that the definitive manuscript run deliberately does not use. Even if that check is bypassed, `solve_representative_case` calls the legacy `build_problem`, which creates a non-row-orthonormal Gaussian matrix, non-`±1` signal, SNR-rescaled noise, and the old common-residual problem path. A workaround such as running `s20` or passing an untuned preset would therefore produce a figure from the wrong experiment.

## Approved outcome

The existing command remains valid:

```powershell
julia --project=. scripts/compressed_sensing/s70_figures_tables.jl --figures=signal_panels --png
```

It must:

1. read the definitive production rows already stored in `results/compressed_sensing/experiments.db`;
2. rebuild the selected case with `build_manuscript_problem` and the published case-specific streams;
3. select the exact seeded initial point identified by the production row's `seed_idx`/`dataset_idx` value;
4. build AEFBFP from `MANUSCRIPT_AEFBFP_PARAMS` and every baseline from its fixed `:paper` preset, exactly as production `s30` does;
5. write only below `jcode/results/compressed_sensing/figures/` and the existing CS log directory;
6. require no tuned-winner row and present no `s20`/untuned fallback as part of the definitive path.

## Design

### Pure protocol selection

Add a small numerical helper alongside the manuscript problem builder in `s30_benchmark.jl`. Given a case ID and start index, it will:

- resolve the case and its stable index in `DEFAULT_CASES`;
- validate that the start index is within `1:DEFAULT_INITIAL_POINTS`;
- call `build_manuscript_problem(case; case_index, gamma, n_inits=DEFAULT_INITIAL_POINTS)`;
- return the problem and `problem.initial_points[start_index]`.

Keeping this helper outside the plot script makes the critical data-selection behavior testable without importing `Plots` or `LaTeXStrings`.

### Figure re-solves

`solve_representative_case` will use the pure helper instead of `build_problem`/`dataset_seed`. Its existing `dataset_idx` value is the DB alias for `results.seed_idx`; it therefore maps directly to the manuscript start index. The representative-start chooser continues to rank production rows by convergence and distance from method medians, so only the reconstruction step changes.

`build_plot_algorithm` will delegate directly to the already-definitive `build_algorithm`, whose AEFBFP branch returns `AEFBFP(; MANUSCRIPT_AEFBFP_PARAMS...)`. The plotting path will not accept or resolve tuned AEFBFP parameters.

### CLI and banner

Remove the stale `snr-db`, `aefbfp-preset`, and `aefbfp-round-digits` data flow from the definitive plotting configuration and call sites. The banner will explicitly state `manuscript_v1` and print `MANUSCRIPT_AEFBFP_PARAMS`; it will not query `tuned_winners`.

The supported figure keys and `--png` behavior remain unchanged. Output stems remain `cs_signal_panels`, `cs_convergence_*`, `cs_convergence_cpu_*`, and `cs_resolvent_convergence_*` under `FIGDIR`, which is rooted at `JCODE_ROOT/results/compressed_sensing/figures`.

## Error handling

- Unknown case IDs remain errors.
- Start indices outside `1:10` produce an `ArgumentError` before any solve.
- Missing production rows retain the current explicit error.
- Unsupported figure keys retain the current explicit error.
- No code path recommends `s20` for definitive figures.

## Test strategy

All Codex-run tests remain plot-free.

1. Add failing tests to `test/test_cs_protocol.jl` for the wished-for pure helper:
   - Case 3/start 4 returns metadata protocol `manuscript_v1`.
   - Its `C`, `x_star`, `noise`, `y`, and selected-start hashes match a direct `build_manuscript_problem` call.
   - The chosen initial-point label and seed index are `seed4` and `4`.
   - Index 0 and 11 throw `ArgumentError`.
2. Run the tests before implementation and confirm failure because the helper is absent.
3. Implement the minimal helper and patch the plotting-only call chain.
4. Rerun `test/test_cs_protocol.jl` and the core suite.
5. Perform static assertions that the plot script contains `build_manuscript_problem`/the pure helper and contains no executable tuned-winner resolution in `report_main` or legacy `build_problem` call in `solve_representative_case`.
6. Do not import a plotting library or execute `s70_figures_tables.jl`. Mohammed performs the final visual run with the unchanged command.

## Scope

This repair changes only the CS figure reconstruction handoff and its plot-free protocol test seam. It does not alter production rows, timings, tables, manifests, solver mathematics, other plot families, or anything outside `jcode/`.
