# Full Numerical Plot Handoff Repair Design

**Date:** 2026-08-01

## Objective

Repair the complete numerical-figure handoff so that every generated plot uses
the definitive manuscript protocol and every PDF/PNG output has consistent,
publication-ready presentation. Preserve the underlying cases, starts, method
curves, numerical observations, and established method colors. Keep every
generated artifact under `jcode/results/`.

Codex must not execute plotting code or import `Plots` or `LaTeXStrings`.
Mohammed will regenerate the figures after the plot-free implementation and
verification pass.

## Evidence and Root Causes

### Optimal-control protocol drift

The definitive double-integrator and harmonic-oscillator benchmark entry
points use the manuscript AEFBFP parameters

```text
mu=0.32, tau_0=0.05, xi_exp=1.11, sigma_exp=0.97, sigma_scale=0.024.
```

Each optimal-control `s70_figures_tables.jl` instead contains a stale local
tuple with the earlier full-precision tuned values. The fresh figure logs show
that stale tuple was used for convergence re-solves and the AEFBFP control and
state reference runs. The plotting scripts therefore diverge from their own
definitive `s30_benchmark.jl` files.

Root cause: manuscript constants were copied into four independent files. The
two benchmark copies were repaired previously, but the two plotting copies
were not.

### PDF/PNG layout drift

The fresh PNG previews place the horizontal guide incorrectly in
`di_control.png` and `game_random500_aefbfp_tau.png`. Rendering the matching
PDFs independently through Ghostscript shows correctly centered guides.

Both affected scripts render the PDF first and then reuse the same mutable
`Plots.Plot` object for PNG output. The divergence occurs at this format
boundary rather than in the plot's axis specification.

Root-cause hypothesis: backend layout preparation for the first `savefig`
mutates state reused by the second format render. The repair isolates each
format render with an independent deep copy of the pristine plot.

## Selected Architecture

### One authoritative optimal-control parameter source

Create `scripts/optimal_control/manuscript_protocol.jl` containing only the
definitive, plot-free `OC_MANUSCRIPT_AEFBFP_PARAMS` tuple. Both optimal-control
`s30_benchmark.jl` files and both optimal-control `s70_figures_tables.jl` files
include this file and set their compatibility alias
`OC_SHARED_AEFBFP_PARAMS = OC_MANUSCRIPT_AEFBFP_PARAMS`.

No optimal-control plotting entry point may contain the stale tuned tuple or a
second manuscript tuple. The two `s70` report configurations and re-solve paths
will no longer accept tuning, preset, rounding, or untuned fallbacks: manuscript
figures always use the authoritative tuple. Exploratory selection remains
available only in the non-plotting search/benchmark workflow.

### Format-isolated saving

Each figure entry point saves every requested format from a separate deep copy
of the pristine plot:

```julia
savefig(deepcopy(plt), pdf_path)
png && savefig(deepcopy(plt), png_path)
```

Apply this boundary to:

- compressed-sensing `s70_figures_tables.jl`;
- both optimal-control `s70_figures_tables.jl` files; and
- saddle-point `s71_figures.jl` through a small local save helper.

This preserves the current plotting backend and output formats while
preventing one renderer from influencing the next.

## Publication Presentation

### Shared principles

- Preserve all data, method ordering, curves, and established colors.
- Use white backgrounds, box frames, restrained grids, readable fonts, and
  adequate margins.
- Keep single-panel numerical figures at 900 by 620 pixels with 220-DPI PNG
  output. Preserve the accepted tall compressed-sensing signal-panel layout.
- Keep legends inside unused plot regions where they do not cover important
  trajectories.
- Use centered mathematical time labels and consistent tick systems between
  the control and state plot for the same problem.

### Compressed sensing

The accepted signal-panel content and layout remain unchanged. Only the
format-isolated save boundary is added, followed by a fresh visual audit of the
PDF and PNG.

### Double-integrator optimal control

- Route every AEFBFP re-solve through the authoritative manuscript tuple.
- Use matching time ticks `[0, 1.2, 2]` on the control and state figures.
- Keep the centered mathematical label `t` on both figures.
- Extend the convergence residual axis to `10^-6` so the final below-tolerance
  events are visible.
- Add a restrained dashed `10^-5` stopping-tolerance reference line.

### Harmonic-oscillator optimal control

- Route every AEFBFP re-solve through the authoritative manuscript tuple.
- Use matching ticks `0, pi/2, pi, 3pi/2, 2pi, 5pi/2, 3pi` on the control and
  state figures, with publication-facing pi labels.
- Keep the centered mathematical label `t` on both figures.
- Extend the convergence residual axis to `10^-6` and add the same dashed
  `10^-5` stopping-tolerance reference line.

### Saddle-point diagnostics

- Use the same 900 by 620 single-panel presentation standard.
- Keep every residual and branch observation; do not downsample.
- Label the horizontal coordinate as `Event index` for the AEFBFP tau panel.
- Reduce branch-marker size and opacity so the tau trajectory remains visible.
- Add data-derived vertical padding so the initial `tau_0=0.05` event does not
  touch the upper frame.
- Preserve the existing residual and tau data, method order, and colors.

## Testing Strategy

Create `test/test_figure_handoff.jl`, which does not include any plotting entry
point. It reads plotting sources as text and includes only the new plot-free
manuscript parameter file.

The regression suite will assert:

1. the authoritative tuple has exactly the five definitive numerical values
   and the two `:power` rules;
2. all four optimal-control `s30`/`s70` scripts include the authoritative file;
3. neither optimal-control `s70` contains the stale full-precision numbers;
4. neither optimal-control `s70` exposes preset, rounding, untuned, or tuned
   winner resolution in its report/re-solve path;
5. all four plotting entry points use format-isolated `deepcopy` saves;
6. the required tolerance lines, extended residual limits, matching time ticks,
   saddle sizing, branch-marker presentation, and tau headroom logic are present;
7. all output roots remain descendants of `jcode/results/`.

Follow red-green development:

1. add the regression assertions and run them to observe the expected failures;
2. add the authoritative parameter file and update the four OC consumers;
3. run the suite to isolate the remaining export/presentation failures;
4. implement format isolation and presentation changes;
5. rerun the new suite plus the existing CS, OC, saddle-point, saddle-runner,
   and core suites; and
6. syntax-parse every modified Julia file without executing it.

## Handoff and Visual Acceptance

After plot-free tests pass, Mohammed reruns the existing four commands from
`jcode/` with `JULIA_NUM_THREADS=1`. Expected filenames and output roots do not
change.

The final acceptance pass checks both the generated PNGs and Ghostscript
renders of the matching PDFs for:

- centered and unclipped guides;
- matching tick systems within each optimal-control problem;
- visible below-tolerance convergence endpoints and tolerance references;
- legible legends, curves, and branch markers;
- nonblank pages and correct panel content; and
- logs that print the definitive AEFBFP manuscript tuple.

Only after that visual and log audit should the approved PDFs be copied to
`paper/imgs/`. The legacy `tables.tex` files emitted by plotting entry points
remain excluded from the manuscript handoff.

## Scope Boundaries

- Do not alter benchmark databases, stored rows, stopping rules, cases, seeds,
  initial points, method formulas, or table generators.
- Do not execute plotting code or import a plotting library under Codex.
- Do not write generated plots outside `jcode/results/`.
- Do not copy files into `paper/imgs/` as part of this repair.
- Do not refactor the full duplicated OC workflow; centralize only the protocol
  constant needed to prevent recurrence of this defect.
