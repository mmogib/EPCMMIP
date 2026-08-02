# Full Numerical Plot Handoff Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make all manuscript plot entry points use definitive numerical parameters and produce format-isolated, publication-ready PDF/PNG outputs under `jcode/results/`.

**Architecture:** Put the definitive optimal-control AEFBFP tuple in one plot-free source consumed by both benchmark and figure entry points. Keep report re-solves on a dedicated definitive builder, isolate every format save with a pristine plot copy, and make focused axis/style changes without altering stored data or numerical algorithms. Verify through source-level Julia regression tests and syntax parsing; Mohammed alone executes plotting entry points.

**Tech Stack:** Julia 1.12.6, `Test`, existing Plots/GR figure entry points (source edits only under Codex), PowerShell static checks, Ghostscript for post-regeneration PDF review.

---

The workspace is not exposed as a Git checkout, so this plan has no executable commit steps. All edits and generated test artifacts remain inside `jcode/`. Never include or execute an `s70_figures_tables.jl` or `s71_figures.jl` file during Codex verification because those entry points import plotting packages.

### Task 1: Add the plot-free handoff regression suite

**Files:**
- Create: `test/test_figure_handoff.jl`
- Read only: `scripts/compressed_sensing/s70_figures_tables.jl`
- Read only: `scripts/optimal_control/double_integrator_control/s30_benchmark.jl`
- Read only: `scripts/optimal_control/double_integrator_control/s70_figures_tables.jl`
- Read only: `scripts/optimal_control/harmonic_oscillator/s30_benchmark.jl`
- Read only: `scripts/optimal_control/harmonic_oscillator/s70_figures_tables.jl`
- Read only: `scripts/saddle_point/s71_figures.jl`

- [ ] **Step 1: Create the failing source-level regression test**

```julia
using Test

const TEST_JCODE_ROOT = normpath(joinpath(@__DIR__, ".."))
const OC_PROTOCOL_PATH = joinpath(TEST_JCODE_ROOT, "scripts", "optimal_control",
                                  "manuscript_protocol.jl")
const DI_S30_PATH = joinpath(TEST_JCODE_ROOT, "scripts", "optimal_control",
                             "double_integrator_control", "s30_benchmark.jl")
const DI_S70_PATH = joinpath(TEST_JCODE_ROOT, "scripts", "optimal_control",
                             "double_integrator_control", "s70_figures_tables.jl")
const HO_S30_PATH = joinpath(TEST_JCODE_ROOT, "scripts", "optimal_control",
                             "harmonic_oscillator", "s30_benchmark.jl")
const HO_S70_PATH = joinpath(TEST_JCODE_ROOT, "scripts", "optimal_control",
                             "harmonic_oscillator", "s70_figures_tables.jl")
const CS_S70_PATH = joinpath(TEST_JCODE_ROOT, "scripts", "compressed_sensing",
                             "s70_figures_tables.jl")
const GAME_S71_PATH = joinpath(TEST_JCODE_ROOT, "scripts", "saddle_point",
                               "s71_figures.jl")

@testset "manuscript figure handoff" begin
    @test isfile(OC_PROTOCOL_PATH)
    if isfile(OC_PROTOCOL_PATH)
        include(OC_PROTOCOL_PATH)
        @test OC_MANUSCRIPT_AEFBFP_PARAMS == (
            mu = 0.32,
            tau_0 = 0.05,
            xi_rule = :power,
            sigma_rule = :power,
            xi_exp = 1.11,
            sigma_exp = 0.97,
            sigma_scale = 0.024,
        )
        protocol_source = read(OC_PROTOCOL_PATH, String)
        @test !occursin(r"(?m)^\s*(using|import)\s+(Plots|LaTeXStrings)", protocol_source)
    end

    include_token = "include(joinpath(@__DIR__, \"..\", \"manuscript_protocol.jl\"))"
    for path in (DI_S30_PATH, DI_S70_PATH, HO_S30_PATH, HO_S70_PATH)
        @test occursin(include_token, read(path, String))
    end

    stale_values = ("0.32475054644276846", "0.052386951978823273",
                    "1.1090105055152015", "0.9709015323685187",
                    "0.02390913974996533")
    for path in (DI_S70_PATH, HO_S70_PATH)
        source = read(path, String)
        @test all(value -> !occursin(value, source), stale_values)
        report_tail = last(split(source, "function read_report_config(args)"; limit = 2))
        @test !occursin("cfg.allow_untuned_aefbfp", report_tail)
        @test !occursin("cfg.aefbfp_preset", report_tail)
        @test !occursin("cfg.aefbfp_round_digits", report_tail)
        @test occursin("build_report_algorithm(method_name)", report_tail)
    end

    for path in (CS_S70_PATH, DI_S70_PATH, HO_S70_PATH, GAME_S71_PATH)
        source = read(path, String)
        @test occursin("savefig(deepcopy(plt)", source)
        @test !occursin("savefig(plt,", source)
    end

    di_source = read(DI_S70_PATH, String)
    @test occursin("const DI_TIME_TICKS", di_source)
    @test occursin("ylims = (1.0e-6, 1.0e1)", di_source)
    @test occursin("hline!(plt, [1.0e-5]", di_source)

    ho_source = read(HO_S70_PATH, String)
    @test occursin("const HO_TIME_TICKS", ho_source)
    @test occursin("ylims = (1.0e-6, 1.0e0)", ho_source)
    @test occursin("hline!(plt, [1.0e-5]", ho_source)

    game_source = read(GAME_S71_PATH, String)
    @test occursin("size = (900, 620)", game_source)
    @test occursin("dpi = 220", game_source)
    @test occursin("xlabel = \"Event index\"", game_source)
    @test occursin("markeralpha = 0.45", game_source)
    @test occursin("tau_padding", game_source)

    cs_source = read(CS_S70_PATH, String)
    @test occursin("joinpath(FIGDIR, \"cs_signal_panels\")", cs_source)
    @test occursin("local_figdir(spec)", di_source)
    @test occursin("local_figdir(spec)", ho_source)
    @test occursin("joinpath(GAME_FIGDIR", game_source)
end
```

- [ ] **Step 2: Run the new suite and verify RED**

Run from `jcode/`:

```powershell
$env:JULIA_NUM_THREADS='1'
& 'C:\Users\mmogi\.julia\juliaup\julia-1.12.6+0.x64.w64.mingw32\bin\julia.exe' --project=. test/test_figure_handoff.jl
```

Expected: nonzero exit. The authoritative file/include tests, stale-value tests,
report-route tests, format-isolation tests, and presentation tests fail for the
currently observed defects. The test itself must load no plotting package.

### Task 2: Centralize the definitive optimal-control parameters

**Files:**
- Create: `scripts/optimal_control/manuscript_protocol.jl`
- Modify: `scripts/optimal_control/double_integrator_control/s30_benchmark.jl:34-64`
- Modify: `scripts/optimal_control/double_integrator_control/s70_figures_tables.jl:49-81`
- Modify: `scripts/optimal_control/harmonic_oscillator/s30_benchmark.jl:34-64`
- Modify: `scripts/optimal_control/harmonic_oscillator/s70_figures_tables.jl:49-81`
- Test: `test/test_figure_handoff.jl`
- Test: `test/test_oc_protocol.jl`

- [ ] **Step 1: Add the authoritative plot-free tuple**

```julia
# Definitive manuscript parameters shared by optimal-control benchmarks and figures.
const OC_MANUSCRIPT_AEFBFP_PARAMS = (
    mu = 0.32,
    tau_0 = 0.05,
    xi_rule = :power,
    sigma_rule = :power,
    xi_exp = 1.11,
    sigma_exp = 0.97,
    sigma_scale = 0.024,
)
```

- [ ] **Step 2: Include the authority from all four consumers**

In each local `s30_benchmark.jl` and `s70_figures_tables.jl`, add after the
problem-definition include:

```julia
include(joinpath(@__DIR__, "..", "manuscript_protocol.jl"))
```

Delete each local `OC_MANUSCRIPT_AEFBFP_PARAMS = (...)` or stale
`OC_SHARED_AEFBFP_PARAMS = (...)` tuple and retain only:

```julia
const OC_SHARED_AEFBFP_PARAMS = OC_MANUSCRIPT_AEFBFP_PARAMS
```

- [ ] **Step 3: Run the handoff suite and inspect partial GREEN**

Run the Task-1 command.

Expected: the file/include/value assertions pass. Report-route,
format-isolation, and publication-presentation assertions remain red.

- [ ] **Step 4: Run the existing OC protocol suite**

```powershell
$env:JULIA_NUM_THREADS='1'
& 'C:\Users\mmogi\.julia\juliaup\julia-1.12.6+0.x64.w64.mingw32\bin\julia.exe' --project=. test/test_oc_protocol.jl
```

Expected: all existing definitive parameter, repetition-signature, hash, seed,
and problem-regeneration assertions pass without loading a plotting package.

### Task 3: Make OC report re-solves unconditionally definitive

**Files:**
- Modify: `scripts/optimal_control/double_integrator_control/s70_figures_tables.jl:606-623,865-881,943-965,1033-1123`
- Modify: `scripts/optimal_control/harmonic_oscillator/s70_figures_tables.jl:606-623,865-881,942-972,1040-1130`
- Test: `test/test_figure_handoff.jl`

- [ ] **Step 1: Add a report-only algorithm builder in both OC figure files**

Place this next to the existing exploratory `build_algorithm`:

```julia
function build_report_algorithm(method_name::AbstractString)
    method_name == "AEFBFP" && return AEFBFP(; OC_MANUSCRIPT_AEFBFP_PARAMS...)
    return OC_METHOD_BY_NAME[method_name](:paper)
end
```

- [ ] **Step 2: Make the report parameter banner definitive in both files**

Replace `print_method_parameter_block` with:

```julia
function print_method_parameter_block(tee)
    println(tee, "  method parameters:")
    println(tee, "    AEFBFP source : definitive manuscript set")
    println(tee, "    AEFBFP params : $(OC_MANUSCRIPT_AEFBFP_PARAMS)")
    for method_name in [name(T) for T in OC_METHOD_TYPES]
        alg = build_report_algorithm(method_name)
        println(tee, "    $(method_name) config: $(sprint(show, alg))")
    end
    return nothing
end
```

Update `print_report_banner` to call:

```julia
print_method_parameter_block(tee)
```

- [ ] **Step 3: Remove tuning inputs from each report configuration**

Use this report-only configuration shape in both files:

```julia
function read_report_config(args)
    opts, flags = parse_cli(args)
    quick = "quick" in flags
    eps = haskey(opts, "eps") ? parse(Float64, opts["eps"]) : OC_EPS_REF
    maxiter = haskey(opts, "maxiter") ? parse(Int, opts["maxiter"]) : canonical_maxiter(eps)
    ref_eps = haskey(opts, "ref-eps") ? parse(Float64, opts["ref-eps"]) : eps
    ref_maxiter = haskey(opts, "ref-maxiter") ? parse(Int, opts["ref-maxiter"]) : canonical_maxiter(ref_eps)
    ref_seed = haskey(opts, "ref-seed") ? parse(Int, opts["ref-seed"]) : OC_REF_INIT
    png = "png" in flags
    1 <= ref_seed <= OC_BENCH_INITS ||
        throw(ArgumentError("ref-seed must be between 1 and $(OC_BENCH_INITS), got $ref_seed"))
    return (quick = quick, eps = eps, maxiter = maxiter,
            ref_eps = ref_eps, ref_maxiter = ref_maxiter,
            ref_seed = ref_seed, png = png, production = !quick)
end
```

- [ ] **Step 4: Route representative and convergence re-solves through the report builder**

In both files, reduce `solve_reference_run` to definitive inputs:

```julia
function solve_reference_run(db, spec; eps::Float64, maxiter::Int,
                             method_name::String = "AEFBFP",
                             dim::Int = OC_REF_DIM,
                             seed_idx::Int = OC_REF_INIT)
    prob = spec.build_problem(dim; n_inits = OC_BENCH_INITS)
    init = prob.initial_points[seed_idx]
    alg = build_report_algorithm(method_name)
    stopping = make_stopping(prob, eps, maxiter; consec = 2)
    hc = HistoryCallback()
    nrec = NativeResRecorder(prob.native_residual)
    result = solve(alg, prob, copy(init.x0);
                   stopping = stopping, observers = (hc, nrec))
    return (prob = prob, init = init, result = result,
            history = hc.history, native_residual = nrec.value)
end
```

In `build_reference_convergence_rows`, replace the exploratory builder call
with:

```julia
alg = build_report_algorithm(method_name)
```

In `write_report_reference_figures`, call only:

```julia
rep = solve_reference_run(db, spec;
                          eps = cfg.ref_eps,
                          maxiter = cfg.ref_maxiter,
                          method_name = "AEFBFP",
                          dim = OC_REF_DIM,
                          seed_idx = cfg.ref_seed)
```

- [ ] **Step 5: Run the handoff suite and inspect partial GREEN**

Run the Task-1 command.

Expected: all authoritative/report-route assertions pass. Format-isolation and
presentation assertions remain red.

### Task 4: Isolate every PDF and PNG render

**Files:**
- Modify: `scripts/compressed_sensing/s70_figures_tables.jl:380-394`
- Modify: `scripts/optimal_control/double_integrator_control/s70_figures_tables.jl:849-852`
- Modify: `scripts/optimal_control/harmonic_oscillator/s70_figures_tables.jl:849-852`
- Modify: `scripts/saddle_point/s71_figures.jl:21-49`
- Test: `test/test_figure_handoff.jl`

- [ ] **Step 1: Clone the compressed-sensing plot per format**

Within its existing error-handling helper, use:

```julia
savefig(deepcopy(plt), pdf_path)
```

and:

```julia
savefig(deepcopy(plt), png_path)
```

- [ ] **Step 2: Replace each OC save helper**

Use this exact helper in both OC figure files:

```julia
function save_plot_files(plt, stem::String; png::Bool)
    savefig(deepcopy(plt), stem * ".pdf")
    png && savefig(deepcopy(plt), stem * ".png")
end
```

- [ ] **Step 3: Add and use a saddle format-isolation helper**

```julia
function save_game_plot_files(plt, stem::String)
    savefig(deepcopy(plt), joinpath(GAME_FIGDIR, stem * ".pdf"))
    savefig(deepcopy(plt), joinpath(GAME_FIGDIR, stem * ".png"))
    return nothing
end
```

Replace the four direct saddle `savefig` calls with:

```julia
save_game_plot_files(residual_plot, "game_random500_residual")
save_game_plot_files(tau_plot, "game_random500_aefbfp_tau")
```

- [ ] **Step 4: Run the handoff suite and inspect partial GREEN**

Run the Task-1 command.

Expected: format-isolation assertions pass. Only publication-presentation
assertions remain red.

### Task 5: Apply publication presentation consistently

**Files:**
- Modify: `scripts/optimal_control/double_integrator_control/s70_figures_tables.jl:82-846,918-1024`
- Modify: `scripts/optimal_control/harmonic_oscillator/s70_figures_tables.jl:82-846,917-1035`
- Modify: `scripts/saddle_point/s71_figures.jl:6-55`
- Test: `test/test_figure_handoff.jl`

- [ ] **Step 1: Define shared time ticks in each OC plot entry point**

For the double integrator:

```julia
const DI_TIME_TICKS = ([0.0, 1.2, 2.0], ["0", "1.2", "2"])
```

For the harmonic oscillator:

```julia
const HO_TIME_TICKS = (
    collect(0.0:pi / 2:3pi),
    ["0", "π/2", "π", "3π/2", "2π", "5π/2", "3π"],
)
```

- [ ] **Step 2: Complete each OC single-panel style**

Retain the existing attributes and add:

```julia
dpi = 220,
right_margin = 6Plots.mm,
top_margin = 5Plots.mm,
```

- [ ] **Step 3: Show the convergence stopping threshold and endpoints**

After plotting method curves in both files, add:

```julia
hline!(plt, [1.0e-5];
       label = L"\varepsilon=10^{-5}",
       color = RGB(0.25, 0.25, 0.25),
       lw = 1.4,
       linestyle = :dash)
```

Use `ylims = (1.0e-6, 1.0e1)` for the double integrator and
`ylims = (1.0e-6, 1.0e0)` for the harmonic oscillator.

- [ ] **Step 4: Apply matching ticks to each control/state pair**

Use this in both double-integrator plot builders:

```julia
xticks = DI_TIME_TICKS,
```

Use this in both harmonic-oscillator plot builders:

```julia
xticks = HO_TIME_TICKS,
```

Remove the now-duplicated local harmonic tick arrays from
`build_control_figure`.

- [ ] **Step 5: Add a publication-style saddle plot constructor**

```julia
function game_single_panel_plot(; kwargs...)
    return plot(; size = (900, 620),
                dpi = 220,
                background_color = :white,
                background_color_inside = :white,
                foreground_color_subplot = :black,
                foreground_color_legend = :black,
                background_color_legend = :white,
                legend_font_pointsize = 11,
                guidefont = font(12),
                tickfont = font(10),
                left_margin = 10Plots.mm,
                right_margin = 6Plots.mm,
                bottom_margin = 8Plots.mm,
                top_margin = 5Plots.mm,
                gridalpha = 0.18,
                framestyle = :box,
                kwargs...)
end
```

Construct both saddle figures with this helper rather than bare `plot` calls.

- [ ] **Step 6: Improve the saddle tau panel without dropping observations**

```julia
tau_values = Float64.(tau_rows.tau)
tau_span = maximum(tau_values) - minimum(tau_values)
tau_padding = max(0.06 * tau_span, 1.0e-3)
tau_plot = game_single_panel_plot(;
    xlabel = "Event index",
    ylabel = L"\tau_k",
    legend = :topright,
    ylims = (minimum(tau_values) - tau_padding,
             maximum(tau_values) + tau_padding),
)
plot!(tau_plot, tau_rows.k, tau_values;
      label = "AEFBFP", linewidth = 2)

ratio_rows = tau_rows[coalesce.(tau_rows.ratio_branch .== 1, false), :]
nrow(ratio_rows) == 0 ||
    scatter!(tau_plot, ratio_rows.k, Float64.(ratio_rows.tau);
             label = "ratio branch",
             markersize = 1.6,
             markeralpha = 0.45,
             markerstrokewidth = 0.0)
```

Construct the residual panel with `game_single_panel_plot`, retaining
`xlabel = "Iteration"`, `ylabel = L"R_k"`, logarithmic scaling, the existing
method order, and the current palette assignment.

- [ ] **Step 7: Run the handoff suite and verify GREEN**

Run the Task-1 command.

Expected: all manuscript parameter, definitive route, format isolation,
presentation, and output-root assertions pass without importing plotting code.

### Task 6: Complete plot-free verification and handoff

**Files:**
- Verify: `scripts/optimal_control/manuscript_protocol.jl`
- Verify: all six modified `s30`/`s70`/`s71` sources
- Verify: `test/test_figure_handoff.jl`

- [ ] **Step 1: Run the existing focused suites**

Run each command from `jcode/` with `JULIA_NUM_THREADS=1`:

```powershell
& 'C:\Users\mmogi\.julia\juliaup\julia-1.12.6+0.x64.w64.mingw32\bin\julia.exe' --project=. test/test_cs_protocol.jl
& 'C:\Users\mmogi\.julia\juliaup\julia-1.12.6+0.x64.w64.mingw32\bin\julia.exe' --project=. test/test_oc_protocol.jl
& 'C:\Users\mmogi\.julia\juliaup\julia-1.12.6+0.x64.w64.mingw32\bin\julia.exe' --project=. test/test_saddle_point.jl
& 'C:\Users\mmogi\.julia\juliaup\julia-1.12.6+0.x64.w64.mingw32\bin\julia.exe' --project=. test/test_saddle_runner.jl
& 'C:\Users\mmogi\.julia\juliaup\julia-1.12.6+0.x64.w64.mingw32\bin\julia.exe' --project=. test/runtests.jl
```

Expected: every suite exits zero with no failures.

- [ ] **Step 2: Syntax-parse every modified Julia source without execution**

```powershell
$env:JULIA_NUM_THREADS='1'
& 'C:\Users\mmogi\.julia\juliaup\julia-1.12.6+0.x64.w64.mingw32\bin\julia.exe' --project=. -e 'for path in ARGS; Meta.parseall(read(path, String)); println(path); end' `
  scripts/optimal_control/manuscript_protocol.jl `
  scripts/optimal_control/double_integrator_control/s30_benchmark.jl `
  scripts/optimal_control/double_integrator_control/s70_figures_tables.jl `
  scripts/optimal_control/harmonic_oscillator/s30_benchmark.jl `
  scripts/optimal_control/harmonic_oscillator/s70_figures_tables.jl `
  scripts/compressed_sensing/s70_figures_tables.jl `
  scripts/saddle_point/s71_figures.jl `
  test/test_figure_handoff.jl
```

Expected: each path prints once and Julia exits zero. Parsing must not evaluate
`using Plots` or any plotting call.

- [ ] **Step 3: Verify the non-plot boundary and stale-value removal**

```powershell
rg -n '^\s*(using|import)\s+(Plots|LaTeXStrings)' src test scripts/optimal_control/manuscript_protocol.jl scripts/optimal_control/*/s30_benchmark.jl
rg -n '0\.32475054644276846|0\.052386951978823273|1\.1090105055152015|0\.9709015323685187|0\.02390913974996533' scripts/optimal_control/*/s70_figures_tables.jl
```

Expected: both commands have no matches. Plot imports remain only in the
Mohammed-run figure entry points, outside the scanned non-plot boundary.

- [ ] **Step 4: Return the four unchanged regeneration commands**

```powershell
Set-Location 'D:\Dropbox\Research\Projects\IRC_MOIN\NewAlgorithm_EPCMMIP\jcode'
$env:JULIA_NUM_THREADS='1'
julia --project=. scripts/compressed_sensing/s70_figures_tables.jl --figures=signal_panels --png
julia --project=. scripts/optimal_control/double_integrator_control/s70_figures_tables.jl --png
julia --project=. scripts/optimal_control/harmonic_oscillator/s70_figures_tables.jl --png
julia --project=. scripts/saddle_point/s71_figures.jl
```

Expected outputs remain the nine PDF/PNG pairs under:

```text
jcode/results/compressed_sensing/figures/
jcode/results/optimal_control/double_integrator_control/figures/
jcode/results/optimal_control/harmonic_oscillator/figures/
jcode/results/saddle_point/figures/
```

- [ ] **Step 5: After Mohammed regenerates, perform the final visual acceptance pass**

Inspect all nine PNGs directly. Render all nine PDFs with Ghostscript into a
temporary `jcode/tmp/pdfs/` review directory, inspect those renders, and remove
only that verified temporary directory afterward. Confirm the six OC figures'
fresh logs print the definitive tuple before approving PDFs for
`paper/imgs/`.
