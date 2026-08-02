# Compressed-Sensing Figure Protocol Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the existing CS plotting command reconstruct the definitive manuscript problem/start and frozen algorithms without requiring an `s20` tuned-winner row.

**Architecture:** Add one pure, plot-free selector in `s30_benchmark.jl` that maps a production case ID and DB seed index to the exact manuscript problem and initial point. Make `s70_figures_tables.jl` consume that selector and the existing definitive `build_algorithm` path, while removing all stale tuned/SNR option flow. Verify behavior through the ordinary CS protocol suite plus source-level assertions, never by importing or executing a plotting package.

**Tech Stack:** Julia 1.12.6, `Test`, existing `TestProblem`/SQLite benchmark infrastructure, PowerShell static checks.

---

The workspace is not exposed as a Git checkout (`git status` reports no repository), so this plan does not include executable commit steps. All edits remain inside `jcode/`.

### Task 1: Add failing protocol-selection regression tests

**Files:**
- Modify: `test/test_cs_protocol.jl`
- Read only: `scripts/compressed_sensing/s70_figures_tables.jl`

- [ ] **Step 1: Add a failing test for the pure selector and stale plot-path strings**

Append inside the existing `@testset`:

```julia
    @test isdefined(Main, :manuscript_problem_start)
    if isdefined(Main, :manuscript_problem_start)
        selected = manuscript_problem_start(DEFAULT_CASES[3].problem, 4)
        direct = build_manuscript_problem(DEFAULT_CASES[3];
                                          case_index = 3,
                                          gamma = GAMMA_REF,
                                          n_inits = DEFAULT_INITIAL_POINTS)
        @test selected.case_index == 3
        @test selected.prob.metadata.protocol == "manuscript_v1"
        @test selected.prob.metadata.hashes == direct.metadata.hashes
        @test selected.init.label == "seed4"
        @test selected.init.seed_idx == 4
        @test array_sha256(selected.init.x0) == direct.metadata.hashes.starts[4]
        @test_throws ArgumentError manuscript_problem_start(DEFAULT_CASES[3].problem, 0)
        @test_throws ArgumentError manuscript_problem_start(DEFAULT_CASES[3].problem, 11)
        @test_throws ArgumentError manuscript_problem_start("unknown_case", 1)
    end

    plot_source = read(joinpath(@__DIR__, "..", "scripts", "compressed_sensing",
                                "s70_figures_tables.jl"), String)
    @test occursin("manuscript_problem_start", plot_source)
    @test !occursin("resolved_aefbfp_params(db;", plot_source)
    @test !occursin("prob = build_problem(case;", plot_source)
```

- [ ] **Step 2: Run the CS protocol suite and verify RED**

Run from `jcode/`:

```powershell
$env:JULIA_NUM_THREADS='1'
& 'C:\Users\mmogi\.julia\juliaup\julia-1.12.6+0.x64.w64.mingw32\bin\julia.exe' --project=. test/test_cs_protocol.jl
```

Expected: nonzero exit. The selector-definition assertion and source assertions fail because the helper is absent and the plot file still calls `resolved_aefbfp_params` and legacy `build_problem`.

### Task 2: Implement the pure manuscript problem/start selector

**Files:**
- Modify: `scripts/compressed_sensing/s30_benchmark.jl` immediately after `build_manuscript_problem`
- Test: `test/test_cs_protocol.jl`

- [ ] **Step 1: Add the minimal selector**

```julia
"Build one definitive manuscript case and select its published initial point."
function manuscript_problem_start(case_id::AbstractString, start_index::Int;
                                  gamma::Float64 = GAMMA_REF)
    case_index = findfirst(case -> case.problem == case_id, DEFAULT_CASES)
    case_index === nothing && throw(ArgumentError("Unknown compressed-sensing case '$case_id'"))
    1 <= start_index <= DEFAULT_INITIAL_POINTS ||
        throw(ArgumentError("start_index must be in 1:$(DEFAULT_INITIAL_POINTS), got $start_index"))

    case = DEFAULT_CASES[case_index]
    prob = build_manuscript_problem(case;
                                    case_index = case_index,
                                    gamma = gamma,
                                    n_inits = DEFAULT_INITIAL_POINTS)
    return (prob = prob, init = prob.initial_points[start_index], case_index = case_index)
end
```

- [ ] **Step 2: Run the CS suite and inspect the partial GREEN state**

Run the Task-1 command again.

Expected: selector behavior assertions pass; the two source assertions remain failed until Task 3 patches the plotting file. This demonstrates that the numerical seam is correct independently of plotting.

### Task 3: Route all CS figure re-solves through the definitive protocol

**Files:**
- Modify: `scripts/compressed_sensing/s70_figures_tables.jl:82-115`
- Modify: `scripts/compressed_sensing/s70_figures_tables.jl:350-389`
- Modify: all three `solve_representative_case` call groups around lines 478, 561, and 656
- Modify: `scripts/compressed_sensing/s70_figures_tables.jl:694-715`
- Test: `test/test_cs_protocol.jl`

- [ ] **Step 1: Remove stale CLI configuration fields**

Keep only definitive fields in `read_report_config`:

```julia
function read_report_config(args)
    opts, flags = parse_cli(args)
    figures = haskey(opts, "figures") ? split_csv_items(opts["figures"]) : String[]
    png = "png" in flags
    production = !("quick" in flags)
    eps = haskey(opts, "eps") ? parse(Float64, opts["eps"]) : REPORT_EPS_DEFAULT
    maxiter = haskey(opts, "maxiter") ? parse(Int, opts["maxiter"]) : canonical_maxiter(eps)
    convergence_cases = haskey(opts, "convergence-cases") ?
        parse_case_list(opts["convergence-cases"]) : DEFAULT_CONVERGENCE_CASES
    signal_case = haskey(opts, "signal-case") ?
        only(parse_case_list(opts["signal-case"])) : DEFAULT_SIGNAL_CASE

    for fig in figures
        fig in ("convergence", "resolvent_convergence", "signal_panels") ||
            throw(ArgumentError("Unsupported figure key '$fig'"))
    end

    return (figures = figures, png = png, production = production,
            eps = eps, maxiter = maxiter,
            convergence_cases = convergence_cases, signal_case = signal_case)
end
```

- [ ] **Step 2: Replace algorithm and problem reconstruction**

Replace the helper block with:

```julia
function build_plot_algorithm(db, method::String)
    return build_algorithm(db, method)
end

function build_resolvent_plot_algorithm(db, method::String)
    method in RESOLVENT_METHOD_NAMES && return build_plot_algorithm(db, method)
    throw(ArgumentError("Unsupported resolvent plot method '$method'"))
end

function solve_representative_case(db, case_id::String, method::String;
                                   eps::Float64, maxiter::Int, dataset_idx::Int,
                                   gamma::Float64 = GAMMA_REF, consec::Int = 2,
                                   algorithm_builder::Function = build_plot_algorithm)
    selected = manuscript_problem_start(case_id, dataset_idx; gamma = gamma)
    prob = selected.prob
    init = selected.init
    alg = algorithm_builder(db, method)
    stopping = make_stopping(prob, eps, maxiter; consec = consec)
    hc = HistoryCallback()
    nhc = NativeResidualHistoryCallback(prob.native_residual)
    nrec = NativeResRecorder(prob.native_residual)
    result = solve(alg, prob, copy(init.x0);
                   stopping = stopping, observers = (hc, nhc, nrec))
    mse = reconstruction_mse(result.x, prob.metadata.x_star)
    return (prob = prob, init = init, result = result, history = hc.history,
            native_history_k = nhc.ks, native_history_elapsed = nhc.elapsed,
            native_history = nhc.values, native_residual = nrec.value, mse = mse)
end
```

- [ ] **Step 3: Remove stale keyword arguments from all call sites**

Each call must end after its true numerical inputs, for example:

```julia
rep = solve_representative_case(db, case_id, method;
                                eps = cfg.eps,
                                maxiter = cfg.maxiter,
                                dataset_idx = dataset_idx)
```

The resolvent call retains only its builder override:

```julia
rep = solve_representative_case(db, case_id, method;
                                eps = cfg.eps,
                                maxiter = cfg.maxiter,
                                dataset_idx = dataset_idx,
                                algorithm_builder = build_resolvent_plot_algorithm)
```

- [ ] **Step 4: Replace tuned-winner banner resolution**

At the start of `report_main`, do not query `tuned_winners`. Print the definitive protocol instead:

```julia
figures_label = isempty(cfg.figures) ? "(tables only)" : join(cfg.figures, ", ")
println(tee, "="^78)
println(tee, "  Figures/tables: $(PROBLEM_NAME)")
println(tee, "="^78)
println(tee, "  db_path    : $(DB_PATH)")
println(tee, "  figure_dir : $(FIGDIR)")
println(tee, "  figures    : $(figures_label)")
println(tee, "  protocol   : manuscript_v1")
println(tee, "  aefbfp_params : $(MANUSCRIPT_AEFBFP_PARAMS)")
println(tee)
```

- [ ] **Step 5: Run the CS suite and verify GREEN**

Run the Task-1 command.

Expected: all existing 24 assertions plus the new selector/source assertions pass; no plotting package is loaded.

### Task 4: Final plot-free verification and handoff

**Files:**
- Verify: `scripts/compressed_sensing/s30_benchmark.jl`
- Verify: `scripts/compressed_sensing/s70_figures_tables.jl`
- Verify: `test/test_cs_protocol.jl`

- [ ] **Step 1: Run the core suite**

```powershell
$env:JULIA_NUM_THREADS='1'
& 'C:\Users\mmogi\.julia\juliaup\julia-1.12.6+0.x64.w64.mingw32\bin\julia.exe' --project=. test/runtests.jl
```

Expected: all core reproducibility and projection assertions pass.

- [ ] **Step 2: Run static plotting-boundary checks**

```powershell
rg -n "manuscript_problem_start|resolved_aefbfp_params\(db;|prob = build_problem\(case;" scripts/compressed_sensing/s70_figures_tables.jl
rg -n "^\s*(using|import)\s+(Plots|LaTeXStrings)" src scripts/compressed_sensing/s30_benchmark.jl test/test_cs_protocol.jl
```

Expected: the first command shows only `manuscript_problem_start`; it shows neither stale call. The second command has no matches. `using Plots` and `using LaTeXStrings` remain only in the Mohammed-run figure entry point and are not loaded by these checks.

- [ ] **Step 3: Confirm output roots statically**

```powershell
rg -n "RESULT_ROOT|FIGDIR|LOGDIR|cs_signal_panels" scripts/compressed_sensing/s30_benchmark.jl scripts/compressed_sensing/s70_figures_tables.jl
```

Expected: `RESULT_ROOT` is rooted at `JCODE_ROOT/results/compressed_sensing`; `FIGDIR` and `LOGDIR` descend from it; signal panels use `joinpath(FIGDIR, "cs_signal_panels")`.

- [ ] **Step 4: Return the unchanged Mohammed command**

Do not execute it under Codex. Hand back:

```powershell
Set-Location 'D:\Dropbox\Research\Projects\IRC_MOIN\NewAlgorithm_EPCMMIP\jcode'
$env:JULIA_NUM_THREADS='1'
julia --project=. scripts/compressed_sensing/s70_figures_tables.jl --figures=signal_panels --png
```

Expected outputs remain:

```text
jcode/results/compressed_sensing/figures/cs_signal_panels.pdf
jcode/results/compressed_sensing/figures/cs_signal_panels.png
jcode/results/compressed_sensing/logs/log_s70_figures_tables_*.txt
```
