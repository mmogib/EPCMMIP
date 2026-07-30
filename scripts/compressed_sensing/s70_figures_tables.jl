# ============================================================================
# s70_figures_tables.jl
# ============================================================================
#
# Purpose
#   Build paper-facing tables and optional figures for the compressed-sensing
#   benchmark from the local per-case database.
#
# Outputs
#   - results/compressed_sensing/figures/tables.tex
#   - optional PDF/PNG figures selected through --figures
#   - results/compressed_sensing/logs/log_s70_figures_tables_*.txt
#
# Supported figure keys
#   - convergence
#   - resolvent_convergence
#   - signal_panels
#
# How to run
#   julia --project=. scripts/compressed_sensing/s70_figures_tables.jl
#   julia --project=. scripts/compressed_sensing/s70_figures_tables.jl --figures=convergence
#   julia --project=. scripts/compressed_sensing/s70_figures_tables.jl --figures=resolvent_convergence
#   julia --project=. scripts/compressed_sensing/s70_figures_tables.jl --figures=convergence,resolvent_convergence,signal_panels --png
#
# Step-by-step overview
#   Step 1. Load the benchmark helpers from `s30_benchmark.jl`.
#   Step 2. Read CLI options such as `--figures`, `--eps`, and `--png`.
#   Step 3. Load the benchmark rows from the local SQLite database.
#   Step 4. Write the summary LaTeX table (`tables.tex`).
#   Step 5. Re-solve one representative dataset per requested figure/case.
#   Step 6. Save the requested plots as PDF/PNG under `results/compressed_sensing/figures/`.
#
# ============================================================================

include(joinpath(@__DIR__, "s30_benchmark.jl"))

gr()

const DEFAULT_CONVERGENCE_CASES = [DEFAULT_CASES[1].problem, DEFAULT_CASES[end].problem]
const DEFAULT_SIGNAL_CASE = DEFAULT_CASES[3].problem
const DISPLAY_METHOD_NAMES = ["AEFBFP", "VAFBS", "MDITSM", "RFBSM", "IRFBSM", "IFRAB"]
const RESOLVENT_METHOD_NAMES = DISPLAY_METHOD_NAMES
const REPORT_EPS_DEFAULT = 1.0e-5
const SIGNAL_PANEL_YLIMS = (-1.0, 1.0)

# ============================================================================
# Step 1. Small helper callbacks and CLI parsing
# ============================================================================

mutable struct NativeResidualHistoryCallback{F} <: AbstractObserverCallback
    native_fn::F
    ks::Vector{Int}
    elapsed::Vector{Float64}
    values::Vector{Float64}
end

NativeResidualHistoryCallback(native_fn::F) where {F} =
    NativeResidualHistoryCallback{F}(native_fn, Int[], Float64[], Float64[])

function on_event!(cb::NativeResidualHistoryCallback, state::SolverState, event::Symbol)
    if event === :iter
        push!(cb.ks, state.k)
        push!(cb.elapsed, state.elapsed)
        push!(cb.values, cb.native_fn(state.x, state.x_prev))
    end
    return nothing
end

function split_csv_items(text::String)
    vals = String[]
    for part in split(text, ",")
        item = strip(part)
        isempty(item) && continue
        push!(vals, item)
    end
    return vals
end

function read_report_config(args)
    opts, flags = parse_cli(args)
    figures = haskey(opts, "figures") ? split_csv_items(opts["figures"]) : String[]
    png = "png" in flags
    production = !("quick" in flags)
    eps = haskey(opts, "eps") ? parse(Float64, opts["eps"]) : REPORT_EPS_DEFAULT
    maxiter = haskey(opts, "maxiter") ? parse(Int, opts["maxiter"]) : canonical_maxiter(eps)
    snr_db = haskey(opts, "snr-db") ? parse(Float64, opts["snr-db"]) : SNR_DB_REF
    convergence_cases = haskey(opts, "convergence-cases") ? parse_case_list(opts["convergence-cases"]) : DEFAULT_CONVERGENCE_CASES
    signal_case = haskey(opts, "signal-case") ? only(parse_case_list(opts["signal-case"])) : DEFAULT_SIGNAL_CASE
    aefbfp_preset = haskey(opts, "aefbfp-preset") ? parse_symbol_option(opts["aefbfp-preset"]) : nothing
    aefbfp_round_digits = haskey(opts, "aefbfp-round-digits") ? parse(Int, opts["aefbfp-round-digits"]) : nothing

    for fig in figures
        fig in ("convergence", "resolvent_convergence", "signal_panels") ||
            throw(ArgumentError("Unsupported figure key '$fig'"))
    end
    aefbfp_preset === nothing || haskey(AEFBFP_PRESETS, aefbfp_preset) ||
        throw(ArgumentError("Unknown AEFBFP preset :$(aefbfp_preset). Known: $(join(sort(collect(keys(AEFBFP_PRESETS))), ", "))"))
    aefbfp_round_digits === nothing || aefbfp_round_digits >= 0 ||
        throw(ArgumentError("aefbfp-round-digits must be >= 0, got $(aefbfp_round_digits)"))

    return (
        figures = figures,
        png = png,
        production = production,
        eps = eps,
        maxiter = maxiter,
        snr_db = snr_db,
        convergence_cases = convergence_cases,
        signal_case = signal_case,
        aefbfp_preset = aefbfp_preset,
        aefbfp_round_digits = aefbfp_round_digits,
    )
end

# ============================================================================
# Step 2. Table helpers
# ============================================================================

function load_table_rows(db, cfg)
    ensure_local_tables!(db)
    rows = load_result_rows(db, (
        production = cfg.production,
        eps = cfg.eps,
        maxiter = cfg.maxiter,
        cases = [case.problem for case in DEFAULT_CASES],
    ))
    metrics = DBInterface.execute(db, """
        SELECT config_hash, problem, dimension, init_point, production,
               objective, reconstruction_mse, common_residual
        FROM cs_final_metrics
    """) |> DataFrame
    return leftjoin(rows, metrics,
                    on = [:config_hash, :problem, :dimension, :init_point, :production])
end

function count_solved_runs(df, method::String, case_id::String)
    sub = df[(df.method .== method) .& (df.problem .== case_id), :]
    return (sum(sub.converged .== 1), nrow(sub))
end

function median_solved_metric(df, method::String, case_id::String, col::Symbol)
    sub = df[(df.method .== method) .& (df.problem .== case_id) .& (df.converged .== 1), :]
    nrow(sub) == 0 && return Inf
    values = collect(skipmissing(sub[!, col]))
    isempty(values) && return Inf
    return median(Float64.(values))
end

latex_escape(text::AbstractString) = replace(String(text), "_" => "\\_")
latex_bold(text::AbstractString) = "\\textbf{$(text)}"

case_header(case) = "M=$(case.M), N=$(case.N), k=$(case.k)"

function case_metric_summary(df, method::String, case_id::String)
    nconv, ntot = count_solved_runs(df, method, case_id)
    return (
        nconv = nconv,
        ntot = ntot,
        iter = median_solved_metric(df, method, case_id, :iterations),
        feval = median_solved_metric(df, method, case_id, :f_evals),
        cpu = median_solved_metric(df, method, case_id, :cpu_time),
        objective = median_solved_metric(df, method, case_id, :objective),
        mse = median_solved_metric(df, method, case_id, :reconstruction_mse),
        common_residual = median_solved_metric(df, method, case_id, :common_residual),
    )
end

function best_case_stats(df, case_id::String; methods = [name(T) for T in METHOD_TYPES])
    rows = [case_metric_summary(df, method, case_id) for method in methods]
    best_nconv = maximum(row.nconv for row in rows)
    best_iter = minimum(row.iter for row in rows if row.nconv == best_nconv && isfinite(row.iter); init = Inf)
    best_feval = minimum(row.feval for row in rows if row.nconv == best_nconv && isfinite(row.feval); init = Inf)
    best_cpu = minimum(row.cpu for row in rows if row.nconv == best_nconv && isfinite(row.cpu); init = Inf)
    return (nconv = best_nconv, iter = best_iter, feval = best_feval, cpu = best_cpu)
end

function format_iter_cell(stats, best)
    text = isfinite(stats.iter) ? @sprintf("%.1f", stats.iter) : "DNC"
    is_best = stats.nconv == best.nconv && isfinite(stats.iter) && isfinite(best.iter) && stats.iter == best.iter
    return is_best ? latex_bold(text) : text
end

function format_feval_cell(stats, best)
    text = isfinite(stats.feval) ? @sprintf("%.1f", stats.feval) : "DNC"
    is_best = stats.nconv == best.nconv && isfinite(stats.feval) && isfinite(best.feval) && stats.feval == best.feval
    return is_best ? latex_bold(text) : text
end

function format_cpu_cell(stats, best)
    text = isfinite(stats.cpu) ? @sprintf("%.3f", stats.cpu) : "DNC"
    is_best = stats.nconv == best.nconv && isfinite(stats.cpu) && isfinite(best.cpu) && stats.cpu == best.cpu
    return is_best ? latex_bold(text) : text
end

format_accuracy_cell(value::Real) = isfinite(value) ? @sprintf("%.2e", value) : "DNC"

function method_style_map(methods; variant::Symbol = :convergence)
    palette = if variant == :signal
        Dict(
            "AEFBFP" => (color = RGB(0.52, 0.08, 0.12), linestyle = :solid, marker = :none, lw = 1.8),
            "VAFBS"  => (color = RGB(0.00, 0.48, 0.54), linestyle = :solid, marker = :none, lw = 1.8),
            "MDITSM" => (color = RGB(0.08, 0.24, 0.47), linestyle = :solid, marker = :none, lw = 1.8),
            "RFBSM"  => (color = RGB(0.69, 0.32, 0.06), linestyle = :solid, marker = :none, lw = 1.8),
            "IRFBSM" => (color = RGB(0.07, 0.39, 0.21), linestyle = :solid, marker = :none, lw = 1.8),
            "IFRAB"  => (color = RGB(0.34, 0.16, 0.58), linestyle = :solid, marker = :none, lw = 1.8),
        )
    else
        Dict(
            "AEFBFP" => (color = RGB(0.0, 0.6056031704619725, 0.9786801190138923), linestyle = :solid, marker = :none, lw = 2.4),
            "VAFBS"  => (color = RGB(0.8888735440600661, 0.435649148506399, 0.2781230452972764), linestyle = :solid, marker = :none, lw = 2.4),
            "MDITSM" => (color = RGB(0.24222393333911896, 0.6432750821113586, 0.304448664188385), linestyle = :solid, marker = :none, lw = 2.4),
            "RFBSM"  => (color = RGB(0.7644400000572205, 0.44411176443099976, 0.8242975473403931), linestyle = :solid, marker = :none, lw = 2.4),
            "IRFBSM" => (color = RGB(0.6755439043045044, 0.555662214756012, 0.09423444420099258), linestyle = :solid, marker = :none, lw = 2.4),
            "IFRAB"  => (color = RGB(0.0, 0.6657590270042419, 0.6809969544410706), linestyle = :solid, marker = :none, lw = 2.4),
        )
    end
    fallback = (color = RGB(0.35, 0.35, 0.35), linestyle = :solid, marker = :none, lw = 2.4)
    return Dict(method => get(palette, method, fallback) for method in methods)
end

# Write the simple benchmark summary table used by the paper draft.
function write_summary_table_tex(df, path::String)
    methods = [name(T) for T in METHOD_TYPES]
    best_by_case = Dict(case.problem => best_case_stats(df, case.problem; methods = methods) for case in DEFAULT_CASES)
    open(path, "w") do io
        println(io, "% Auto-generated by compressed_sensing/s70_figures_tables.jl")
        println(io, "\\begin{table}[H]\\centering")
        println(io, "\\caption{Compressed-sensing benchmark summary. Each row reports one method. For every case, the columns show unrounded median iterations, unrounded median forward-operator evaluations, and median CPU time over ten initial points. Best values in each case are boldfaced.}\\label{tab:cs_summary}")
        println(io, "\\begin{tabular}{l" * repeat("rrr", length(DEFAULT_CASES)) * "}")
        println(io, "\\toprule")
        header_top = ["Method"]
        append!(header_top, ["\\multicolumn{3}{c}{$(case_header(case))}" for case in DEFAULT_CASES])
        println(io, join(header_top, " & ") * " \\\\")
        cmidrules = ["\\cmidrule(lr){2-4}", "\\cmidrule(lr){5-7}", "\\cmidrule(lr){8-10}", "\\cmidrule(lr){11-13}"]
        println(io, join(cmidrules, " "))
        header_bottom = [" "]
        for _ in DEFAULT_CASES
            append!(header_bottom, ["Median Iter", "Median F-Eval", "Median CPU"])
        end
        println(io, join(header_bottom, " & ") * " \\\\")
        println(io, "\\midrule")
        for method in methods
            row = [method]
            for case in DEFAULT_CASES
                stats = case_metric_summary(df, method, case.problem)
                best = best_by_case[case.problem]
                push!(row, format_iter_cell(stats, best))
                push!(row, format_feval_cell(stats, best))
                push!(row, format_cpu_cell(stats, best))
            end
            println(io, join(row, " & ") * " \\\\")
        end
        println(io, "\\bottomrule")
        println(io, "\\end{tabular}")
        println(io, "\\end{table}\n")

        println(io, "\\begin{table}[H]\\centering")
        println(io, "\\caption{Compressed-sensing final accuracy at the common LASSO optimality-residual stopping rule. For every case, the columns show the median final objective, reconstruction MSE, and common optimality residual over converged starts.}\\label{tab:cs_accuracy}")
        println(io, "\\begin{tabular}{l" * repeat("rrr", length(DEFAULT_CASES)) * "}")
        println(io, "\\toprule")
        header_top = ["Method"]
        append!(header_top, ["\\multicolumn{3}{c}{$(case_header(case))}" for case in DEFAULT_CASES])
        println(io, join(header_top, " & ") * " \\\\")
        cmidrules = ["\\cmidrule(lr){2-4}", "\\cmidrule(lr){5-7}", "\\cmidrule(lr){8-10}", "\\cmidrule(lr){11-13}"]
        println(io, join(cmidrules, " "))
        header_bottom = [" "]
        for _ in DEFAULT_CASES
            append!(header_bottom, ["Median Obj.", "Median MSE", "Median Res."])
        end
        println(io, join(header_bottom, " & ") * " \\\\")
        println(io, "\\midrule")
        for method in methods
            row = [method]
            for case in DEFAULT_CASES
                stats = case_metric_summary(df, method, case.problem)
                append!(row, [format_accuracy_cell(stats.objective),
                              format_accuracy_cell(stats.mse),
                              format_accuracy_cell(stats.common_residual)])
            end
            println(io, join(row, " & ") * " \\\\")
        end
        println(io, "\\bottomrule")
        println(io, "\\end{tabular}")
        println(io, "\\end{table}\n")
    end
end

# ============================================================================
# Step 3. Pick one representative dataset and rebuild the solved curves/signals
# ============================================================================

# Pick one dataset that is representative of the case-wide behaviour.
# Example: if several datasets solve, choose the one closest to the median iteration profile.
function pick_representative_dataset(df::DataFrame, case_id::String; methods = [name(T) for T in METHOD_TYPES])
    sub = df[df.problem .== case_id, :]
    isempty(sub) && throw(ArgumentError("No benchmark rows found for case=$case_id"))

    med_iters = Dict{String,Float64}()
    successful_methods = String[]
    for method in methods
        rs = sub[sub.method .== method, :]
        solved = rs[rs.converged .== 1, :]
        isempty(solved) && continue
        med_iters[method] = median(Float64.(solved.iterations))
        push!(successful_methods, method)
    end

    isempty(successful_methods) && return minimum(Int.(sub.dataset_idx))

    best_dataset = 0
    best_score = Inf
    for dataset_idx in sort(unique(Int.(sub.dataset_idx)))
        rs = sub[sub.dataset_idx .== dataset_idx, :]
        score = 0.0
        feasible = true
        for method in successful_methods
            row = rs[rs.method .== method, :]
            if nrow(row) != 1 || Int(row.converged[1]) != 1
                feasible = false
                break
            end
            score += abs(Float64(row.iterations[1]) - med_iters[method]) / med_iters[method]
        end
        feasible || continue
        if score < best_score
            best_score = score
            best_dataset = dataset_idx
        end
    end

    if best_dataset > 0
        return best_dataset
    end

    ref_method = "AEFBFP" in successful_methods ? "AEFBFP" : first(successful_methods)
    ref_rows = sub[(sub.method .== ref_method) .& (sub.converged .== 1), :]
    if nrow(ref_rows) > 0
        target = median(Float64.(ref_rows.iterations))
        scores = [(abs(Float64(r.iterations) - target), Int(r.dataset_idx)) for r in eachrow(ref_rows)]
        sort!(scores, by = first)
        return scores[1][2]
    end

    return minimum(Int.(sub.dataset_idx))
end

# Solve one stored case/method pair again so the plotting script can rebuild
# the convergence curve and final recovered signal from a chosen dataset.
function build_plot_algorithm(db, method::String; aefbfp_preset::Union{Nothing,Symbol} = nothing,
                              aefbfp_round_digits::Union{Nothing,Int} = nothing)
    return build_algorithm(db, method;
                           allow_untuned_aefbfp = false,
                           aefbfp_preset = aefbfp_preset,
                           aefbfp_round_digits = aefbfp_round_digits)
end

function build_resolvent_plot_algorithm(db, method::String; aefbfp_preset::Union{Nothing,Symbol} = nothing,
                                        aefbfp_round_digits::Union{Nothing,Int} = nothing)
    method in RESOLVENT_METHOD_NAMES &&
        return build_plot_algorithm(db, method; aefbfp_preset = aefbfp_preset, aefbfp_round_digits = aefbfp_round_digits)
    throw(ArgumentError("Unsupported resolvent plot method '$method'"))
end

function solve_representative_case(db, case_id::String, method::String; eps::Float64, maxiter::Int, dataset_idx::Int,
                                   gamma::Float64 = GAMMA_REF, snr_db::Float64 = SNR_DB_REF, consec::Int = 2,
                                   algorithm_builder::Function = build_plot_algorithm,
                                   aefbfp_preset::Union{Nothing,Symbol} = nothing,
                                   aefbfp_round_digits::Union{Nothing,Int} = nothing)
    case = CASE_BY_NAME[case_id]
    prob = build_problem(case; gamma = gamma, snr_db = snr_db, data_seed = dataset_seed(case, dataset_idx))
    init = prob.initial_points[1]
    alg = algorithm_builder(db, method; aefbfp_preset = aefbfp_preset, aefbfp_round_digits = aefbfp_round_digits)
    stopping = make_stopping(prob, eps, maxiter; consec = consec)
    hc = HistoryCallback()
    nhc = NativeResidualHistoryCallback(prob.native_residual)
    nrec = NativeResRecorder(prob.native_residual)
    result = solve(alg, prob, copy(init.x0); stopping = stopping, observers = (hc, nhc, nrec))
    mse = reconstruction_mse(result.x, prob.metadata.x_star)
    return (
        prob = prob,
        init = init,
        result = result,
        history = hc.history,
        native_history_k = nhc.ks,
        native_history_elapsed = nhc.elapsed,
        native_history = nhc.values,
        native_residual = nrec.value,
        mse = mse,
    )
end

resolvent_residual_series(history) = [0.5 * max(rec.residual, 0.0)^2 for rec in history]

# Save one plot stem as PDF and, optionally, PNG.
function save_plot_files(plt, stem::String, tee; png::Bool)
    pdf_path = stem * ".pdf"
    try
        savefig(plt, pdf_path)
        println(tee, "  wrote $(pdf_path)")
    catch err
        println(tee, "  warning: could not write $(pdf_path) ($(sprint(showerror, err)))")
    end
    if png
        png_path = stem * ".png"
        savefig(plt, png_path)
        println(tee, "  wrote $(png_path)")
    end
    return nothing
end

function convergence_output_stem(case_id::String)
    return joinpath(FIGDIR, "cs_convergence_" * lowercase(case_id))
end

function convergence_cpu_output_stem(case_id::String)
    return joinpath(FIGDIR, "cs_convergence_cpu_" * lowercase(case_id))
end

function resolvent_convergence_output_stem(case_id::String)
    return joinpath(FIGDIR, "cs_resolvent_convergence_" * lowercase(case_id))
end

function case_title_with_capital_k(case)
    return "M=$(case.M), N=$(case.N), K=$(case.k)"
end

# ============================================================================
# Step 4. Build convergence figures
# ============================================================================

function build_convergence_figures(db, df, cfg, tee)
    methods = DISPLAY_METHOD_NAMES
    styles = method_style_map(methods; variant = :convergence)

    for case_id in cfg.convergence_cases
        dataset_idx = pick_representative_dataset(df, case_id; methods = methods)
        println(tee, "  convergence case $(case_id): representative dataset=$(dataset_idx), native_tol=$(cfg.eps), maxiter=$(cfg.maxiter)")

        plt = plot(layout = (1, 1),
                   size = (900, 620),
                   dpi = 220,
                   background_color = :white,
                   background_color_inside = :white,
                   foreground_color_subplot = :black,
                   foreground_color_legend = :black,
                   background_color_legend = :white,
                   legend_font_pointsize = 11,
                   guidefontsize = 12,
                   tickfontsize = 10,
                   titlefontsize = 12,
                   left_margin = 16Plots.mm,
                   right_margin = 10Plots.mm,
                   bottom_margin = 12Plots.mm,
                   top_margin = 8Plots.mm)
        cpu_plt = plot(layout = (1, 1),
                       size = (900, 620),
                       dpi = 220,
                       background_color = :white,
                       background_color_inside = :white,
                       foreground_color_subplot = :black,
                       foreground_color_legend = :black,
                       background_color_legend = :white,
                       legend_font_pointsize = 11,
                       guidefontsize = 12,
                       tickfontsize = 10,
                       titlefontsize = 12,
                       left_margin = 16Plots.mm,
                       right_margin = 10Plots.mm,
                       bottom_margin = 12Plots.mm,
                       top_margin = 8Plots.mm)
        max_k = 0
        max_t = 0.0

        for method in methods
            rep = solve_representative_case(db, case_id, method;
                                            eps = cfg.eps,
                                            maxiter = cfg.maxiter,
                                            snr_db = cfg.snr_db,
                                            dataset_idx = dataset_idx,
                                            aefbfp_preset = cfg.aefbfp_preset,
                                            aefbfp_round_digits = cfg.aefbfp_round_digits)
            ks = rep.native_history_k
            ts = rep.native_history_elapsed
            rs = [max(v, 1.0e-12) for v in rep.native_history]
            isempty(ks) || (max_k = max(max_k, maximum(ks)))
            isempty(ts) || (max_t = max(max_t, maximum(ts)))
            sty = styles[method]
            plot!(plt[1], ks, rs;
                  yscale = :log10,
                  label = method,
                  lw = sty.lw,
                  color = sty.color,
                  ls = sty.linestyle)
            plot!(cpu_plt[1], ts, rs;
                  yscale = :log10,
                  label = method,
                  lw = sty.lw,
                  color = sty.color,
                  ls = sty.linestyle)
        end

        plot!(plt[1];
              xlabel = "Iteration",
              ylabel = L"\mathcal{E}_n",
              xlims = (0, max(1, max_k)),
              ylims = (1.0e-5, 1.0e-1),
              grid = true,
              gridalpha = 0.18,
              gridlinewidth = 0.9,
              gridstyle = :solid,
              framestyle = :box,
              widen = false,
              legend = :topright)
        plot!(cpu_plt[1];
              xlabel = "CPU time (s)",
              ylabel = L"\mathcal{E}_n",
              xlims = (0.0, max(1.0e-9, max_t)),
              ylims = (1.0e-5, 1.0e-1),
              grid = true,
              gridalpha = 0.18,
              gridlinewidth = 0.9,
              gridstyle = :solid,
              framestyle = :box,
              widen = false,
              legend = :topright)
        save_plot_files(plt, convergence_output_stem(case_id), tee; png = cfg.png)
        save_plot_files(cpu_plt, convergence_cpu_output_stem(case_id), tee; png = cfg.png)
    end
end

function build_resolvent_convergence_figures(db, df, cfg, tee)
    methods = RESOLVENT_METHOD_NAMES
    styles = method_style_map(methods; variant = :convergence)

    for case_id in cfg.convergence_cases
        dataset_idx = pick_representative_dataset(df, case_id; methods = methods)
        println(tee, "  resolvent-convergence case $(case_id): representative dataset=$(dataset_idx), native_tol=$(cfg.eps), maxiter=$(cfg.maxiter)")

        plt = plot(layout = (1, 1),
                   size = (900, 620),
                   dpi = 220,
                   background_color = :white,
                   background_color_inside = :white,
                   foreground_color_subplot = :black,
                   foreground_color_legend = :black,
                   background_color_legend = :white,
                   legend_font_pointsize = 11,
                   guidefontsize = 12,
                   tickfontsize = 10,
                   titlefontsize = 12,
                   left_margin = 16Plots.mm,
                   right_margin = 10Plots.mm,
                   bottom_margin = 12Plots.mm,
                   top_margin = 8Plots.mm)
        max_k = 0

        for method in methods
            rep = solve_representative_case(db, case_id, method;
                                            eps = cfg.eps,
                                            maxiter = cfg.maxiter,
                                            snr_db = cfg.snr_db,
                                            dataset_idx = dataset_idx,
                                            algorithm_builder = build_resolvent_plot_algorithm,
                                            aefbfp_preset = cfg.aefbfp_preset,
                                            aefbfp_round_digits = cfg.aefbfp_round_digits)
            ks = [rec.k for rec in rep.history]
            rs = [max(v, 1.0e-16) for v in resolvent_residual_series(rep.history)]
            isempty(ks) || (max_k = max(max_k, maximum(ks)))
            sty = styles[method]
            plot!(plt[1], ks, rs;
                  yscale = :log10,
                  label = method,
                  lw = sty.lw,
                  color = sty.color,
                  ls = sty.linestyle)
        end

        plot!(plt[1];
              xlabel = "Iteration",
              ylabel = L"\mathcal{R}_n",
              xlims = (0, max(1, max_k)),
              ylims = (1.0e-6, 1.0e0),
              grid = true,
              gridalpha = 0.18,
              gridlinewidth = 0.9,
              gridstyle = :solid,
              framestyle = :box,
              widen = false,
              legend = :topright)
        save_plot_files(plt, resolvent_convergence_output_stem(case_id), tee; png = cfg.png)
    end
end

# ============================================================================
# Step 5. Build signal-panel figures
# ============================================================================

# Plot a sparse vector using sticks so spikes are easy to compare.
# Example use: original signal or a recovered signal panel.
function plot_spike_panel!(panel, values::Vector{Float64}, title::String; color, ylabel::String = "")
    plot!(panel, 1:length(values), values;
          seriestype = :sticks,
          color = color,
          linewidth = 0.8,
          marker = (:circle, 2.1),
          markerstrokewidth = 0.35,
          markerstrokecolor = color,
          markercolor = :white,
          label = false)
    plot!(panel;
          xlabel = "Index",
          ylabel = ylabel,
          xlims = (0, max(1, length(values))),
          ylims = SIGNAL_PANEL_YLIMS,
          title = title,
          framestyle = :box,
          widen = false)
end

# Plot noisy measurements using the same spike-like view as the signal panels.
# Example use: the noisy-measurement panel placed under the original signal.
function plot_measurement_panel!(panel, values::Vector{Float64}, title::String)
    plot!(panel, 1:length(values), values;
          seriestype = :sticks,
          color = RGB(0.25, 0.25, 0.25),
          linewidth = 0.8,
          marker = (:circle, 1.9),
          markercolor = :white,
          markerstrokecolor = RGB(0.25, 0.25, 0.25),
          markerstrokewidth = 0.3,
          label = false)
    plot!(panel;
          xlabel = "Measurement index",
          ylabel = "Value",
          xlims = (0, max(1, length(values))),
          ylims = SIGNAL_PANEL_YLIMS,
          title = title,
          framestyle = :box,
          widen = false)
end

# Build the long vertical signal-panel figure:
#   1. original signal
#   2. noisy measurements
#   3. one recovered-signal panel per displayed method
function build_signal_panels_figure(db, df, cfg, tee)
    case_id = cfg.signal_case
    dataset_idx = pick_representative_dataset(df, case_id; methods = DISPLAY_METHOD_NAMES)
    println(tee, "  signal-panel case $(case_id): representative dataset=$(dataset_idx)")

    reps = Dict{String,NamedTuple}()
    for method in DISPLAY_METHOD_NAMES
        reps[method] = solve_representative_case(db, case_id, method;
                                                 eps = cfg.eps,
                                                 maxiter = cfg.maxiter,
                                                 snr_db = cfg.snr_db,
                                                 dataset_idx = dataset_idx,
                                                 aefbfp_preset = cfg.aefbfp_preset,
                                                 aefbfp_round_digits = cfg.aefbfp_round_digits)
    end

    ref_method = "AEFBFP"
    prob = reps[ref_method].prob
    methods = DISPLAY_METHOD_NAMES
    styles = method_style_map(methods; variant = :signal)
    case = CASE_BY_NAME[case_id]
    plt = plot(layout = (2 + length(methods), 1), size = (1180, 260 * (2 + length(methods))))

    plot_spike_panel!(plt[1], prob.metadata.x_star,
                      "(a) Original signal | $(case_title_with_capital_k(case))";
                      color = styles[ref_method].color, ylabel = "Amplitude")
    plot_measurement_panel!(plt[2], prob.metadata.y,
                            "(b) Noisy measurements | y = Ax* + ε, ε ~ N(0, 10⁻⁴I)")

    letters = collect('c':'z')
    for (panel_idx, method, letter) in zip(3:(2 + length(methods)), methods, letters)
        rep = reps[method]
        cpu_text = @sprintf("%.2fs", rep.result.cpu_time)
        mse_text = @sprintf("%.2e", rep.mse)
        title = "($(letter)) $(method) | Iter $(rep.result.iterations) | MSE $(mse_text) | CPU $(cpu_text)"
        plot_spike_panel!(plt[panel_idx], rep.result.x, title; color = styles[method].color)
    end

    save_plot_files(plt, joinpath(FIGDIR, "cs_signal_panels"), tee; png = cfg.png)
end

# ============================================================================
# Step 6. Main entry point
# ============================================================================

function report_main(args = ARGS)
    cfg = read_report_config(args)

    logpath, tee, _ = setup_logging("s70_figures_tables"; logdir = LOGDIR)
    db = open_db(DB_PATH)
    try
        figures_label = isempty(cfg.figures) ? "(tables only)" : join(cfg.figures, ", ")
        aefbfp_label = cfg.aefbfp_preset === nothing ? "(tuned winner for re-solves)" : string(cfg.aefbfp_preset)
        aefbfp_round_label = cfg.aefbfp_round_digits === nothing ? "(full precision)" : string(cfg.aefbfp_round_digits)
        aefbfp_params = resolved_aefbfp_params(db;
                                               aefbfp_preset = cfg.aefbfp_preset,
                                               round_digits = cfg.aefbfp_round_digits)
        println(tee, "="^78)
        println(tee, "  Figures/tables: $(PROBLEM_NAME)")
        println(tee, "="^78)
        println(tee, "  db_path    : $(DB_PATH)")
        println(tee, "  figure_dir : $(FIGDIR)")
        println(tee, "  figures    : $(figures_label)")
        println(tee, "  aefbfp_preset : $(aefbfp_label)")
        println(tee, "  aefbfp_round_digits : $(aefbfp_round_label)")
        println(tee, "  aefbfp_params : $(aefbfp_params)")
        println(tee)

        df = load_table_rows(db, cfg)
        nrow(df) == 0 && error("No benchmark rows found. Run s30_benchmark.jl first.")

        tables_path = joinpath(FIGDIR, "tables.tex")
        write_summary_table_tex(df, tables_path)
        println(tee, "  wrote $(tables_path)")

        for fig in cfg.figures
            if fig == "convergence"
                build_convergence_figures(db, df, cfg, tee)
            elseif fig == "resolvent_convergence"
                build_resolvent_convergence_figures(db, df, cfg, tee)
            elseif fig == "signal_panels"
                build_signal_panels_figure(db, df, cfg, tee)
            end
        end
    finally
        teardown_logging(tee, logpath)
        SQLite.close(db)
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    report_main()
end
