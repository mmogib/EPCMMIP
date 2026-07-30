# ============================================================================
# s50_tolerance_profile.jl
# ============================================================================
#
# Purpose
#   Independently evaluate tolerance-dependent performance for the compressed-
#   sensing benchmark.  This script does not modify the s30 benchmark database
#   or any existing figures/tables.  Every run writes a new timestamped folder
#   under results/compressed_sensing/tolerance_profile/.
#
# Metrics (one matrix per metric)
#   1. Iterations to a sustained residual target.
#   2. Forward-operator evaluations to that target.
#   3. CPU time to that target.
#   4. Success rate at that target.
#
# Usage
#   julia --project=. scripts/compressed_sensing/s50_tolerance_profile.jl
#   julia --project=. scripts/compressed_sensing/s50_tolerance_profile.jl --quick
#   julia --project=. scripts/compressed_sensing/s50_tolerance_profile.jl --datasets=5 --maxiter=30000
#   julia --project=. scripts/compressed_sensing/s50_tolerance_profile.jl --cases=M256_N512_k30,M512_N1024_k80
#
# The default profile covers residual tolerances 1e-2, ..., 1e-9.  A target is
# counted as reached only after two consecutive iterates satisfy it, matching
# the benchmark stopping convention.
# ============================================================================

include(joinpath(@__DIR__, "s30_benchmark.jl"))

const PROFILE_TOLERANCES = 10.0 .^ (-2:-1:-9)
const PROFILE_METHODS = [
    "EPCM", "EFBFP", "AEFBFP", "MAEFBFP",
    "VAFBS", "MDITSM", "MFRBSM", "RFBSM", "IRFBSM",
    "MTTM", "IMTTM", "IFRAB", "SFRBM",
]
const PROFILE_ROOT = joinpath(RESULT_ROOT, "tolerance_profile")
const PROFILE_MAXITER_DEFAULT = 20_000
const PROFILE_CONSEC = 2

mutable struct ToleranceProfileCallback{F} <: AbstractObserverCallback
    native_fn::F
    ks::Vector{Int}
    f_evals::Vector{Int}
    elapsed::Vector{Float64}
    residuals::Vector{Float64}
end

ToleranceProfileCallback(native_fn::F) where {F} =
    ToleranceProfileCallback{F}(native_fn, Int[], Int[], Float64[], Float64[])

function split_csv_items(text::String)
    items = String[]
    for part in split(text, ",")
        item = strip(part)
        isempty(item) || push!(items, item)
    end
    return items
end

function on_event!(cb::ToleranceProfileCallback, state::SolverState, event::Symbol)
    if event === :iter
        push!(cb.ks, state.k)
        push!(cb.f_evals, state.f_evals)
        push!(cb.elapsed, state.elapsed)
        push!(cb.residuals, cb.native_fn(state.x, state.x_prev))
    end
    return nothing
end

function parse_profile_cases(text::String)
    cases = split_csv_items(text)
    for case_id in cases
        haskey(CASE_BY_NAME, case_id) || throw(ArgumentError("Unknown case '$case_id'"))
    end
    return cases
end

function parse_profile_methods(text::String)
    methods = split_csv_items(text)
    for method in methods
        method in PROFILE_METHODS || throw(ArgumentError("Unsupported profile method '$method'"))
    end
    return methods
end

function read_profile_config(args)
    opts, flags = parse_cli(args)
    quick = "quick" in flags
    datasets = haskey(opts, "datasets") ? parse(Int, opts["datasets"]) : (quick ? 2 : DEFAULT_DATASETS)
    maxiter = haskey(opts, "maxiter") ? parse(Int, opts["maxiter"]) : PROFILE_MAXITER_DEFAULT
    cases = haskey(opts, "cases") ? parse_profile_cases(opts["cases"]) : [case.problem for case in DEFAULT_CASES]
    methods = haskey(opts, "methods") ? parse_profile_methods(opts["methods"]) : copy(PROFILE_METHODS)
    aefbfp_preset = haskey(opts, "aefbfp-preset") ? parse_symbol_option(opts["aefbfp-preset"]) : nothing
    aefbfp_round_digits = haskey(opts, "aefbfp-round-digits") ? parse(Int, opts["aefbfp-round-digits"]) : nothing
    1 <= datasets <= DEFAULT_DATASETS || throw(ArgumentError("datasets must be in 1:$(DEFAULT_DATASETS)"))
    maxiter > 0 || throw(ArgumentError("maxiter must be positive"))
    return (
        quick = quick,
        datasets = datasets,
        maxiter = maxiter,
        cases = cases,
        methods = methods,
        aefbfp_preset = aefbfp_preset,
        aefbfp_round_digits = aefbfp_round_digits,
    )
end

function first_sustained_hit(cb::ToleranceProfileCallback, tol::Float64; consec::Int = PROFILE_CONSEC)
    streak = 0
    for i in eachindex(cb.residuals)
        if isfinite(cb.residuals[i]) && cb.residuals[i] <= tol
            streak += 1
            streak >= consec && return i
        else
            streak = 0
        end
    end
    return nothing
end

# `:P4` is the repository's LASSO/compressed-sensing preset for the three
# proposed-method variants.  The remaining additions use their repository
# source-paper presets, exactly as the existing competitor factory does.
function build_profile_algorithm(db, method::String, cfg)
    method == "EPCM"    && return EPCM(:P4)
    method == "EFBFP"   && return EFBFP(:P4)
    method == "MAEFBFP" && return MAEFBFP(:P4)
    method == "MTTM"    && return MTTM(:paper)
    return build_algorithm(db, method;
                           aefbfp_preset = cfg.aefbfp_preset,
                           aefbfp_round_digits = cfg.aefbfp_round_digits)
end

function profile_run(db, case_id::String, method::String, dataset_idx::Int, cfg)
    case = CASE_BY_NAME[case_id]
    prob = build_problem(case;
                         gamma = GAMMA_REF,
                         snr_db = SNR_DB_REF,
                         data_seed = dataset_seed(case, dataset_idx),
                         n_inits = 1)
    init = only(prob.initial_points)
    alg = build_profile_algorithm(db, method, cfg)
    callback = ToleranceProfileCallback(prob.native_residual)
    stopping = make_stopping(prob, minimum(PROFILE_TOLERANCES), cfg.maxiter; consec = PROFILE_CONSEC)
    result = solve(alg, prob, copy(init.x0); stopping = stopping, observers = (callback,))
    return callback, result
end

csv_cell(x) = x === missing ? "" : string(x)

function write_csv(path::String, rows::Vector{<:NamedTuple})
    isempty(rows) && error("Cannot write an empty CSV: $path")
    isfile(path) && error("Refusing to overwrite existing file: $path")
    names = collect(keys(first(rows)))
    open(path, "w") do io
        println(io, join(string.(names), ","))
        for row in rows
            println(io, join((csv_cell(getfield(row, name)) for name in names), ","))
        end
    end
end

function median_or_missing(values)
    clean = Float64[x for x in values if x !== missing]
    return isempty(clean) ? missing : median(clean)
end

function build_summary_rows(run_rows)
    summary = NamedTuple[]
    for case_id in unique(row.case for row in run_rows), tol in PROFILE_TOLERANCES, method in unique(row.method for row in run_rows)
        sub = [row for row in run_rows if row.case == case_id && row.tolerance == tol && row.method == method]
        hits = [row for row in sub if row.hit]
        push!(summary, (
            case = case_id,
            tolerance = tol,
            method = method,
            n_hits = length(hits),
            n_total = length(sub),
            success_rate = isempty(sub) ? 0.0 : length(hits) / length(sub),
            median_iterations = median_or_missing(row.iterations for row in hits),
            median_f_evals = median_or_missing(row.f_evals for row in hits),
            median_cpu_time = median_or_missing(row.cpu_time for row in hits),
        ))
    end
    return summary
end

function build_metric_matrix(summary_rows, metric::Symbol, methods)
    rows = NamedTuple[]
    for case_id in unique(row.case for row in summary_rows), tol in PROFILE_TOLERANCES
        values = Dict(row.method => getfield(row, metric) for row in summary_rows if row.case == case_id && row.tolerance == tol)
        pairs = Pair{Symbol,Any}[:case => case_id, :tolerance => tol]
        append!(pairs, Symbol(method) => get(values, method, missing) for method in methods)
        push!(rows, (; pairs...))
    end
    return rows
end

function build_winner_rows(summary_rows, methods)
    rows = NamedTuple[]
    metrics = (:median_iterations, :median_f_evals, :median_cpu_time)
    for case_id in unique(row.case for row in summary_rows), tol in PROFILE_TOLERANCES
        sub = [row for row in summary_rows if row.case == case_id && row.tolerance == tol]
        for metric in metrics
            eligible = [row for row in sub if getfield(row, metric) !== missing && row.success_rate == 1.0]
            if isempty(eligible)
                push!(rows, (case = case_id, tolerance = tol, metric = string(metric), winner = "DNC", best_value = missing))
            else
                best = minimum(Float64(getfield(row, metric)) for row in eligible)
                winners = join([row.method for row in eligible if Float64(getfield(row, metric)) == best], "+")
                push!(rows, (case = case_id, tolerance = tol, metric = string(metric), winner = winners, best_value = best))
            end
        end
        rates = Dict(row.method => row.success_rate for row in sub)
        best_rate = maximum(values(rates))
        winners = join([method for method in methods if get(rates, method, 0.0) == best_rate], "+")
        push!(rows, (case = case_id, tolerance = tol, metric = "success_rate", winner = winners, best_value = best_rate))
    end
    return rows
end

function print_profile_summary(tee, winner_rows)
    println(tee, "\nWinners by tolerance and metric (full-success methods for work metrics):")
    for row in winner_rows
        @printf(tee, "  %-16s tol=%7.1e  %-22s %s\n", row.case, row.tolerance, row.metric, row.winner)
    end
end

function main(args = ARGS)
    cfg = read_profile_config(args)
    run_id = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
    outdir = joinpath(PROFILE_ROOT, "run_" * run_id)
    mkpath(outdir)
    logdir = joinpath(PROFILE_ROOT, "logs")
    mkpath(logdir)
    logpath, tee, _ = setup_logging("s50_tolerance_profile"; logdir = logdir)
    db = open_db(DB_PATH)

    try
        println(tee, "="^78)
        println(tee, "  compressed-sensing tolerance profile")
        println(tee, "="^78)
        println(tee, "  output dir : $(outdir)")
        println(tee, "  tolerances : $(join(string.(PROFILE_TOLERANCES), ", "))")
        println(tee, "  cases      : $(join(cfg.cases, ", "))")
        println(tee, "  methods    : $(join(cfg.methods, ", "))")
        println(tee, "  datasets   : $(cfg.datasets)")
        println(tee, "  maxiter    : $(cfg.maxiter)")

        run_rows = NamedTuple[]
        for case_id in cfg.cases, method in cfg.methods, dataset_idx in 1:cfg.datasets
            println(tee, "  $(case_id) | $(method) | dataset $(dataset_idx)")
            try
                callback, result = profile_run(db, case_id, method, dataset_idx, cfg)
                for tol in PROFILE_TOLERANCES
                    hit_idx = first_sustained_hit(callback, tol)
                    if hit_idx === nothing
                        push!(run_rows, (case = case_id, dataset = dataset_idx, method = method, tolerance = tol,
                                         hit = false, iterations = missing, f_evals = missing, cpu_time = missing,
                                         final_residual = result.residual, flag = string(result.flag)))
                    else
                        push!(run_rows, (case = case_id, dataset = dataset_idx, method = method, tolerance = tol,
                                         hit = true, iterations = callback.ks[hit_idx], f_evals = callback.f_evals[hit_idx],
                                         cpu_time = callback.elapsed[hit_idx], final_residual = result.residual,
                                         flag = string(result.flag)))
                    end
                end
            catch err
                println(tee, "    failed: $(sprint(showerror, err))")
                for tol in PROFILE_TOLERANCES
                    push!(run_rows, (case = case_id, dataset = dataset_idx, method = method, tolerance = tol,
                                     hit = false, iterations = missing, f_evals = missing, cpu_time = missing,
                                     final_residual = missing, flag = "error"))
                end
            end
        end

        summary_rows = build_summary_rows(run_rows)
        winner_rows = build_winner_rows(summary_rows, cfg.methods)
        write_csv(joinpath(outdir, "run_level.csv"), run_rows)
        write_csv(joinpath(outdir, "summary.csv"), summary_rows)
        write_csv(joinpath(outdir, "iterations_matrix.csv"), build_metric_matrix(summary_rows, :median_iterations, cfg.methods))
        write_csv(joinpath(outdir, "f_evals_matrix.csv"), build_metric_matrix(summary_rows, :median_f_evals, cfg.methods))
        write_csv(joinpath(outdir, "cpu_time_matrix.csv"), build_metric_matrix(summary_rows, :median_cpu_time, cfg.methods))
        write_csv(joinpath(outdir, "success_rate_matrix.csv"), build_metric_matrix(summary_rows, :success_rate, cfg.methods))
        write_csv(joinpath(outdir, "winners.csv"), winner_rows)
        print_profile_summary(tee, winner_rows)
    finally
        teardown_logging(tee, logpath)
        SQLite.close(db)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
