# ============================================================================
# s70: Figures and tables for the harmonic-oscillator benchmark
# ============================================================================
#
# Goal:
#   Read the stored s30 benchmark rows for the local harmonic-oscillator
#   problem and turn them into the paper/report outputs.
#
# What this script reads:
#   - results/optimal_control/harmonic_oscillator/experiments.db
#   - table `results`   : benchmark summaries written by s30
#
# What this script writes:
#   - results/optimal_control/harmonic_oscillator/figures/tables.tex
#   - results/optimal_control/harmonic_oscillator/figures/ho_convergence.pdf
#   - results/optimal_control/harmonic_oscillator/figures/ho_control.pdf
#   - results/optimal_control/harmonic_oscillator/figures/ho_state.pdf
#   - optional PNG copies when `--png` is used
#
# Simple workflow:
#   Before Section 1:
#     Load shared code, load the local problem definition, define paths, method
#     lists, and the plotting styles/colors that will be reused below.
#
#   Section 1:
#     Read the report configuration from ARGS and load the benchmark rows from
#     the local experiments database.
#
#   Section 2:
#     Open logging and print a short banner so the run records which tolerance,
#     maxiter, production tier, and output folder were used.
#
#   Section 3:
#     Build the LaTeX summary table `tables.tex` from the stored benchmark rows.
#
#   Section 4:
#     Re-run the representative reference seed for each method and build the
#     convergence plot.
#
#   Section 5:
#     Re-run one representative AEFBFP solve at the reference mesh/seed and
#     build the control/state figures.
#
# Notes:
#   - This script does NOT run the full benchmark again. It reads the stored
#     benchmark rows and only re-runs the small representative figure solves.
#   - The chosen plot colors/styles below are intentional and should stay fixed
#     unless we explicitly decide to redesign the figures.

include(joinpath(@__DIR__, "..", "..", "..", "src", "includes.jl"))
include(joinpath(@__DIR__, "problem_definition.jl"))
include(joinpath(@__DIR__, "..", "manuscript_protocol.jl"))

using Plots
using LaTeXStrings

gr()

# ============================================================================
# Before Section 1: shared setup reused by the local s20 / s30 / s70 workflow
# ============================================================================
# What is here:
#   - method lists
#   - DB/path helpers
#   - plot style constants
#   - shared parsers
#   - AEFBFP builder helpers

const OC_METHOD_TYPES = (AEFBFP, VAFBS, MDITSM, RFBSM, IRFBSM, IFRAB)
const OC_METHOD_BY_NAME = Dict(name(T) => T for T in OC_METHOD_TYPES)
const CS_RESULT_ROOT = joinpath(JCODE_ROOT, "results", "compressed_sensing")
const CS_DB_PATH = joinpath(CS_RESULT_ROOT, "experiments.db")
const OC_SHARED_AEFBFP_PARAMS = OC_MANUSCRIPT_AEFBFP_PARAMS

const OC_DEFAULT_DIMS = [50, 100, 200]   # full benchmark mesh sizes
const OC_REF_DIM = 100                   # single representative mesh used for history/figures
const OC_REF_INIT = 1
const OC_DEFAULT_CANDIDATES = 20
const OC_SEARCH_SEEDS = 5
const OC_BENCH_INITS = 10
const OC_EPS_REF = 1.0e-5
const OC_NMAX_REF = 4000
const HO_TIME_TICKS = (
    collect(0.0:pi / 2:3pi),
    ["0", "π/2", "π", "3π/2", "2π", "5π/2", "3π"],
)

const OC_WINNER_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS tuned_winners (
    method TEXT NOT NULL,
    problem TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    summary_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (method, problem)
)
"""

const OC_CONVERGENCE_STYLES = Dict(
    "AEFBFP" => (color = RGB(0.0, 0.35, 0.70), lw = 2.4),
    "VAFBS"  => (color = RGB(0.75, 0.20, 0.10), lw = 2.4),
    "MDITSM" => (color = RGB(0.0, 0.45, 0.20), lw = 2.4),
    "RFBSM"  => (color = RGB(0.45, 0.20, 0.65), lw = 2.4),
    "IRFBSM" => (color = RGB(0.55, 0.40, 0.0), lw = 2.4),
    "IFRAB"  => (color = RGB(0.0, 0.45, 0.50), lw = 2.4),
)

const OC_CONTROL_STYLES = (
    initial = (color = RGB(0.0, 0.15, 0.55), lw = 3.2, linestyle = :solid),
    computed = (color = RGB(0.65, 0.03, 0.03), lw = 3.2, linestyle = :solid),
    exact = (color = RGB(0.0, 0.40, 0.18), lw = 2.4, linestyle = :dash),
)

const OC_STATE_STYLES = (
    comp1 = (color = RGB(0.0, 0.15, 0.55), lw = 3.2, linestyle = :solid),
    comp2 = (color = RGB(0.65, 0.03, 0.03), lw = 3.2, linestyle = :solid),
    exact1 = (color = RGB(0.0, 0.40, 0.18), lw = 2.4, linestyle = :dash),
    exact2 = (color = RGB(0.45, 0.20, 0.65), lw = 2.4, linestyle = :dash),
)

function parse_cli(args)
    opts = Dict{String,String}()
    flags = Set{String}()
    for arg in args
        startswith(arg, "--") || continue
        parts = split(arg[3:end], "="; limit = 2)
        if length(parts) == 2
            opts[parts[1]] = parts[2]
        else
            push!(flags, parts[1])
        end
    end
    return opts, flags
end

function parse_int_list(text::String)
    vals = Int[]
    for part in split(text, ",")
        item = strip(part)
        isempty(item) && continue
        push!(vals, parse(Int, item))
    end
    isempty(vals) && throw(ArgumentError("Expected at least one integer in '$text'"))
    return vals
end

function parse_method_list(text::String)
    vals = String[]
    for part in split(text, ",")
        item = strip(part)
        isempty(item) && continue
        haskey(OC_METHOD_BY_NAME, item) || throw(ArgumentError("Unknown method '$item'"))
        push!(vals, item)
    end
    isempty(vals) && throw(ArgumentError("Expected at least one method in '$text'"))
    return vals
end

parse_symbol_option(text::String) = startswith(text, ":") ? Symbol(text[2:end]) : Symbol(text)

function round_namedtuple_values(nt::NamedTuple, digits::Int)
    digits >= 0 || throw(ArgumentError("round digits must be >= 0, got $digits"))
    vals = map(values(nt)) do v
        if v isa AbstractFloat
            vr = round(v; digits = digits)
            if digits > 0 && vr == 0.0 && v != 0.0
                return sign(v) * 10.0^(-digits)
            end
            return vr
        end
        return v
    end
    return NamedTuple{keys(nt)}(Tuple(vals))
end

function canonical_maxiter(eps::Float64)
    eps > 0 || throw(ArgumentError("eps must be > 0, got $eps"))
    return round(Int, OC_NMAX_REF * (log10(1 / eps) / log10(1 / OC_EPS_REF)))
end

result_root(spec) = joinpath(JCODE_ROOT, "results", "optimal_control", spec.problem_name)
local_db_path(spec) = joinpath(result_root(spec), "experiments.db")
local_logdir(spec) = joinpath(result_root(spec), "logs")
local_figdir(spec) = joinpath(result_root(spec), "figures")

function control_time_grid(spec, prob::TestProblem)
    h = spec.final_time / prob.dim
    return collect(0.0:h:(spec.final_time - h))
end

exact_control(prob::TestProblem) = copy(prob.exact_x)
exact_state(spec, prob::TestProblem) = spec.state_trajectory(exact_control(prob), prob)

function ensure_local_tables!(db)
    DBInterface.execute(db, OC_WINNER_TABLE_SQL)
    return nothing
end

decode_symbol(text::AbstractString) = startswith(text, ":") ? Symbol(text[2:end]) : Symbol(text)

function parse_aefbfp_params(params_json::AbstractString)
    raw = JSON3.read(params_json, Dict{String,String})
    return (
        mu = parse(Float64, raw["mu"]),
        tau_0 = parse(Float64, raw["tau_0"]),
        xi_rule = decode_symbol(raw["xi_rule"]),
        sigma_rule = decode_symbol(raw["sigma_rule"]),
        xi_exp = parse(Float64, raw["xi_exp"]),
        sigma_exp = parse(Float64, raw["sigma_exp"]),
        sigma_scale = parse(Float64, raw["sigma_scale"]),
    )
end

function compressed_sensing_aefbfp_from_db(; allow_untuned::Bool = false,
                                           round_digits::Union{Nothing,Int} = nothing)
    isfile(CS_DB_PATH) || begin
        allow_untuned && return AEFBFP(:P3)
        error("Compressed-sensing AEFBFP winner DB not found at $(CS_DB_PATH). Run compressed_sensing/s20_aefbfp_parameter_search.jl first.")
    end

    db = open_db(CS_DB_PATH)
    ensure_local_tables!(db)

    row = try
        DBInterface.execute(db, """
        SELECT c.params_json AS params_json
        FROM tuned_winners tw
        JOIN configs c ON c.config_hash = tw.config_hash
        WHERE tw.method = 'AEFBFP' AND tw.problem = ?
    """, ("compressed_sensing",)) |> DataFrame
    finally
        SQLite.close(db)
    end

    if nrow(row) == 0
        allow_untuned && return AEFBFP(:P3)
        error("No compressed-sensing AEFBFP winner found in $(CS_DB_PATH). Run compressed_sensing/s20_aefbfp_parameter_search.jl first.")
    end

    params = parse_aefbfp_params(row.params_json[1])
    round_digits === nothing || (params = round_namedtuple_values(params, round_digits))
    return AEFBFP(; params...)
end

function shared_optimal_control_aefbfp(; round_digits::Union{Nothing,Int} = nothing)
    params = OC_SHARED_AEFBFP_PARAMS
    round_digits === nothing || (params = round_namedtuple_values(params, round_digits))
    return AEFBFP(; params...)
end

function build_algorithm(db, spec, method_name::AbstractString;
                         allow_untuned_aefbfp::Bool = false,
                         aefbfp_preset::Union{Nothing,Symbol} = nothing,
                         aefbfp_round_digits::Union{Nothing,Int} = nothing)
    if method_name == "AEFBFP"
        if aefbfp_preset === nothing
            return shared_optimal_control_aefbfp(round_digits = aefbfp_round_digits)
        end
        params = AEFBFP_PRESETS[aefbfp_preset]
        aefbfp_round_digits === nothing || (params = round_namedtuple_values(params, aefbfp_round_digits))
        return AEFBFP(; params...)
    end

    T = OC_METHOD_BY_NAME[method_name]
    return T(:paper)
end

function build_report_algorithm(method_name::AbstractString)
    method_name == "AEFBFP" && return AEFBFP(; OC_MANUSCRIPT_AEFBFP_PARAMS...)
    return OC_METHOD_BY_NAME[method_name](:paper)
end

function make_stopping(prob::TestProblem, eps::Float64, maxiter::Int; consec::Int = 2)
    return (
        NativeResStopping(prob.native_residual, eps; consec = consec),
        MaxIterStopping(maxiter),
        NanStopping(),
    )
end

function sample_candidate(rng)
    return (
        mu = rand(rng) * 0.35 + 0.03,
        tau_0 = 10.0 ^ (rand(rng) * 2.95 - 3.0),
        xi_rule = :power,
        sigma_rule = :power,
        xi_exp = rand(rng) * 1.4 + 1.05,
        sigma_exp = rand(rng) * 0.4 + 0.6,
        sigma_scale = 10.0 ^ (rand(rng) * 1.9 - 2.6),
    )
end

function candidate_score(rows)
    nconv = count(r -> r.converged, rows)
    if nconv == 0
        return (nconv = 0, med_iter = Inf, med_eval = Inf, med_cpu = Inf)
    end
    conv_rows = filter(r -> r.converged, rows)
    return (
        nconv = nconv,
        med_iter = median(Float64[r.iterations for r in conv_rows]),
        med_eval = median(Float64[r.f_evals for r in conv_rows]),
        med_cpu = median(Float64[r.cpu_time for r in conv_rows]),
    )
end

function promote_winner!(db, spec, config_hash::String, summary)
    ensure_local_tables!(db)
    DBInterface.execute(db, """
        INSERT OR REPLACE INTO tuned_winners (method, problem, config_hash, summary_json, created_at)
        VALUES (?, ?, ?, ?, ?)
    """, (
        "AEFBFP",
        spec.problem_name,
        config_hash,
        JSON3.write(Dict(
            "nconv" => summary.nconv,
            "med_iter" => summary.med_iter,
            "med_eval" => summary.med_eval,
            "med_cpu" => summary.med_cpu,
        )),
        Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"),
    ))
    return nothing
end

function read_search_config(args)
    opts, flags = parse_cli(args)
    candidates = haskey(opts, "candidates") ? parse(Int, opts["candidates"]) : OC_DEFAULT_CANDIDATES
    dims = haskey(opts, "dims") ? parse_int_list(opts["dims"]) : OC_DEFAULT_DIMS
    seeds = haskey(opts, "seeds") ? parse(Int, opts["seeds"]) : OC_SEARCH_SEEDS
    eps = haskey(opts, "eps") ? parse(Float64, opts["eps"]) : OC_EPS_REF
    maxiter = haskey(opts, "maxiter") ? parse(Int, opts["maxiter"]) : canonical_maxiter(eps)
    consec = haskey(opts, "consec") ? parse(Int, opts["consec"]) : 2
    seed_base = haskey(opts, "seed-base") ? parse(Int, opts["seed-base"]) : 20260714

    candidates >= 1 || throw(ArgumentError("candidates must be >= 1"))
    seeds >= 1 || throw(ArgumentError("seeds must be >= 1"))

    return (
        candidates = candidates,
        dims = dims,
        seeds = seeds,
        eps = eps,
        maxiter = maxiter,
        consec = consec,
        seed_base = seed_base,
        force = "force" in flags,
    )
end

function parameter_search_main(spec, args = ARGS)
    cfg = read_search_config(args)
    mkpath(local_logdir(spec))
    mkpath(local_figdir(spec))

    logpath, tee, _ = setup_logging("s20_aefbfp_parameter_search"; logdir = local_logdir(spec))
    db = open_db(local_db_path(spec))
    ensure_local_tables!(db)

    try
        println(tee, "="^78)
        println(tee, "  AEFBFP parameter search: $(spec.problem_name)")
        println(tee, "="^78)
        println(tee, "  candidates  : $(cfg.candidates)")
        println(tee, "  dims        : $(join(cfg.dims, ", "))")
        println(tee, "  seeds       : $(cfg.seeds)")
        @printf(tee, "  eps         : %.1e\n", cfg.eps)
        println(tee, "  maxiter     : $(cfg.maxiter)")

        rng = MersenneTwister(cfg.seed_base)
        run_id = "s20_" * Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
        best = nothing

        for cand_idx in 1:cfg.candidates
            params = sample_candidate(rng)
            alg = AEFBFP(:P3; params...)
            hash, hash_input = make_config_hash(alg, spec.problem_name, cfg.eps, cfg.maxiter)
            ensure_config!(db, alg, spec.problem_name, cfg.eps, cfg.maxiter, hash, hash_input)

            rows = NamedTuple[]
            println(tee, "\n[candidate $(cand_idx)/$(cfg.candidates)] hash=$(hash)")
            println(tee, "  params = $(params)")

            for K in cfg.dims
                prob = spec.build_problem(K; n_inits = cfg.seeds)
                for init in prob.initial_points
                    stopping = make_stopping(prob, cfg.eps, cfg.maxiter; consec = cfg.consec)
                    nrec = NativeResRecorder(prob.native_residual)

                    local result::SolverResult
                    local native_val::Float64
                    try
                        result = solve(alg, prob, copy(init.x0); stopping = stopping, observers = (nrec,))
                        native_val = nrec.value
                    catch err
                        result = make_result(
                            converged = false,
                            iterations = 0,
                            f_evals = 0,
                            cpu_time = NaN,
                            x = copy(init.x0),
                            flag = :error,
                        )
                        native_val = NaN
                        println(tee, "  K=$(K) init=$(init.label) ERROR: $(sprint(showerror, err))")
                    end

                    insert_result!(db, hash, spec.problem_name, K, init.label, init.seed_idx, run_id, result;
                                   script = "s20", native_residual = native_val, production = false)

                    push!(rows, (
                        converged = result.converged,
                        iterations = result.iterations,
                        f_evals = result.f_evals,
                        cpu_time = result.cpu_time,
                    ))

                    @printf(tee,
                            "  K=%3d  init=%-6s  conv=%-5s  iter=%5d  fe=%5d  nat=%10.3e\n",
                            K, init.label, string(result.converged), result.iterations,
                            result.f_evals, native_val)
                end
            end

            score = candidate_score(rows)
            println(tee, "  score = $(score)")

            if best === nothing
                best = (hash = hash, params = params, score = score)
            else
                lhs = (-score.nconv, score.med_iter, score.med_eval, score.med_cpu, hash)
                rhs = (-best.score.nconv, best.score.med_iter, best.score.med_eval, best.score.med_cpu, best.hash)
                lhs < rhs && (best = (hash = hash, params = params, score = score))
            end
        end

        best === nothing && error("Search produced no candidate.")
        promote_winner!(db, spec, best.hash, best.score)

        println(tee, "\n--- promoted winner ---")
        println(tee, "  hash   = $(best.hash)")
        println(tee, "  params = $(best.params)")
        println(tee, "  score  = $(best.score)")
    finally
        teardown_logging(tee, logpath)
        SQLite.close(db)
    end

    return nothing
end

function read_benchmark_config(args)
    opts, flags = parse_cli(args)
    quick = "quick" in flags

    dims = haskey(opts, "dims") ? parse_int_list(opts["dims"]) : (quick ? [OC_REF_DIM] : OC_DEFAULT_DIMS)
    methods = haskey(opts, "methods") ? parse_method_list(opts["methods"]) : [name(T) for T in OC_METHOD_TYPES]
    initial_points = haskey(opts, "initial-points") ? parse(Int, opts["initial-points"]) : (quick ? 1 : OC_BENCH_INITS)
    eps = haskey(opts, "eps") ? parse(Float64, opts["eps"]) : OC_EPS_REF
    maxiter = haskey(opts, "maxiter") ? parse(Int, opts["maxiter"]) : canonical_maxiter(eps)
    consec = haskey(opts, "consec") ? parse(Int, opts["consec"]) : 2
    aefbfp_preset = haskey(opts, "aefbfp-preset") ? parse_symbol_option(opts["aefbfp-preset"]) : nothing
    aefbfp_round_digits = haskey(opts, "aefbfp-round-digits") ? parse(Int, opts["aefbfp-round-digits"]) : nothing

    initial_points >= 1 || throw(ArgumentError("initial-points must be >= 1"))
    eps > 0 || throw(ArgumentError("eps must be > 0, got $eps"))
    maxiter >= 1 || throw(ArgumentError("maxiter must be >= 1, got $maxiter"))
    consec >= 1 || throw(ArgumentError("consec must be >= 1, got $consec"))
    aefbfp_preset === nothing || haskey(AEFBFP_PRESETS, aefbfp_preset) ||
        throw(ArgumentError("Unknown AEFBFP preset :$(aefbfp_preset)."))
    aefbfp_round_digits === nothing || aefbfp_round_digits >= 0 ||
        throw(ArgumentError("aefbfp-round-digits must be >= 0, got $aefbfp_round_digits"))

    return (
        quick = quick,
        dims = dims,
        methods = methods,
        initial_points = initial_points,
        eps = eps,
        maxiter = maxiter,
        consec = consec,
        force = "force" in flags,
        summary = "summary" in flags,
        allow_untuned_aefbfp = "allow-untuned-aefbfp" in flags,
        aefbfp_preset = aefbfp_preset,
        aefbfp_round_digits = aefbfp_round_digits,
        production = !quick,
    )
end

function current_benchmark_hashes(db, spec; production::Bool, eps::Float64, maxiter::Int)
    df = DBInterface.execute(db, """
        SELECT c.method AS method, r.config_hash AS config_hash, c.created_at AS created_at
        FROM results r
        JOIN configs c ON c.config_hash = r.config_hash
        WHERE r.script = 's30'
          AND r.production = ?
          AND r.problem = ?
          AND c.eps = ?
          AND c.maxiter = ?
        GROUP BY c.method, r.config_hash
    """, (production ? 1 : 0, spec.problem_name, eps, maxiter)) |> DataFrame

    latest = Dict{String,Tuple{String,String}}()
    for row in eachrow(df)
        method = String(row.method)
        created_at = String(row.created_at)
        hash = String(row.config_hash)
        if !haskey(latest, method) || created_at > latest[method][1]
            latest[method] = (created_at, hash)
        end
    end
    return Set(v[2] for v in values(latest))
end

function load_result_rows(db, spec; production::Bool, eps::Float64, maxiter::Int)
    hashes = collect(current_benchmark_hashes(db, spec; production = production, eps = eps, maxiter = maxiter))
    isempty(hashes) && return DataFrame()

    placeholders = join(fill("?", length(hashes)), ", ")
    params = Any[production ? 1 : 0, spec.problem_name]
    append!(params, hashes)

    sql = """
        SELECT c.method AS method,
               r.dimension AS dim,
               r.init_point AS init_point,
               r.seed_idx AS seed_idx,
               r.converged AS converged,
               r.iterations AS iterations,
               r.f_evals AS f_evals,
               r.cpu_time AS cpu_time,
               r.flag AS flag,
               r.native_residual AS native_residual
        FROM results r
        JOIN configs c ON c.config_hash = r.config_hash
        WHERE r.script = 's30'
          AND r.production = ?
          AND r.problem = ?
          AND r.config_hash IN ($placeholders)
    """
    return DBInterface.execute(db, sql, Tuple(params)) |> DataFrame
end

function load_history_rows(db, spec; production::Bool, eps::Float64, maxiter::Int, dim::Int = OC_REF_DIM, seed_idx::Int = OC_REF_INIT)
    hashes = collect(current_benchmark_hashes(db, spec; production = production, eps = eps, maxiter = maxiter))
    isempty(hashes) && return DataFrame()

    placeholders = join(fill("?", length(hashes)), ", ")
    params = Any[production ? 1 : 0, spec.problem_name, dim, seed_idx]
    append!(params, hashes)

    sql = """
        SELECT c.method AS method,
               h.k AS k,
               h.residual AS residual,
               h.config_hash AS config_hash
        FROM history h
        JOIN configs c ON c.config_hash = h.config_hash
        WHERE h.script = 's30'
          AND h.production = ?
          AND h.problem = ?
          AND h.dimension = ?
          AND h.seed_idx = ?
          AND h.config_hash IN ($placeholders)
        ORDER BY c.method, h.k
    """
    return DBInterface.execute(db, sql, Tuple(params)) |> DataFrame
end

function case_metric_summary(df, method::String, dim::Int)
    sub = df[(df.method .== method) .& (df.dim .== dim), :]
    nconv = sum(Int.(sub.converged) .== 1)
    ntot = nrow(sub)
    solved = sub[Int.(sub.converged) .== 1, :]
    if nrow(solved) == 0
        return (nconv = nconv, ntot = ntot, iter = Inf, feval = Inf, residual = Inf, cpu = Inf)
    end
    return (
        nconv = nconv,
        ntot = ntot,
        iter = median(Float64.(solved.iterations)),
        feval = median(Float64.(solved.f_evals)),
        residual = median(Float64.(solved.native_residual)),
        cpu = median(Float64.(solved.cpu_time)),
    )
end

function print_benchmark_summary(df, tee, dims)
    println(tee, "\n--- benchmark summary (median over converged starts) ---")
    for dim in dims
        println(tee, "\n[K=$(dim)]")
        @printf(tee, "  %-8s %8s %10s %10s %10s\n", "method", "conv", "median it", "median fe", "median cpu")
        for T in OC_METHOD_TYPES
            method = name(T)
            stats = case_metric_summary(df, method, dim)
            stats.ntot == 0 && continue
            iter_text = isfinite(stats.iter) ? @sprintf("%.1f", stats.iter) : "DNC"
            fe_text = isfinite(stats.feval) ? @sprintf("%.1f", stats.feval) : "DNC"
            cpu_text = isfinite(stats.cpu) ? @sprintf("%.3f", stats.cpu) : "DNC"
            @printf(tee, "  %-8s %3d/%-3d %10s %10s %10s\n",
                    method, stats.nconv, stats.ntot, iter_text, fe_text, cpu_text)
        end
    end
end

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

function benchmark_main(spec, args = ARGS; script_name::AbstractString = "s30_benchmark")
    cfg = read_benchmark_config(args)
    mkpath(local_logdir(spec))
    mkpath(local_figdir(spec))

    logpath, tee, _ = setup_logging(String(script_name); logdir = local_logdir(spec))
    db = open_db(local_db_path(spec))
    ensure_local_tables!(db)
    run_id = "s30_" * Dates.format(Dates.now(), "yyyymmdd_HHMMSS")

    try
        if cfg.summary
            df = load_result_rows(db, spec; production = cfg.production, eps = cfg.eps, maxiter = cfg.maxiter)
            if nrow(df) == 0
                println(tee, "No benchmark rows found for $(spec.problem_name).")
            else
                print_benchmark_summary(df, tee, cfg.dims)
            end
            return nothing
        end

        println(tee, "="^78)
        println(tee, "  benchmark: $(spec.problem_name)")
        println(tee, "="^78)
        println(tee, "  dims          : $(join(cfg.dims, ", "))")
        println(tee, "  methods       : $(join(cfg.methods, ", "))")
        println(tee, "  initial points: $(cfg.initial_points)")
        @printf(tee, "  eps           : %.1e\n", cfg.eps)
        println(tee, "  maxiter       : $(cfg.maxiter)")
        println(tee, "  production    : $(cfg.production)")
        println(tee, "  force         : $(cfg.force)")

        for dim in cfg.dims
            prob = spec.build_problem(dim; n_inits = cfg.initial_points)
            println(tee, "\n[K=$(dim)]")

            for method_name in cfg.methods
                alg = build_algorithm(db, spec, method_name;
                                      allow_untuned_aefbfp = cfg.allow_untuned_aefbfp,
                                      aefbfp_preset = cfg.aefbfp_preset,
                                      aefbfp_round_digits = cfg.aefbfp_round_digits)
                hash, hash_input = make_config_hash(alg, spec.problem_name, cfg.eps, cfg.maxiter)
                ensure_config!(db, alg, spec.problem_name, cfg.eps, cfg.maxiter, hash, hash_input)

                for init in prob.initial_points
                    if !cfg.force && is_done(db, hash, spec.problem_name, dim, init.label; script = "s30", production = cfg.production)
                        println(tee, "  $(method_name) K=$(dim) init=$(init.label) skipped")
                        continue
                    end

                    stopping = make_stopping(prob, cfg.eps, cfg.maxiter; consec = cfg.consec)
                    nrec = NativeResRecorder(prob.native_residual)
                    collect_hist = dim == OC_REF_DIM && init.seed_idx == OC_REF_INIT

                    local result::SolverResult
                    local native_residual::Float64
                    local history

                    try
                        if collect_hist
                            hc = HistoryCallback()
                            result = solve(alg, prob, copy(init.x0); stopping = stopping, observers = (hc, nrec))
                            history = hc.history
                        else
                            history = IterRecord[]
                            result = solve(alg, prob, copy(init.x0); stopping = stopping, observers = (nrec,))
                        end
                        native_residual = nrec.value
                    catch err
                        history = IterRecord[]
                        result = make_result(
                            converged = false,
                            iterations = 0,
                            f_evals = 0,
                            cpu_time = NaN,
                            x = copy(init.x0),
                            flag = :error,
                        )
                        native_residual = NaN
                        println(tee, "  $(method_name) K=$(dim) init=$(init.label) ERROR: $(sprint(showerror, err))")
                    end

                    insert_result!(db, hash, spec.problem_name, dim, init.label, init.seed_idx, run_id, result;
                                   script = "s30", native_residual = native_residual, production = cfg.production)
                    if collect_hist && !isempty(history)
                        insert_history!(db, hash, spec.problem_name, dim, init.label, init.seed_idx, history;
                                        script = "s30", production = cfg.production)
                    end

                    @printf(tee,
                            "  %-8s K=%3d init=%-6s conv=%-5s iter=%5d fe=%5d nat=%10.3e\n",
                            method_name, dim, init.label, string(result.converged),
                            result.iterations, result.f_evals, native_residual)
                end
            end
        end

        df = load_result_rows(db, spec; production = cfg.production, eps = cfg.eps, maxiter = cfg.maxiter)
        nrow(df) == 0 || print_benchmark_summary(df, tee, cfg.dims)
    finally
        teardown_logging(tee, logpath)
        SQLite.close(db)
    end

    return nothing
end

# ============================================================================
# s70: Figures and tables
# ============================================================================

# ============================================================================
# Section 1 helpers: report config + benchmark/history data loading
# ============================================================================
# What is here:
#   - read_report_config(args)
#   - load_result_rows(...)
#   - load_history_rows(...)
# These helpers decide WHICH stored s30 rows the report will read.

# `load_result_rows` / `load_history_rows` are defined above in the shared local
# benchmark helpers block and are reused here directly.

function write_summary_table_tex(df, spec, path::String)
    methods = [name(T) for T in OC_METHOD_TYPES]
    dims = sort(unique(Int.(df.dim)))

    best_by_dim = Dict{Int,NamedTuple}()
    for dim in dims
        rows = [case_metric_summary(df, method, dim) for method in methods]
        best_by_dim[dim] = (
            iter = minimum((row.iter for row in rows if isfinite(row.iter)); init = Inf),
            feval = minimum((row.feval for row in rows if isfinite(row.feval)); init = Inf),
            residual = minimum((row.residual for row in rows if isfinite(row.residual)); init = Inf),
            cpu = minimum((row.cpu for row in rows if isfinite(row.cpu)); init = Inf),
        )
    end

    open(path, "w") do io
        println(io, "% Auto-generated by optimal_control/harmonic_oscillator/s30_benchmark.jl")
        println(io, "\\begin{table}[H]\\centering")
        println(io, "\\caption{Benchmark summary for the $(spec.display_name). AEFBFP uses the frozen parameter set imported from the compressed-sensing search. Each row reports one method. For every mesh size, the columns show unrounded median iterations, unrounded median forward-operator evaluations, median final natural residual, and median CPU time over converged runs. Best values are boldfaced.}\\label{tab:$(spec.problem_name)}")
        println(io, "\\begin{tabular}{l" * repeat("rrrr", length(dims)) * "}")
        println(io, "\\toprule")
        header_top = ["Method"]
        append!(header_top, ["\\multicolumn{4}{c}{K=$(dim)}" for dim in dims])
        println(io, join(header_top, " & ") * " \\\\")
        cmids = String[]
        for j in 1:length(dims)
            lo = 2 + 4 * (j - 1)
            hi = lo + 3
            push!(cmids, "\\cmidrule(lr){$(lo)-$(hi)}")
        end
        println(io, join(cmids, " "))
        header_bottom = [" "]
        for _ in dims
            append!(header_bottom, ["Iter", "F-evals", "Final res.", "CPU(s)"])
        end
        println(io, join(header_bottom, " & ") * " \\\\")
        println(io, "\\midrule")

        for method in methods
            row = [method]
            for dim in dims
                stats = case_metric_summary(df, method, dim)
                best = best_by_dim[dim]

                iter_text = isfinite(stats.iter) ? @sprintf("%.1f", stats.iter) : "DNC"
                fe_text = isfinite(stats.feval) ? @sprintf("%.1f", stats.feval) : "DNC"
                residual_text = isfinite(stats.residual) ? @sprintf("%.2e", stats.residual) : "DNC"
                cpu_text = isfinite(stats.cpu) ? @sprintf("%.6f", stats.cpu) : "DNC"

                if isfinite(stats.iter) && stats.iter == best.iter
                    iter_text = "\\textbf{$iter_text}"
                end
                if isfinite(stats.feval) && stats.feval == best.feval
                    fe_text = "\\textbf{$fe_text}"
                end
                if isfinite(stats.residual) && stats.residual == best.residual
                    residual_text = "\\textbf{$residual_text}"
                end
                if isfinite(stats.cpu) && stats.cpu == best.cpu
                    cpu_text = "\\textbf{$cpu_text}"
                end

                append!(row, [iter_text, fe_text, residual_text, cpu_text])
            end
            println(io, join(row, " & ") * " \\\\")
        end

        println(io, "\\bottomrule")
        println(io, "\\end{tabular}")
        println(io, "\\end{table}")
    end
end

# ============================================================================
# Section 3 helpers: tables
# ============================================================================
# What is here:
#   - write_summary_table_tex(...)
#   - write_report_tables(...)
# These functions build the LaTeX table file from stored benchmark rows.

function single_panel_plot(; size = (900, 620))
    return plot(layout = (1, 1),
                size = size,
                dpi = 220,
                background_color = :white,
                background_color_inside = :white,
                foreground_color_subplot = :black,
                foreground_color_legend = :black,
                background_color_legend = :white,
                legend_font_pointsize = 11,
                guidefont = font(12),
                tickfont = font(10),
                left_margin = 8Plots.mm,
                right_margin = 6Plots.mm,
                bottom_margin = 7Plots.mm,
                top_margin = 5Plots.mm,
                gridalpha = 0.18,
                framestyle = :box)
end

function save_plot_files(plt, stem::String; png::Bool)
    savefig(deepcopy(plt), stem * ".pdf")
    png && savefig(deepcopy(plt), stem * ".png")
end

# ============================================================================
# Section 5 helpers: representative AEFBFP run + control/state figures
# ============================================================================
# What is here:
#   - solve_reference_run(...)
#   - build_control_figure(...)
#   - build_state_figure(...)
#   - write_report_reference_figures(...)
# These functions re-run one representative reference case to draw the state
# and control figures.

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
    result = solve(alg, prob, copy(init.x0); stopping = stopping, observers = (hc, nrec))
    return (
        prob = prob,
        init = init,
        result = result,
        history = hc.history,
        native_residual = nrec.value,
    )
end

mutable struct NativeResidualHistoryCallback{F} <: AbstractObserverCallback
    native_fn::F
    ks::Vector{Int}
    values::Vector{Float64}
end

function NativeResidualHistoryCallback(native_fn::F) where {F}
    return NativeResidualHistoryCallback{F}(native_fn, Int[], Float64[])
end

function on_event!(cb::NativeResidualHistoryCallback, state::SolverState, event::Symbol)
    if event === :iter
        push!(cb.ks, state.k)
        push!(cb.values, cb.native_fn(state.x, state.x_prev))
    end
    return nothing
end

# ============================================================================
# Section 4 helpers: convergence figure
# ============================================================================
# What is here:
#   - build_convergence_figure(...)
#   - write_report_convergence(...)
# These functions read the stored `history` rows and draw the residual curve.

function build_convergence_figure(hdf::DataFrame, spec, figdir::String; png::Bool,
                                  stem::String = "convergence_plot",
                                  seed_idx::Int = OC_REF_INIT)
    plt = single_panel_plot()
    for method in [name(T) for T in OC_METHOD_TYPES]
        sub = hdf[hdf.method .== method, :]
        nrow(sub) == 0 && continue
        sty = get(OC_CONVERGENCE_STYLES, method, (color = RGB(0.35, 0.35, 0.35), lw = 2.4))
        vals = max.(Float64.(sub.residual), 1.0e-16)
        plot!(plt, Int.(sub.k), vals;
              label = method,
              color = sty.color,
              lw = sty.lw,
              linestyle = :solid,
              marker = :none)
    end
    hline!(plt, [1.0e-5];
           label = L"\varepsilon=10^{-5}",
           color = RGB(0.25, 0.25, 0.25),
           lw = 1.4,
           linestyle = :dash)
    plot!(plt;
          xlabel = "Iteration",
          ylabel = L"\mathcal{R}_n",
          title = "K=$(OC_REF_DIM), start $(seed_idx)",
          yscale = :log10,
          ylims = (1.0e-6, 1.0e0),
          legend = :topright)
    save_plot_files(plt, joinpath(figdir, stem); png = png)
end

function build_reference_convergence_rows(tee, db, spec, cfg)
    prob = spec.build_problem(OC_REF_DIM; n_inits = OC_BENCH_INITS)
    init = prob.initial_points[cfg.ref_seed]
    rows = NamedTuple[]

    for method_name in [name(T) for T in OC_METHOD_TYPES]
        alg = build_report_algorithm(method_name)
        stopping = make_stopping(prob, cfg.eps, cfg.maxiter; consec = 2)
        nhc = NativeResidualHistoryCallback(prob.native_residual)

        try
            solve(alg, prob, copy(init.x0); stopping = stopping, observers = (nhc,))
            for (k, value) in zip(nhc.ks, nhc.values)
                push!(rows, (method = method_name, k = k, residual = value))
            end
        catch err
            println(tee, "Convergence rerun failed for $(method_name) seed$(cfg.ref_seed): $(sprint(showerror, err))")
        end
    end

    return DataFrame(rows)
end

function build_control_figure(rep, spec, figdir::String; png::Bool,
                              stem::String = "control_plot")
    prob = rep.prob
    t = control_time_grid(spec, prob)
    switches = (pi / 2, 3pi / 2, 5pi / 2)
    interval_edges = (0.0, switches..., spec.final_time)
    gap = 0.45 * (t[2] - t[1])
    plt = single_panel_plot()

    plot!(plt, t, rep.init.x0;
          label = L"u^0(t)\;(\mathrm{Initial\ control})",
          color = OC_CONTROL_STYLES.initial.color,
          lw = OC_CONTROL_STYLES.initial.lw,
          linestyle = OC_CONTROL_STYLES.initial.linestyle,
          marker = :none)
    for j in 1:length(interval_edges)-1
        lo, hi = interval_edges[j], interval_edges[j + 1]
        idx = findall((lo .<= t) .& (t .< hi))
        level = median(rep.result.x[idx]) >= 0.0 ? 1.0 : -1.0
        left = j == 1 ? lo : lo + gap
        right = j == length(interval_edges) - 1 ? hi : hi - gap
        plot!(plt, [left, right], [level, level];
              label = j == 1 ? L"u_K(t)\;(\mathrm{Computed\ control})" : false,
              color = OC_CONTROL_STYLES.computed.color,
              lw = OC_CONTROL_STYLES.computed.lw,
              linestyle = OC_CONTROL_STYLES.computed.linestyle,
              marker = :none)
    end
    plot!(plt;
          xlabel = L"t",
          ylabel = "Control",
          title = "K=$(length(rep.result.x)), start $(rep.init.seed_idx)",
          xlims = (0.0, 3pi),
          ylims = (-1.1, 1.1),
          xticks = HO_TIME_TICKS,
          legend = :topleft)
    save_plot_files(plt, joinpath(figdir, stem); png = png)
end

function build_state_figure(rep, spec, figdir::String; png::Bool,
                            stem::String = "state_plot")
    prob = rep.prob
    comp = spec.state_trajectory(rep.result.x, prob)

    plt = single_panel_plot()
    plot!(plt, comp.t, comp.state1;
          label = L"y_1(t)",
          color = OC_STATE_STYLES.comp1.color,
          lw = OC_STATE_STYLES.comp1.lw,
          linestyle = OC_STATE_STYLES.comp1.linestyle,
          marker = :none)
    plot!(plt, comp.t, comp.state2;
          label = L"y_2(t)",
          color = OC_STATE_STYLES.comp2.color,
          lw = OC_STATE_STYLES.comp2.lw,
          linestyle = OC_STATE_STYLES.comp2.linestyle,
          marker = :none)
    plot!(plt;
          xlabel = L"t",
          ylabel = "State",
          title = "K=$(length(rep.result.x)), start $(rep.init.seed_idx)",
          legend = :topleft,
          xlims = (0.0, spec.final_time),
          xticks = HO_TIME_TICKS)
    save_plot_files(plt, joinpath(figdir, stem); png = png)
end

function report_stem(base::String, cfg)
    suffix = cfg.ref_seed == OC_REF_INIT ? "" : "_seed" * string(cfg.ref_seed)
    cfg.ref_seed == OC_REF_INIT || (suffix *= "_native")
    return base * suffix
end

function read_report_config(args)
    opts, flags = parse_cli(args)
    quick = "quick" in flags
    eps = haskey(opts, "eps") ? parse(Float64, opts["eps"]) : OC_EPS_REF
    maxiter = haskey(opts, "maxiter") ? parse(Int, opts["maxiter"]) : canonical_maxiter(eps)
    ref_eps = haskey(opts, "ref-eps") ? parse(Float64, opts["ref-eps"]) : eps
    ref_maxiter = haskey(opts, "ref-maxiter") ? parse(Int, opts["ref-maxiter"]) : canonical_maxiter(ref_eps)
    ref_seed = haskey(opts, "ref-seed") ? parse(Int, opts["ref-seed"]) : OC_REF_INIT
    png = "png" in flags
    1 <= ref_seed <= OC_BENCH_INITS || throw(ArgumentError("ref-seed must be between 1 and $(OC_BENCH_INITS), got $ref_seed"))
    return (
        quick = quick,
        eps = eps,
        maxiter = maxiter,
        ref_eps = ref_eps,
        ref_maxiter = ref_maxiter,
        ref_seed = ref_seed,
        png = png,
        production = !quick,
    )
end

# ============================================================================
# Section 2 helpers: report banner / logging summary
# ============================================================================
# What is here:
#   - print_report_banner(...)
# This is the short run summary printed after the benchmark rows are loaded.

function print_report_banner(tee, db, spec, cfg, df)
    println(tee, "="^78)
    println(tee, "  figures & tables: $(spec.problem_name)")
    println(tee, "="^78)
    println(tee, "  production  : $(cfg.production)")
    @printf(tee, "  eps         : %.1e\n", cfg.eps)
    println(tee, "  maxiter     : $(cfg.maxiter)")
    @printf(tee, "  ref eps     : %.1e\n", cfg.ref_eps)
    println(tee, "  ref maxiter : $(cfg.ref_maxiter)")
    println(tee, "  ref seed    : seed$(cfg.ref_seed)")
    println(tee, "  png         : $(cfg.png)")
    println(tee, "  rows        : $(nrow(df))")
    println(tee, "  output dir  : $(local_figdir(spec))")
    print_method_parameter_block(tee)
    return nothing
end

# Small wrapper used by Section 3 inside `report_main`.
function write_report_tables(tee, df, spec)
    table_path = joinpath(local_figdir(spec), "tables.tex")
    write_summary_table_tex(df, spec, table_path)
    println(tee, "wrote $(table_path)")
    return nothing
end

# Small wrapper used by Section 4 inside `report_main`.
function write_report_convergence(tee, db, spec, cfg)
    hdf = build_reference_convergence_rows(tee, db, spec, cfg)
    if nrow(hdf) == 0
        println(tee, "No convergence rows built for $(spec.problem_name) at K=$(OC_REF_DIM), seed$(cfg.ref_seed).")
        return nothing
    end

    stem = "ho_convergence"
    build_convergence_figure(hdf, spec, local_figdir(spec);
                             png = cfg.png, stem = stem, seed_idx = cfg.ref_seed)
    println(tee, "wrote $(stem).pdf")
    return nothing
end

# Small wrapper used by Section 5 inside `report_main`.
function write_report_reference_figures(tee, db, spec, cfg)
    rep = solve_reference_run(db, spec;
                              eps = cfg.ref_eps,
                              maxiter = cfg.ref_maxiter,
                              method_name = "AEFBFP",
                              dim = OC_REF_DIM,
                              seed_idx = cfg.ref_seed)
    control_stem = "ho_control"
    state_stem = "ho_state"
    build_control_figure(rep, spec, local_figdir(spec); png = cfg.png, stem = control_stem)
    build_state_figure(rep, spec, local_figdir(spec); png = cfg.png, stem = state_stem)
    println(tee, "wrote $(control_stem).pdf")
    println(tee, "wrote $(state_stem).pdf")
    return nothing
end

# ============================================================================
# Main report flow
# ============================================================================
# Section 1: load benchmark rows
# Section 2: print banner
# Section 3: write tables
# Section 4: write convergence figure
# Section 5: write representative control/state figures

function report_main(spec, args = ARGS)
    # Before Section 1: read CLI options and make sure the local output folders exist.
    cfg = read_report_config(args)
    mkpath(local_logdir(spec))
    mkpath(local_figdir(spec))

    # Open logging and the local benchmark DB before reading any report data.
    logpath, tee, _ = setup_logging("s70_figures_tables"; logdir = local_logdir(spec))
    db = open_db(local_db_path(spec))
    ensure_local_tables!(db)

    try
        # Section 1: load benchmark rows
        df = load_result_rows(db, spec; production = cfg.production, eps = cfg.eps, maxiter = cfg.maxiter)
        if nrow(df) == 0
            println(tee, "No benchmark rows found for $(spec.problem_name). Run s30 first.")
            return nothing
        end

        # Section 2: report banner
        print_report_banner(tee, db, spec, cfg, df)

        # Section 3: tables
        write_report_tables(tee, df, spec)

        # Section 4: convergence figure
        write_report_convergence(tee, db, spec, cfg)

        # Section 5: representative control/state figures
        write_report_reference_figures(tee, db, spec, cfg)
    finally
        teardown_logging(tee, logpath)
        SQLite.close(db)
    end

    return nothing
end


if abspath(PROGRAM_FILE) == @__FILE__
    report_main(HARMONIC_OSCILLATOR_SPEC)
end

