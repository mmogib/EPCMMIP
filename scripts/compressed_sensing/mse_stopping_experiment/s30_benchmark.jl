# ============================================================================
# MSE-stopping compressed-sensing benchmark (independent experiment)
# ============================================================================
#
# This experiment uses the original compressed-sensing data construction:
# a row-orthonormal sensing matrix and a K-sparse planted vector whose nonzero
# entries are -1 or +1.  Unlike the original benchmark, stopping is based on
# recovery MSE(x, x_star), not on successive-iterate displacement.
#
# Usage:
#   julia --project=. scripts/compressed_sensing/mse_stopping_experiment/s30_benchmark.jl
#   julia --project=. scripts/compressed_sensing/mse_stopping_experiment/s30_benchmark.jl --quick
#   julia --project=. scripts/compressed_sensing/mse_stopping_experiment/s30_benchmark.jl --cases=M256_N512_K30
#   julia --project=. scripts/compressed_sensing/mse_stopping_experiment/s30_benchmark.jl --methods=AEFBFP,IFRAB
#   julia --project=. scripts/compressed_sensing/mse_stopping_experiment/s30_benchmark.jl --summary
# ============================================================================

include(joinpath(@__DIR__, "..", "..", "..", "src", "includes.jl"))

const EXPERIMENT_NAME = "cs_mse_stopping"
const RESULT_ROOT = joinpath(@__DIR__, "results")
const DB_PATH = joinpath(RESULT_ROOT, "experiments.db")
const LOGDIR = joinpath(RESULT_ROOT, "logs")
const FIGDIR = joinpath(RESULT_ROOT, "figures")
const RANKING_CSV = joinpath(RESULT_ROOT, "mse_ranking_summary.csv")
const SCRIPT_ID = "mse_s30"

const METHOD_TYPES = (AEFBFP, VAFBS, MDITSM, RFBSM, IRFBSM, IFRAB)
const METHOD_BY_NAME = Dict(name(T) => T for T in METHOD_TYPES)
const DEFAULT_CASES = [
    (problem = "M256_N512_K30", M = 256, N = 512, K = 30),
    (problem = "M256_N512_K50", M = 256, N = 512, K = 50),
    (problem = "M512_N1024_K50", M = 512, N = 1024, K = 50),
    (problem = "M512_N1024_K80", M = 512, N = 1024, K = 80),
]
const CASE_BY_NAME = Dict(case.problem => case for case in DEFAULT_CASES)
const DEFAULT_DATASETS = 1
const DEFAULT_INITIAL_POINTS = 10
const DEFAULT_REPS = 3
const REF_DATASET = 1
const REF_INIT = 1
const MSE_TOL_REF = 1.0e-5
const NMAX_REF = 5000
const GAMMA_REF = 1.0e-3
const NOISE_VAR_REF = 1.0e-4
const DATA_SEED_BASE = 20260720

mkpath(LOGDIR)
mkpath(FIGDIR)

"Stop once MSE is strictly below tolerance for `consec` successive iterations."
mutable struct MSEStopping <: AbstractStoppingCallback
    x_star::Vector{Float64}
    tol::Float64
    consec::Int
    below_count::Int
end

function MSEStopping(x_star::Vector{Float64}, tol::Real; consec::Int = 2)
    tol > 0 || throw(ArgumentError("MSE tolerance must be positive"))
    consec >= 1 || throw(ArgumentError("consec must be at least 1"))
    return MSEStopping(copy(x_star), Float64(tol), consec, 0)
end

mse(x::AbstractVector, x_star::AbstractVector) = sum(abs2, x .- x_star) / length(x_star)

function check_stop(cb::MSEStopping, state::SolverState)
    value = mse(state.x, cb.x_star)
    if isfinite(value) && value < cb.tol
        cb.below_count += 1
        cb.below_count >= cb.consec && return (true, :converged)
    else
        cb.below_count = 0
    end
    return (false, :none)
end

"Records the recovery MSE at every iteration for the representative run."
mutable struct MSEHistoryCallback <: AbstractObserverCallback
    x_star::Vector{Float64}
    ks::Vector{Int}
    f_evals::Vector{Int}
    elapsed::Vector{Float64}
    values::Vector{Float64}
end

MSEHistoryCallback(x_star::Vector{Float64}) =
    MSEHistoryCallback(copy(x_star), Int[], Int[], Float64[], Float64[])

function on_event!(cb::MSEHistoryCallback, state::SolverState, event::Symbol)
    if event === :iter
        push!(cb.ks, state.k)
        push!(cb.f_evals, state.f_evals)
        push!(cb.elapsed, state.elapsed)
        push!(cb.values, mse(state.x, cb.x_star))
    end
    return nothing
end

function canonical_maxiter(mse_tol::Float64)
    mse_tol > 0 || throw(ArgumentError("MSE tolerance must be positive"))
    return round(Int, NMAX_REF * log10(1 / mse_tol) / log10(1 / MSE_TOL_REF))
end

function parse_cli(args)
    opts, flags = Dict{String,String}(), Set{String}()
    for arg in args
        startswith(arg, "--") || continue
        parts = split(arg[3:end], "="; limit = 2)
        length(parts) == 2 ? (opts[parts[1]] = parts[2]) : push!(flags, parts[1])
    end
    return opts, flags
end

function parse_list(text::String, known::Dict)
    out = String[]
    for item in split(text, ',')
        item = strip(item)
        isempty(item) && continue
        haskey(known, item) || throw(ArgumentError("Unknown option '$item'"))
        push!(out, item)
    end
    isempty(out) && throw(ArgumentError("Expected at least one item"))
    return out
end

function read_config(args)
    opts, flags = parse_cli(args)
    quick = "quick" in flags
    cases = haskey(opts, "cases") ? parse_list(opts["cases"], CASE_BY_NAME) : collect(keys(CASE_BY_NAME))
    methods = haskey(opts, "methods") ? parse_list(opts["methods"], METHOD_BY_NAME) : [name(T) for T in METHOD_TYPES]
    mse_tol = haskey(opts, "mse-tol") ? parse(Float64, opts["mse-tol"]) : MSE_TOL_REF
    cfg = (
        cases = cases, methods = methods,
        datasets = haskey(opts, "datasets") ? parse(Int, opts["datasets"]) : DEFAULT_DATASETS,
        initial_points = haskey(opts, "initial-points") ? parse(Int, opts["initial-points"]) : DEFAULT_INITIAL_POINTS,
        reps = haskey(opts, "reps") ? parse(Int, opts["reps"]) : (quick ? 1 : DEFAULT_REPS),
        mse_tol = mse_tol,
        maxiter = haskey(opts, "maxiter") ? parse(Int, opts["maxiter"]) : canonical_maxiter(mse_tol),
        consec = haskey(opts, "consec") ? parse(Int, opts["consec"]) : 2,
        gamma = haskey(opts, "gamma") ? parse(Float64, opts["gamma"]) : GAMMA_REF,
        noise_var = haskey(opts, "noise-var") ? parse(Float64, opts["noise-var"]) : NOISE_VAR_REF,
        force = "force" in flags, summary = "summary" in flags, production = !quick,
    )
    cfg.datasets >= 1 && cfg.initial_points >= 1 && cfg.reps >= 1 || throw(ArgumentError("counts must be positive"))
    cfg.datasets == 1 || throw(ArgumentError("This benchmark uses one fixed dataset per case; got datasets=$(cfg.datasets)"))
    cfg.mse_tol > 0 && cfg.maxiter >= 1 && cfg.consec >= 1 && cfg.gamma > 0 && cfg.noise_var >= 0 ||
        throw(ArgumentError("invalid numerical configuration"))
    return cfg
end

case_title(case) = "M=$(case.M), N=$(case.N), K=$(case.K)"
dataset_seed(case, dataset_idx) = DATA_SEED_BASE + 10_000 * (case.M + case.N + case.K) + dataset_idx

function build_problem(case; gamma::Float64, noise_var::Float64, data_seed::Int, n_inits::Int)
    M, N, K = case.M, case.N, case.K
    M >= 2 && N > M && 1 <= K < N || throw(ArgumentError("Require M >= 2, N > M, and 1 <= K < N"))
    rng = Xoshiro(UInt64(10_000_000 + data_seed))
    # Original compressed-sensing construction: C*C' = I_M, so the forward
    # operator has Lipschitz constant one and the old solver presets apply.
    C0 = randn(rng, M, N)
    qthin = Matrix(qr(C0').Q)[:, 1:M]
    C = collect(qthin')
    x_star = zeros(Float64, N)
    support = randperm(rng, N)[1:K]
    x_star[support] .= rand(rng, (-1.0, 1.0), K)
    noise = sqrt(noise_var) .* randn(rng, M)     # Normal(0, noise_var I)
    y = C * x_star + noise
    B_fn = let C = C, y = y; x -> C' * (C * x - y); end
    resolvent_A_fn = let gamma = gamma; (x, rho) -> soft_thresholding(x, rho * gamma); end
    native_fn = (x, x_prev) -> isempty(x_prev) ? Inf : norm(x - x_prev)
    initial_points = InitialPoint[]
    for i in 1:n_inits
        push!(initial_points, InitialPoint("seed$i", i, 0.1 .* randn(rng, N)))
    end
    metadata = (C=C, y=y, gamma=gamma, L=1.0, M=M, N=N, K=K, x_star=x_star,
                noise_var=noise_var, data_seed=data_seed, case_id=case.problem)
    return TestProblem(4, "CompressedSensing_MSE", N, B_fn, resolvent_A_fn,
                       native_fn, x_star, initial_points, metadata)
end

function build_algorithm(method_name::String)
    method_name == "AEFBFP" && return AEFBFP(:P4)
    method_name == "VAFBS" && return VAFBS(:paper)
    method_name == "MDITSM" && return MDITSM(:paper)
    method_name == "RFBSM" && return RFBSM(:paper)
    method_name == "IRFBSM" && return IRFBSM(:paper)
    method_name == "IFRAB" && return IFRAB(:paper)
    error("Unsupported method '$method_name'")
end

make_stopping(prob, cfg) = (MSEStopping(prob.exact_x, cfg.mse_tol; consec=cfg.consec), MaxIterStopping(cfg.maxiter), NanStopping())

const MSE_HISTORY_SQL = """
CREATE TABLE IF NOT EXISTS mse_history (
    script TEXT NOT NULL, config_hash TEXT NOT NULL, problem TEXT NOT NULL,
    dimension INTEGER NOT NULL, init_point TEXT NOT NULL, seed_idx INTEGER NOT NULL,
    production INTEGER NOT NULL, k INTEGER NOT NULL, f_evals INTEGER, elapsed REAL,
    mse REAL NOT NULL,
    PRIMARY KEY (script, config_hash, problem, dimension, init_point, production, k)
)
"""

function ensure_mse_tables!(db)
    DBInterface.execute(db, MSE_HISTORY_SQL)
end

function experiment_hash(alg, case, cfg)
    _, base = make_config_hash(alg, case.problem, cfg.mse_tol, cfg.maxiter)
    input = base * "|stopping=MSE|consec=$(cfg.consec)|gamma=$(repr(cfg.gamma))|noise_var=$(repr(cfg.noise_var))|signal=K_sparse_Rademacher|matrix=row_orthonormal|formulation=original_compressed_sensing|K=$(case.K)"
    return bytes2hex(sha256(input))[1:12], input
end

function insert_mse_history!(db, hash, case, init, production, cb)
    isempty(cb.ks) && return
    p = production ? 1 : 0
    # Individual statements avoid SQLite's open-statement commit restriction
    # when this script shares the generic benchmark schema.
    DBInterface.execute(db, "DELETE FROM mse_history WHERE script=? AND config_hash=? AND problem=? AND dimension=? AND init_point=? AND production=?",
                        (SCRIPT_ID, hash, case.problem, case.N, init.label, p)) |> DataFrame
    for i in eachindex(cb.ks)
        DBInterface.execute(db, "INSERT INTO mse_history VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                            (SCRIPT_ID, hash, case.problem, case.N, init.label, init.seed_idx, p,
                             cb.ks[i], cb.f_evals[i], cb.elapsed[i], cb.values[i])) |> DataFrame
    end
    return nothing
end

"Local result writer that fully consumes its SQLite statement before history insertion."
function insert_mse_result!(db, hash, case, init, run_id, result, final_mse, production)
    DBInterface.execute(db, """
        INSERT OR REPLACE INTO results
            (script, config_hash, problem, dimension, init_point, seed_idx, run_id,
             converged, iterations, f_evals, cpu_time, flag,
             residual, scaled_residual, native_residual, production, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (SCRIPT_ID, hash, case.problem, case.N, init.label, init.seed_idx, run_id,
          result.converged ? 1 : 0, result.iterations, result.f_evals, result.cpu_time,
          string(result.flag), result.residual, result.scaled_residual, final_mse,
          production ? 1 : 0, Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))) |> DataFrame
    return nothing
end

function row_done(db, hash, case, init, production)
    return is_done(db, hash, case.problem, case.N, init.label; script=SCRIPT_ID, production=production)
end

function summarize(db, cfg, tee)
    println(tee, "\n--- MSE-stopping summary ---")
    @printf(tee, "  %-16s %-8s %10s %12s %12s %10s\n", "case", "method", "solved", "med_iter", "med_eval", "med_cpu")
    for case_id in cfg.cases, method in cfg.methods
        case = CASE_BY_NAME[case_id]
        hash, _ = experiment_hash(build_algorithm(method), case, cfg)
        df = DBInterface.execute(db, """
            SELECT r.converged, r.iterations, r.f_evals, r.cpu_time FROM results r
            JOIN configs c ON c.config_hash=r.config_hash
            WHERE r.script=? AND r.production=? AND r.problem=? AND c.method=? AND r.config_hash=?
        """, (SCRIPT_ID, cfg.production ? 1 : 0, case_id, method, hash)) |> DataFrame
        nrow(df) == 0 && continue
        solved = df[df.converged .== 1, :]
        if nrow(solved) == 0
            @printf(tee, "  %-16s %-8s %4d/%-5d %12s %12s %10s\n", case_id, method, 0, nrow(df), "DNC", "DNC", "DNC")
        else
            @printf(tee, "  %-16s %-8s %4d/%-5d %12d %12d %10.4f\n", case_id, method, nrow(solved), nrow(df),
                    round(Int, median(solved.iterations)), round(Int, median(solved.f_evals)), median(solved.cpu_time))
        end
    end
end

"Write per-case and overall rankings using median final MSE (smaller is better)."
function write_mse_ranking_summary(db, cfg, tee)
    rows = NamedTuple[]
    for case_id in cfg.cases
        case_rows = NamedTuple[]
        for method in cfg.methods
            case = CASE_BY_NAME[case_id]
            hash, _ = experiment_hash(build_algorithm(method), case, cfg)
            values = DBInterface.execute(db, """
                SELECT native_residual FROM results
                WHERE script=? AND production=? AND problem=? AND config_hash=? AND native_residual IS NOT NULL
            """, (SCRIPT_ID, cfg.production ? 1 : 0, case_id, hash)) |> DataFrame
            nrow(values) == 0 && continue
            final_mses = Float64.(values.native_residual)
            push!(case_rows, (case=case_id, method=method, median_final_mse=median(final_mses),
                              mean_final_mse=mean(final_mses), runs=length(final_mses)))
        end
        sort!(case_rows, by=row -> row.median_final_mse)
        for (rank, row) in enumerate(case_rows)
            push!(rows, merge(row, (case_rank=rank,)))
        end
    end

    isempty(rows) && return nothing
    detail = DataFrame(rows)
    method_rows = NamedTuple[]
    for method in sort(unique(String.(detail.method)))
        sub = detail[detail.method .== method, :]
        push!(method_rows, (method=method, cases_ranked=nrow(sub),
                            average_rank=mean(Float64.(sub.case_rank)),
                            average_median_final_mse=mean(Float64.(sub.median_final_mse))))
    end
    overall = DataFrame(method_rows)
    sort!(overall, [:average_rank, :average_median_final_mse])
    overall.overall_rank = collect(1:nrow(overall))
    CSV.write(RANKING_CSV, overall)

    println(tee, "\n--- MSE ranking summary (lower final MSE is better) ---")
    @printf(tee, "  %-5s %-8s %12s %18s\n", "rank", "method", "avg_rank", "avg_median_MSE")
    for row in eachrow(overall)
        @printf(tee, "  %-5d %-8s %12.3f %18.3e\n", row.overall_rank, row.method,
                row.average_rank, row.average_median_final_mse)
    end
    println(tee, "  Saved ranking CSV: $(RANKING_CSV)")
    return overall
end

function benchmark_main(args=ARGS)
    cfg = read_config(args)
    logpath, tee, _ = setup_logging("mse_s30_benchmark"; logdir=LOGDIR)
    db = open_db(DB_PATH); ensure_mse_tables!(db)
    try
        println(tee, "MSE-stopping compressed-sensing benchmark")
        println(tee, "  cases=$(join(cfg.cases, ',')); methods=$(join(cfg.methods, ','))")
        @printf(tee, "  mse_tol=%.1e; noise_var=%.1e; gamma=%.1e\n", cfg.mse_tol, cfg.noise_var, cfg.gamma)
        println(tee, "  MSE must remain below tolerance for $(cfg.consec) consecutive iterations; maxiter=$(cfg.maxiter)")
        if cfg.summary
            summarize(db, cfg, tee)
            write_mse_ranking_summary(db, cfg, tee)
            return nothing
        end
        run_id = "mse_s30_" * Dates.format(now(), "yyyymmdd_HHMMSS")
        for case_id in cfg.cases
            case = CASE_BY_NAME[case_id]
            println(tee, "\n[$case_id] $(case_title(case))")
            # One deterministic dataset per case, shared by every method and start.
            prob = build_problem(case; gamma=cfg.gamma, noise_var=cfg.noise_var,
                                 data_seed=dataset_seed(case, 1), n_inits=cfg.initial_points)
            for method in cfg.methods
                alg = build_algorithm(method)
                hash, input = experiment_hash(alg, case, cfg)
                ensure_config!(db, alg, case_id, cfg.mse_tol, cfg.maxiter, hash, input)
                println(tee, "  $method")
                for init in prob.initial_points
                    !cfg.force && row_done(db, hash, case, init, cfg.production) && (println(tee, "    $(init.label): skipped"); continue)
                    cpus = Float64[]; history_cb = nothing; local result::SolverResult
                    for rep in 1:cfg.reps
                        recorder = MSEHistoryCallback(prob.exact_x)
                        observers = (rep == 1 && init.seed_idx == REF_INIT) ? (recorder,) : ()
                        result = solve(alg, prob, copy(init.x0); stopping=make_stopping(prob, cfg), observers=observers)
                        rep == 1 && init.seed_idx == REF_INIT && (history_cb = recorder)
                        push!(cpus, result.cpu_time)
                    end
                    result_med = make_result(converged=result.converged, iterations=result.iterations, f_evals=result.f_evals,
                        cpu_time=median(cpus), x=result.x, flag=result.flag, residual=result.residual, scaled_residual=result.scaled_residual)
                    final_mse = mse(result.x, prob.exact_x)
                    insert_mse_result!(db, hash, case, init, run_id, result_med, final_mse, cfg.production)
                    history_cb !== nothing && insert_mse_history!(db, hash, case, init, cfg.production, history_cb)
                    @printf(tee, "    %-7s conv=%-5s iter=%5d fe=%5d final_MSE=%10.3e cpu=%8.4f\n",
                            init.label, string(result.converged), result.iterations, result.f_evals, final_mse, median(cpus))
                end
            end
        end
        summarize(db, cfg, tee)
        write_mse_ranking_summary(db, cfg, tee)
    finally
        SQLite.close(db); teardown_logging(tee, logpath)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    benchmark_main()
end
