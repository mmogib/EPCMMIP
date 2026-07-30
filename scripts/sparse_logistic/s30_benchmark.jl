# ============================================================================
# s30_benchmark.jl
# ============================================================================
#
# Purpose
#   Run the sparse-logistic benchmark with a DB-backed workflow similar to
#   `scripts/compressed_sensing/s30_benchmark.jl`.
#
# Mathematical model
#   For each case `(m, N)` we solve the monotone inclusion
#       0 in A(x) + B(x)
#   associated with
#       min_x sum_i log(1 + exp(-b_i a_i' x)) + rho * ||x||_1,
#   where `A = partial (rho * ||x||_1)` and `B(x) = K' * sigmoid(Kx)`.
#
# Main outputs
#   - results/sparse_logistic/experiments.db
#   - results/sparse_logistic/logs/log_s30_benchmark_*.txt
#   - representative history rows (dataset 1, init 1 for each method x case)
#
# How to run
#   julia --project=. scripts/sparse_logistic/s30_benchmark.jl
#   julia --project=. scripts/sparse_logistic/s30_benchmark.jl --quick
#   julia --project=. scripts/sparse_logistic/s30_benchmark.jl --cases=m128_n64,m256_n128
#   julia --project=. scripts/sparse_logistic/s30_benchmark.jl --methods=EPCM,IFRAB
#   julia --project=. scripts/sparse_logistic/s30_benchmark.jl --summary
#   julia --project=. scripts/sparse_logistic/s30_benchmark.jl --force
#
# ============================================================================

include(joinpath(@__DIR__, "..", "..", "src", "includes.jl"))

const PROBLEM_NAME = "sparse_logistic"
const RESULT_ROOT = joinpath(JCODE_ROOT, "results", PROBLEM_NAME)
const DB_PATH = joinpath(RESULT_ROOT, "experiments.db")
const LOGDIR = joinpath(RESULT_ROOT, "logs")
const FIGDIR = joinpath(RESULT_ROOT, "figures")

const METHOD_TYPES = (AEFBFP, IFRAB, VAFBS, MDITSM, RFBSM, IRFBSM)
const METHOD_BY_NAME = Dict(name(T) => T for T in METHOD_TYPES)

const DEFAULT_CASES = [
    (problem = "m128_n64", m = 128, N = 64),
    (problem = "m256_n128", m = 256, N = 128),
    (problem = "m512_n256", m = 512, N = 256),
    (problem = "m1024_n512", m = 1024, N = 512),
]
const CASE_BY_NAME = Dict(case.problem => case for case in DEFAULT_CASES)

const DEFAULT_DATASETS = 1
const DEFAULT_INITIAL_POINTS = 10
const DEFAULT_REPS = 3
const REF_DATASET = 1
const REF_INIT = 1
const EPS_REF_LOCAL = 1.0e-6
const NMAX_REF_LOCAL = 5000
const RHO_SCALE_REF = 5.0e-3
const DATA_SEED_BASE = 20260716

mkpath(LOGDIR)
mkpath(FIGDIR)

function canonical_maxiter(eps::Float64)
    eps > 0 || throw(ArgumentError("eps must be > 0, got $eps"))
    return round(Int, NMAX_REF_LOCAL * (log10(1 / eps) / log10(1 / EPS_REF_LOCAL)))
end

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

function parse_method_list(text::String)
    vals = String[]
    for part in split(text, ",")
        item = strip(part)
        isempty(item) && continue
        haskey(METHOD_BY_NAME, item) || throw(ArgumentError("Unknown method '$item'"))
        push!(vals, item)
    end
    isempty(vals) && throw(ArgumentError("Expected at least one method in '$text'"))
    return vals
end

function parse_case_list(text::String)
    vals = String[]
    for part in split(text, ",")
        item = strip(part)
        isempty(item) && continue
        haskey(CASE_BY_NAME, item) || throw(ArgumentError("Unknown case '$item'"))
        push!(vals, item)
    end
    isempty(vals) && throw(ArgumentError("Expected at least one case in '$text'"))
    return vals
end

function read_benchmark_config(args)
    opts, flags = parse_cli(args)
    quick = "quick" in flags

    cases = haskey(opts, "cases") ? parse_case_list(opts["cases"]) :
            [case.problem for case in DEFAULT_CASES]
    methods = haskey(opts, "methods") ? parse_method_list(opts["methods"]) :
              [name(T) for T in METHOD_TYPES]
    datasets = haskey(opts, "datasets") ? parse(Int, opts["datasets"]) : DEFAULT_DATASETS
    reps = haskey(opts, "reps") ? parse(Int, opts["reps"]) : (quick ? 1 : DEFAULT_REPS)
    eps = haskey(opts, "eps") ? parse(Float64, opts["eps"]) : EPS_REF_LOCAL
    maxiter = haskey(opts, "maxiter") ? parse(Int, opts["maxiter"]) : canonical_maxiter(eps)
    consec = haskey(opts, "consec") ? parse(Int, opts["consec"]) : 2
    rho_scale = haskey(opts, "rho-scale") ? parse(Float64, opts["rho-scale"]) : RHO_SCALE_REF

    datasets >= 1 || throw(ArgumentError("datasets must be >= 1, got $datasets"))
    reps >= 1 || throw(ArgumentError("reps must be >= 1, got $reps"))
    eps > 0 || throw(ArgumentError("eps must be > 0, got $eps"))
    maxiter >= 1 || throw(ArgumentError("maxiter must be >= 1, got $maxiter"))
    consec >= 1 || throw(ArgumentError("consec must be >= 1, got $consec"))
    rho_scale > 0 || throw(ArgumentError("rho-scale must be > 0, got $rho_scale"))

    return (
        quick = quick,
        cases = cases,
        methods = methods,
        datasets = datasets,
        initial_points = DEFAULT_INITIAL_POINTS,
        reps = reps,
        eps = eps,
        maxiter = maxiter,
        consec = consec,
        rho_scale = rho_scale,
        force = "force" in flags,
        summary = "summary" in flags,
        production = !quick,
    )
end

function case_title(case)
    return "m=$(case.m), N=$(case.N)"
end

function dataset_seed(case, dataset_idx::Int)
    return DATA_SEED_BASE + 10_000 * (case.m + case.N) + dataset_idx
end

function sigmoid_stable(t::Float64)
    return t >= 0.0 ? inv(1.0 + exp(-t)) : exp(t) / (1.0 + exp(t))
end

function build_problem(case; rho_scale::Float64, data_seed::Int,
                       n_inits::Int = DEFAULT_INITIAL_POINTS)
    m = case.m
    N = case.N

    m >= 4 || throw(ArgumentError("m must be >= 4, got $m"))
    N >= 2 || throw(ArgumentError("N must be >= 2, got $N"))
    N < m || throw(ArgumentError("Need N < m for the sparse logistic benchmark, got N=$N, m=$m"))
    n_inits >= 1 || throw(ArgumentError("n_inits must be >= 1, got $n_inits"))

    rng = Xoshiro(UInt64(60_000_000 + data_seed))
    X = randn(rng, N, m)
    b = rand(rng, (-1.0, 1.0), N)
    K = (-b) .* X

    K_opnorm = opnorm(K, 2)
    rho = rho_scale * K_opnorm^2
    L = 0.25 * K_opnorm^2

    B_fn = let K = K
        x -> begin
            z = K * x
            sigma = similar(z)
            @inbounds for i in eachindex(z)
                sigma[i] = sigmoid_stable(z[i])
            end
            return K' * sigma
        end
    end

    resolvent_A_fn = let rho = rho
        (x, step) -> soft_thresholding(x, step * rho)
    end

    native_residual_fn = let B_fn = B_fn, resolvent_A_fn = resolvent_A_fn
        (x, x_prev) -> norm(x .- resolvent_A_fn(x .- B_fn(x), 1.0))
    end

    metadata = (
        X = X,
        b = b,
        K = K,
        rho = rho,
        L = L,
        N = N,
        m = m,
        rho_scale = rho_scale,
        data_seed = data_seed,
        case_id = case.problem,
    )

    initial_points = InitialPoint[]
    for n in 1:n_inits
        x0 = -1.0 .+ 2.0 .* rand(rng, m)
        push!(initial_points, InitialPoint("seed$n", n, x0))
    end

    return TestProblem(
        60,
        "SparseLogistic_L1",
        m,
        B_fn,
        resolvent_A_fn,
        native_residual_fn,
        nothing,
        initial_points,
        metadata,
    )
end

function build_algorithm(method_name::AbstractString)
    method_name == "AEFBFP" && return AEFBFP(:P4)
    method_name == "EPCM" && return EPCM(:P4)
    method_name == "IFRAB" && return IFRAB(:paper)
    method_name == "VAFBS" && return VAFBS(:paper)
    method_name == "MDITSM" && return MDITSM(:paper)
    method_name == "RFBSM" && return RFBSM(:paper)
    method_name == "IRFBSM" && return IRFBSM(:paper)
    throw(ArgumentError("Unsupported method '$method_name'"))
end

function make_stopping(prob::TestProblem, eps::Float64, maxiter::Int; consec::Int)
    return (
        NativeResStopping(prob.native_residual, eps; consec = consec),
        MaxIterStopping(maxiter),
        NanStopping(),
    )
end

function prune_stale_init_rows!(db, hash::String, problem::String, dim::Int, n_inits::Int;
                                script::String = "s30", production::Bool = true)
    allowed_labels = Set("seed$n" for n in 1:n_inits)
    p = production ? 1 : 0

    rows = DBInterface.execute(db, """
        SELECT init_point, seed_idx
        FROM results
        WHERE script = ? AND config_hash = ? AND problem = ? AND dimension = ? AND production = ?
    """, (script, hash, problem, dim, p)) |> DataFrame

    for row in eachrow(rows)
        init_label = String(row.init_point)
        seed_idx = Int(row.seed_idx)
        keep = (1 <= seed_idx <= n_inits) && (init_label in allowed_labels)
        keep && continue

        DBInterface.execute(db, """
            DELETE FROM history
            WHERE script = ? AND config_hash = ? AND problem = ? AND dimension = ? AND init_point = ? AND production = ?
        """, (script, hash, problem, dim, init_label, p))

        DBInterface.execute(db, """
            DELETE FROM results
            WHERE script = ? AND config_hash = ? AND problem = ? AND dimension = ? AND init_point = ? AND production = ?
        """, (script, hash, problem, dim, init_label, p))
    end

    return nothing
end

function current_method_hashes(db, production::Bool, eps::Float64, maxiter::Int, cases)
    placeholders = join(fill("?", length(cases)), ",")
    params = Any[production ? 1 : 0, eps, maxiter]
    append!(params, cases)

    df = DBInterface.execute(db, """
        SELECT r.problem AS problem, c.method AS method, r.config_hash AS config_hash,
               MAX(r.created_at) AS created_at
        FROM results r
        JOIN configs c ON c.config_hash = r.config_hash
        WHERE r.script = 's30' AND r.production = ? AND c.eps = ? AND c.maxiter = ?
          AND r.problem IN ($placeholders)
        GROUP BY r.problem, c.method, r.config_hash
    """, Tuple(params)) |> DataFrame

    out = Dict{Tuple{String,String},String}()
    for problem in unique(String.(df.problem))
        pdf = df[df.problem .== problem, :]
        for method in unique(String.(pdf.method))
            sub = pdf[pdf.method .== method, :]
            nrow(sub) == 0 && continue
            sort!(sub, :created_at)
            out[(problem, method)] = String(sub.config_hash[end])
        end
    end
    return out
end

function load_result_rows(db, cfg)
    hashes = current_method_hashes(db, cfg.production, cfg.eps, cfg.maxiter, cfg.cases)
    isempty(hashes) && return DataFrame()

    hash_list = collect(values(hashes))
    hash_placeholders = join(fill("?", length(hash_list)), ",")
    case_placeholders = join(fill("?", length(cfg.cases)), ",")

    params = Any[cfg.production ? 1 : 0]
    append!(params, cfg.cases)
    append!(params, hash_list)

    sql = """
        SELECT c.method AS method, r.problem AS problem, r.dimension AS dimension,
               r.seed_idx AS dataset_idx, r.converged AS converged,
               r.iterations AS iterations, r.f_evals AS f_evals,
               r.cpu_time AS cpu_time, r.flag AS flag
        FROM results r
        JOIN configs c ON c.config_hash = r.config_hash
        WHERE r.script = 's30' AND r.production = ? AND r.problem IN ($case_placeholders)
          AND r.config_hash IN ($hash_placeholders)
    """
    return DBInterface.execute(db, sql, Tuple(params)) |> DataFrame
end

function summarize_rows(df::DataFrame, tee, cfg)
    println(tee, "\n--- summary for $(PROBLEM_NAME) ---")
    @printf(tee, "  %-12s %-8s %10s %12s %12s %10s\n",
            "case", "method", "solved", "med_iter", "med_eval", "med_cpu")
    println(tee, "  " * "-"^70)

    for case_id in cfg.cases
        for method in cfg.methods
            sub = df[(df.problem .== case_id) .& (df.method .== method), :]
            n_total = nrow(sub)
            n_total == 0 && continue
            conv = sub[sub.converged .== 1, :]
            n_solved = nrow(conv)

            if n_solved == 0
                @printf(tee, "  %-12s %-8s %4d/%-5d %12s %12s %10s\n",
                        case_id, method, n_solved, n_total, "DNC", "DNC", "DNC")
                continue
            end

            med_iter = median(Float64.(conv.iterations))
            med_eval = median(Float64.(conv.f_evals))
            med_cpu = median(Float64.(conv.cpu_time))
            @printf(tee, "  %-12s %-8s %4d/%-5d %12d %12d %10.4f\n",
                    case_id, method, n_solved, n_total,
                    Int(round(med_iter)), Int(round(med_eval)), med_cpu)
        end
    end
end

function benchmark_main(args = ARGS; script_name::AbstractString = "s30_benchmark")
    cfg = read_benchmark_config(args)

    logpath, tee, _ = setup_logging(String(script_name); logdir = LOGDIR)
    db = open_db(DB_PATH)

    try
        println(tee, "="^78)
        println(tee, "  Benchmark: $(PROBLEM_NAME)")
        println(tee, "="^78)
        println(tee, "  db_path     : $(DB_PATH)")
        println(tee, "  cases       : ", join(cfg.cases, ", "))
        println(tee, "  methods     : ", join(cfg.methods, ", "))
        println(tee, "  datasets    : ", cfg.datasets)
        println(tee, "  initial_points : ", cfg.initial_points)
        println(tee, "  reps        : ", cfg.reps)
        @printf(tee, "  eps         : %.1e\n", cfg.eps)
        println(tee, "  maxiter     : ", cfg.maxiter)
        println(tee, "  consec      : ", cfg.consec)
        @printf(tee, "  rho_scale   : %.3e\n", cfg.rho_scale)
        println(tee, "  production  : ", cfg.production ? 1 : 0)
        println(tee, "  force       : ", cfg.force)
        println(tee)

        if cfg.summary
            summarize_rows(load_result_rows(db, cfg), tee, cfg)
            return nothing
        end

        run_id = "s30_" * Dates.format(Dates.now(), "yyyymmdd_HHMMSS")

        for case_id in cfg.cases
            case = CASE_BY_NAME[case_id]
            println(tee, "\n[$(case_id)]  $(case_title(case))")

            for method_name in cfg.methods
                alg = build_algorithm(method_name)
                hash, hash_input = make_config_hash(alg, case_id, cfg.eps, cfg.maxiter)
                ensure_config!(db, alg, case_id, cfg.eps, cfg.maxiter, hash, hash_input)
                prune_stale_init_rows!(db, hash, case_id, case.m, cfg.initial_points;
                                       script = "s30", production = cfg.production)

                println(tee, "  $(method_name)")

                for dataset_idx in 1:cfg.datasets
                    data_seed = dataset_seed(case, dataset_idx)
                    prob = build_problem(case; rho_scale = cfg.rho_scale,
                                         data_seed = data_seed, n_inits = cfg.initial_points)

                    for init in prob.initial_points
                        if !cfg.force
                            row = DBInterface.execute(db, """
                                SELECT COUNT(*) AS n
                                FROM results
                                WHERE config_hash = ? AND problem = ? AND dimension = ? AND init_point = ?
                                  AND seed_idx = ? AND script = 's30' AND production = ?
                            """, (hash, case_id, case.m, init.label, init.seed_idx, cfg.production ? 1 : 0)) |> DataFrame
                            if nrow(row) == 1 && Int(row.n[1]) > 0
                                println(tee, "    dataset=$(dataset_idx) init=$(init.label)  skipped (already in DB)")
                                continue
                            end
                        end

                        cpus = Float64[]
                        history = IterRecord[]
                        local result::SolverResult
                        local native_residual::Float64

                        for rep in 1:cfg.reps
                            stopping = make_stopping(prob, cfg.eps, cfg.maxiter; consec = cfg.consec)
                            nrec = NativeResRecorder(prob.native_residual)
                            collect_hist = (rep == 1 && dataset_idx == REF_DATASET && init.seed_idx == REF_INIT)

                            if collect_hist
                                hc = HistoryCallback()
                                result = solve(alg, prob, copy(init.x0); stopping = stopping, observers = (hc, nrec))
                                history = hc.history
                            else
                                result = solve(alg, prob, copy(init.x0); stopping = stopping, observers = (nrec,))
                            end

                            native_residual = nrec.value
                            push!(cpus, result.cpu_time)
                        end

                        med_cpu = median(cpus)
                        result_med = make_result(
                            converged = result.converged,
                            iterations = result.iterations,
                            f_evals = result.f_evals,
                            cpu_time = med_cpu,
                            x = result.x,
                            flag = result.flag,
                            history = history,
                            residual = result.residual,
                            scaled_residual = result.scaled_residual,
                        )

                        insert_result!(db, hash, case_id, case.m, init.label, init.seed_idx, run_id, result_med;
                                       script = "s30", native_residual = native_residual,
                                       production = cfg.production)

                        if !isempty(history)
                            insert_history!(db, hash, case_id, case.m, init.label, init.seed_idx, history;
                                            script = "s30", production = cfg.production)
                        end

                        @printf(tee,
                                "    dataset=%2d init=%-7s conv=%-5s  iter=%5d  fe=%5d  nat=%10.3e  cpu=%8.4f\n",
                                dataset_idx, init.label, string(result.converged), result.iterations,
                                result.f_evals, native_residual, med_cpu)
                    end
                end
            end
        end

        summarize_rows(load_result_rows(db, cfg), tee, cfg)
    finally
        teardown_logging(tee, logpath)
        SQLite.close(db)
    end

    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    benchmark_main()
end
