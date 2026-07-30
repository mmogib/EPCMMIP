# ============================================================================
# s30_benchmark.jl
# ============================================================================
#
# Purpose
#   Run the compressed-sensing benchmark using the same structured workflow as
#   the optimal-control scripts, but with family-wise presets:
#     - AEFBFP uses the locally tuned compressed-sensing winner written by s20
#     - VAFBS, MDITSM, MFRBSM, IMTTM, IFRAB, SFRBM use their source-paper presets
#
# Mathematical model
#   For each case (M, N, k), we solve the LASSO-type monotone inclusion
#       0 in A(x) + B(x)
#   associated with
#       min_x 0.5 * ||C*x - y||^2 + gamma * ||x||_1,
#   where A = gamma * partial ||x||_1 and B(x) = C' * (C*x - y).
#   All methods use the common LASSO fixed-point residual for stopping.
#
# Main outputs
#   - results/compressed_sensing/experiments.db
#   - results/compressed_sensing/logs/log_s30_benchmark_*.txt
#   - representative history rows (dataset 1, init 1 for each method x case)
#
# How to run
#   julia --project=. scripts/compressed_sensing/s30_benchmark.jl
#   julia --project=. scripts/compressed_sensing/s30_benchmark.jl --quick
#   julia --project=. scripts/compressed_sensing/s30_benchmark.jl --cases=M256_N512_k30,M512_N1024_k50
#   julia --project=. scripts/compressed_sensing/s30_benchmark.jl --methods=MTTM,IFRAB
#   julia --project=. scripts/compressed_sensing/s30_benchmark.jl --summary
#   julia --project=. scripts/compressed_sensing/s30_benchmark.jl --force
#
# ============================================================================

include(joinpath(@__DIR__, "..", "..", "src", "includes.jl"))

const PROBLEM_NAME = "compressed_sensing"
const RESULT_ROOT = joinpath(JCODE_ROOT, "results", PROBLEM_NAME)
const DB_PATH = joinpath(RESULT_ROOT, "experiments.db")
const LOGDIR = joinpath(RESULT_ROOT, "logs")
const FIGDIR = joinpath(RESULT_ROOT, "figures")

const METHOD_TYPES = (AEFBFP, VAFBS, MDITSM, RFBSM, IRFBSM, IFRAB)
const METHOD_BY_NAME = Dict(name(T) => T for T in METHOD_TYPES)

const DEFAULT_CASES = [
    (problem = "M256_N512_k30", M = 256, N = 512,  k = 30),
    (problem = "M256_N512_k50", M = 256, N = 512,  k = 50),
    (problem = "M512_N1024_k50", M = 512, N = 1024, k = 50),
    (problem = "M512_N1024_k80", M = 512, N = 1024, k = 80),
]
const CASE_BY_NAME = Dict(case.problem => case for case in DEFAULT_CASES)

const DEFAULT_DATASETS = 1
const DEFAULT_INITIAL_POINTS = 10
const DEFAULT_REPS = 3
const REF_DATASET = 1
const REF_INIT = 1
const EPS_REF = 1.0e-5
const NMAX_REF = 5000
const GAMMA_REF = 1.0e-3
const SNR_DB_REF = 40.0
const DATA_SEED_BASE = 20260701

const WINNER_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS tuned_winners (
    method TEXT NOT NULL,
    problem TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    summary_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (method, problem)
)
"""

const FINAL_METRICS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS cs_final_metrics (
    script              TEXT NOT NULL DEFAULT 's30',
    config_hash         TEXT NOT NULL,
    problem             TEXT NOT NULL,
    dimension           INTEGER NOT NULL,
    init_point          TEXT NOT NULL,
    seed_idx            INTEGER NOT NULL,
    production          INTEGER NOT NULL DEFAULT 0,
    objective           REAL NOT NULL,
    reconstruction_mse  REAL NOT NULL,
    common_residual     REAL NOT NULL,
    created_at          TEXT NOT NULL,
    PRIMARY KEY (script, config_hash, problem, dimension, init_point, production)
)
"""

mkpath(LOGDIR)
mkpath(FIGDIR)

function canonical_maxiter(eps::Float64)
    eps > 0 || throw(ArgumentError("eps must be > 0, got $eps"))
    return round(Int, NMAX_REF * (log10(1 / eps) / log10(1 / EPS_REF)))
end

# Purpose: Read benchmark command-line settings and separate options from flags.
#
# Input example:
#   args = ["--eps=1e-5", "--reps=3", "--quick"]
#
# Output:
#   opts  = Dict("eps" => "1e-5", "reps" => "3")
#   flags = Set(["quick"])
#
# Use `--name=value` for an option with a value, such as `--eps=1e-5`.
# Use `--name` for a flag, such as `--quick`.
#
# This function only reads and separates the inputs. It does not execute
# the benchmark, convert text values to numbers, or validate their values.
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

# Purpose: Convert a text option into a Julia Symbol used by preset/rule code.
#
# Input examples:
#   "paper"  or  ":paper"
#
# Output:
#   :paper
#
# A leading `:` is optional, so both input forms produce the same Symbol.
function parse_symbol_option(text::String)
    return startswith(text, ":") ? Symbol(text[2:end]) : Symbol(text)
end

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

# Purpose: Read and validate a comma-separated list of benchmark method names.
#
# Input example:
#   "AEFBFP,VAFBS,IFRAB"
#
# Output:
#   ["AEFBFP", "VAFBS", "IFRAB"]
#
# Spaces are removed. An empty list or an unknown method name raises an error.
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

# Purpose: Read and validate a comma-separated list of compressed-sensing cases.
#
# Input example:
#   "M256_N512_k30,M512_N1024_k50"
#
# Output:
#   ["M256_N512_k30", "M512_N1024_k50"]
#
# Spaces are removed. An empty list or an unknown case name raises an error.
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

# Purpose: Build one validated compressed-sensing benchmark configuration from
# command-line options and flags.
#
# Example command:
#   julia --project=. scripts/compressed_sensing/s30_benchmark.jl \
#       --methods=AEFBFP,VAFBS --cases=M256_N512_k30 \
#       --gamma=1e-3 --noise-var=1e-4 --reps=3 --force
#
# Options with values:
#   --methods=...  --cases=...  --datasets=...  --reps=...
#   --eps=...      --maxiter=...  --gamma=...  --snr-db=...
#   --aefbfp-preset=...  --aefbfp-round-digits=...
#
# Flags without values:
#   --quick  --force  --summary  --allow-untuned-aefbfp
#
# The consecutive-iteration stopping rule is fixed at `consec = 2`; it is not
# accepted as a command-line option so that every benchmark run uses one rule.
function read_benchmark_config(args)
    opts, flags = parse_cli(args)
    quick = "quick" in flags

    cases = haskey(opts, "cases") ? parse_case_list(opts["cases"]) :
            [case.problem for case in DEFAULT_CASES]
    methods = haskey(opts, "methods") ? parse_method_list(opts["methods"]) :
              [name(T) for T in METHOD_TYPES]
    datasets = haskey(opts, "datasets") ? parse(Int, opts["datasets"]) : DEFAULT_DATASETS
    reps = haskey(opts, "reps") ? parse(Int, opts["reps"]) : (quick ? 1 : DEFAULT_REPS)
    eps = haskey(opts, "eps") ? parse(Float64, opts["eps"]) : EPS_REF
    maxiter = haskey(opts, "maxiter") ? parse(Int, opts["maxiter"]) : canonical_maxiter(eps)
    consec = 2
    gamma = haskey(opts, "gamma") ? parse(Float64, opts["gamma"]) : GAMMA_REF
    snr_db = haskey(opts, "snr-db") ? parse(Float64, opts["snr-db"]) : SNR_DB_REF
    aefbfp_preset = haskey(opts, "aefbfp-preset") ? parse_symbol_option(opts["aefbfp-preset"]) : nothing
    aefbfp_round_digits = haskey(opts, "aefbfp-round-digits") ? parse(Int, opts["aefbfp-round-digits"]) : nothing

    datasets >= 1 || throw(ArgumentError("datasets must be >= 1, got $datasets"))
    reps >= 1 || throw(ArgumentError("reps must be >= 1, got $reps"))
    eps > 0 || throw(ArgumentError("eps must be > 0, got $eps"))
    maxiter >= 1 || throw(ArgumentError("maxiter must be >= 1, got $maxiter"))
    gamma > 0 || throw(ArgumentError("gamma must be > 0, got $gamma"))
    isfinite(snr_db) || throw(ArgumentError("snr-db must be finite, got $snr_db"))
    datasets == 1 || throw(ArgumentError("Compressed-sensing benchmark now uses exactly one fixed dataset per case; got datasets=$datasets"))
    aefbfp_preset === nothing || haskey(AEFBFP_PRESETS, aefbfp_preset) ||
        throw(ArgumentError("Unknown AEFBFP preset :$(aefbfp_preset). Known: $(join(sort(collect(keys(AEFBFP_PRESETS))), ", "))"))
    aefbfp_round_digits === nothing || aefbfp_round_digits >= 0 ||
        throw(ArgumentError("aefbfp-round-digits must be >= 0, got $(aefbfp_round_digits)"))

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
        gamma = gamma,
        snr_db = snr_db,
        force = "force" in flags,
        summary = "summary" in flags,
        allow_untuned_aefbfp = "allow-untuned-aefbfp" in flags,
        aefbfp_preset = aefbfp_preset,
        aefbfp_round_digits = aefbfp_round_digits,
        production = !quick,
    )
end

function case_title(case)
    return "M=$(case.M), N=$(case.N), k=$(case.k)"
end

"""
    reconstruction_mse(x, x_star) -> Float64

Return the mean squared reconstruction error
`(1 / N) * sum_i (x[i] - x_star[i])^2` between a recovered signal and the
planted sparse signal.  Use this same definition in benchmark reporting and
figures.
"""
reconstruction_mse(x::AbstractVector, x_star::AbstractVector) = mean(abs2, x .- x_star)

"""
    cs_optimality_residual(x, B, resolvent_A; lambda=1.0) -> Float64

Return the unit-forward-step LASSO fixed-point residual
``||x - prox_{lambda*gamma*||.||_1}(x - lambda*B(x))||``.  The benchmark
protocol fixes `lambda = 1.0` for every method and instance; this reporting
residual must not depend on the problem Lipschitz constant.
"""
function cs_optimality_residual(x::AbstractVector, B, resolvent_A; lambda::Float64 = 1.0)
    Bx = B(x)
    prox = resolvent_A(x .- lambda .* Bx, lambda)
    return norm(x .- prox)
end

function cs_optimality_residual(x::AbstractVector, prob::TestProblem; lambda::Float64 = 1.0)
    return cs_optimality_residual(x, prob.B, prob.resolvent_A; lambda = lambda)
end

"Return the LASSO fixed-point residual at the conventional safe measurement step 1/L."
cs_scaled_optimality_residual(x::AbstractVector, prob::TestProblem) =
    cs_optimality_residual(x, prob; lambda = inv(prob.metadata.L))

"Return the LASSO objective 0.5*||Cx-y||² + gamma*||x||₁."
function cs_objective(x::AbstractVector, prob::TestProblem)
    meta = prob.metadata
    return 0.5 * sum(abs2, meta.C * x .- meta.y) + meta.gamma * norm(x, 1)
end

function dataset_seed(case, dataset_idx::Int)
    return DATA_SEED_BASE + 10_000 * (case.M + case.N + case.k) + dataset_idx
end

function build_problem(case; gamma::Float64, snr_db::Float64, data_seed::Int,
                       n_inits::Int = DEFAULT_INITIAL_POINTS)
    M = case.M
    N = case.N
    k = case.k

    M >= 2 || throw(ArgumentError("M must be >= 2, got $M"))
    N > M || throw(ArgumentError("Need N > M for compressed sensing, got M=$M, N=$N"))
    1 <= k < N || throw(ArgumentError("k must satisfy 1 <= k < N, got k=$k, N=$N"))
    n_inits >= 1 || throw(ArgumentError("n_inits must be >= 1, got $n_inits"))

    rng = Xoshiro(UInt64(10_000_000 + data_seed))

    C = randn(rng, M, N)  # iid Normal(0,1), deliberately not row-normalized

    x_star = zeros(Float64, N)
    support = randperm(rng, N)[1:k]
    x_star[support] .= 2.0 .* rand(rng, k) .- 1.0

    clean = C * x_star
    raw_noise = randn(rng, M)
    noise = sqrt(sum(abs2, clean) / (10.0^(snr_db / 10) * sum(abs2, raw_noise))) .* raw_noise
    y = clean .+ noise
    L = opnorm(C, 2)^2

    B_fn = let C = C, y = y
        x -> C' * (C * x .- y)
    end

    resolvent_A_fn = let gamma = gamma
        (x, rho) -> soft_thresholding(x, rho * gamma)
    end

    # Common, method-independent LASSO optimality residual.  `x_prev` is
    # unused because the stopping quantity depends only on the current point.
    native_residual_fn = let B_fn = B_fn, resolvent_A_fn = resolvent_A_fn, L = L
        (x, x_prev) -> cs_optimality_residual(x, B_fn, resolvent_A_fn; lambda = inv(L))
    end

    metadata = (
        C = C,
        y = y,
        gamma = gamma,
        L = L,
        M = M,
        N = N,
        k = k,
        x_star = x_star,
        snr_db = snr_db,
        realized_snr_db = 20.0 * log10(norm(clean) / norm(noise)),
        data_seed = data_seed,
        case_id = case.problem,
    )

    # Initial points: fixed seeded random starts for the SAME dataset. This
    # keeps the problem instance fixed while still testing robustness to x0.
    initial_points = InitialPoint[]
    for n in 1:n_inits
        x0 = 0.1 .* randn(rng, N)
        push!(initial_points, InitialPoint("seed$n", n, x0))
    end

    return TestProblem(
        4,
        "CompressedSensing_LocalP4",
        N,
        B_fn,
        resolvent_A_fn,
        native_residual_fn,
        x_star,
        initial_points,
        metadata,
    )
end

function ensure_local_tables!(db)
    DBInterface.execute(db, WINNER_TABLE_SQL)
    DBInterface.execute(db, FINAL_METRICS_TABLE_SQL)
    return nothing
end

function insert_final_metrics!(db, hash::String, problem::String, dim::Int,
                               init::String, seed_idx::Int;
                               objective::Float64, reconstruction_mse::Float64,
                               common_residual::Float64, script::String = "s30",
                               production::Bool = true)
    DBInterface.execute(db, """
        INSERT OR REPLACE INTO cs_final_metrics
            (script, config_hash, problem, dimension, init_point, seed_idx, production,
             objective, reconstruction_mse, common_residual, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (script, hash, problem, dim, init, seed_idx, production ? 1 : 0,
           objective, reconstruction_mse, common_residual,
           Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS")))
    return nothing
end

function decode_symbol(text::AbstractString)
    startswith(text, ":") ? Symbol(text[2:end]) : Symbol(text)
end

# Purpose: Reconstruct usable AEFBFP parameters from JSON text stored in the database.
#
# Input example:
#   JSON containing "mu": "0.32" and "xi_rule": "power"
#
# Output example:
#   (mu = 0.32, xi_rule = :power, ...)
#
# Numeric strings become Float64 values and rule names become Julia Symbols.
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

function resolved_aefbfp_params(db; allow_untuned::Bool = false,
                                aefbfp_preset::Union{Nothing,Symbol} = nothing,
                                round_digits::Union{Nothing,Int} = nothing)
    params = if aefbfp_preset === nothing
        allow_untuned && return nothing
        row = DBInterface.execute(db, """
            SELECT c.params_json AS params_json
            FROM tuned_winners tw
            JOIN configs c ON c.config_hash = tw.config_hash
            WHERE tw.method = 'AEFBFP' AND tw.problem = ?
        """, (PROBLEM_NAME,)) |> DataFrame
        nrow(row) == 0 && error("No tuned AEFBFP winner found in $(DB_PATH). Run s20_aefbfp_parameter_search.jl first, or pass --allow-untuned-aefbfp for exploratory runs.")
        parse_aefbfp_params(row.params_json[1])
    else
        AEFBFP_PRESETS[aefbfp_preset]
    end
    round_digits === nothing || (params = round_namedtuple_values(params, round_digits))
    return params
end

function tuned_aefbfp_from_db(db; allow_untuned::Bool = false, round_digits::Union{Nothing,Int} = nothing)
    ensure_local_tables!(db)
    row = DBInterface.execute(db, """
        SELECT c.params_json AS params_json
        FROM tuned_winners tw
        JOIN configs c ON c.config_hash = tw.config_hash
        WHERE tw.method = 'AEFBFP' AND tw.problem = ?
    """, (PROBLEM_NAME,)) |> DataFrame

    if nrow(row) == 0
        allow_untuned && return AEFBFP(:P3)
        error("No tuned AEFBFP winner found in $(DB_PATH). Run s20_aefbfp_parameter_search.jl first, or pass --allow-untuned-aefbfp for exploratory runs.")
    end

    params = resolved_aefbfp_params(db; allow_untuned = allow_untuned, round_digits = round_digits)
    return AEFBFP(; params...)
end

function build_algorithm(db, method_name::AbstractString; allow_untuned_aefbfp::Bool = false,
                         aefbfp_preset::Union{Nothing,Symbol} = nothing,
                         aefbfp_round_digits::Union{Nothing,Int} = nothing)
    if method_name == "AEFBFP"
        if aefbfp_preset === nothing
            return tuned_aefbfp_from_db(db; allow_untuned = allow_untuned_aefbfp, round_digits = aefbfp_round_digits)
        end
        params = AEFBFP_PRESETS[aefbfp_preset]
        aefbfp_round_digits === nothing || (params = round_namedtuple_values(params, aefbfp_round_digits))
        return AEFBFP(; params...)
    end
    method_name == "VAFBS" && return VAFBS(:paper)
    method_name == "MDITSM" && return MDITSM(:paper)
    method_name == "RFBSM" && return RFBSM(:cs_benchmark)
    method_name == "IRFBSM" && return IRFBSM(:cs_benchmark)
    method_name == "MFRBSM" && return MFRBSM(:paper)
    method_name == "IFRAB" && return IFRAB(:paper)
    method_name == "SFRBM" && return SFRBM(:paper)
    method_name == "IMTTM" && return IMTTM(:paper)
    throw(ArgumentError("Unsupported method '$method_name'"))
end

function cs_config_hash(alg, case_id::String, cfg)
    _, base_input = make_config_hash(alg, case_id, cfg.eps, cfg.maxiter)
    input = base_input * "|matrix=iid_Normal_0_1|signal=k_sparse_Uniform[-1,1]|snr_db=$(repr(cfg.snr_db))|stopping=common_lasso_scaled_residual_lambda_invL|consec=$(cfg.consec)"
    return bytes2hex(sha256(input))[1:12], input
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
    placeholders = join(fill("?", length(hash_list)), ",")
    params = Any[cfg.production ? 1 : 0]
    append!(params, cfg.cases)
    append!(params, hash_list)

    case_placeholders = join(fill("?", length(cfg.cases)), ",")
    sql = """
        SELECT c.method AS method, r.config_hash AS config_hash,
               r.problem AS problem, r.dimension AS dimension, r.init_point AS init_point,
               r.seed_idx AS dataset_idx, r.production AS production, r.converged AS converged,
               r.iterations AS iterations, r.f_evals AS f_evals,
               r.cpu_time AS cpu_time, r.flag AS flag
        FROM results r
        JOIN configs c ON c.config_hash = r.config_hash
        WHERE r.script = 's30' AND r.production = ? AND r.problem IN ($case_placeholders)
          AND r.config_hash IN ($placeholders)
    """
    return DBInterface.execute(db, sql, Tuple(params)) |> DataFrame
end

function summarize_rows(df::DataFrame, tee, cfg)
    println(tee, "\n--- summary for $(PROBLEM_NAME) ---")
    @printf(tee, "  %-14s %-8s %10s %12s %12s %10s\n",
            "case", "method", "solved", "med_iter", "med_eval", "med_cpu")
    println(tee, "  " * "-"^74)

    for case_id in cfg.cases
        for method in cfg.methods
            sub = df[(df.problem .== case_id) .& (df.method .== method), :]
            n_total = nrow(sub)
            n_total == 0 && continue
            conv = sub[sub.converged .== 1, :]
            n_solved = nrow(conv)

            if n_solved == 0
                @printf(tee, "  %-14s %-8s %4d/%-5d %12s %12s %10s\n",
                        case_id, method, n_solved, n_total, "DNC", "DNC", "DNC")
                continue
            end

            med_iter = median(Float64.(conv.iterations))
            med_eval = median(Float64.(conv.f_evals))
            med_cpu = median(Float64.(conv.cpu_time))
            # Keep half-integer medians visible: independently rounding Iter and
            # F-evals can hide their exact accounting difference.
            @printf(tee, "  %-14s %-8s %4d/%-5d %12.1f %12.1f %10.4f\n",
                    case_id, method, n_solved, n_total,
                    med_iter, med_eval, med_cpu)
        end
    end
end

function benchmark_main(args = ARGS; script_name::AbstractString = "s30_benchmark")
    cfg = read_benchmark_config(args)

    logpath, tee, _ = setup_logging(String(script_name); logdir = LOGDIR)
    db = open_db(DB_PATH)
    ensure_local_tables!(db)

    try
        aefbfp_label = cfg.aefbfp_preset === nothing ? "(tuned winner)" : string(cfg.aefbfp_preset)
        aefbfp_round_label = cfg.aefbfp_round_digits === nothing ? "(full precision)" : string(cfg.aefbfp_round_digits)
        aefbfp_params = resolved_aefbfp_params(db;
                                               allow_untuned = cfg.allow_untuned_aefbfp,
                                               aefbfp_preset = cfg.aefbfp_preset,
                                               round_digits = cfg.aefbfp_round_digits)
        println(tee, "="^78)
        println(tee, "  Benchmark: $(PROBLEM_NAME)")
        println(tee, "="^78)
        println(tee, "  db_path     : $(DB_PATH)")
        println(tee, "  cases       : $(join(cfg.cases, ", "))")
        println(tee, "  methods     : $(join(cfg.methods, ", "))")
        println(tee, "  datasets    : $(cfg.datasets)")
        println(tee, "  initial_points : $(cfg.initial_points)")
        println(tee, "  reps        : $(cfg.reps)")
        @printf(tee, "  eps         : %.1e\n", cfg.eps)
        println(tee, "  maxiter     : $(cfg.maxiter)")
        println(tee, "  consec      : $(cfg.consec)")
        @printf(tee, "  gamma       : %.1e\n", cfg.gamma)
        @printf(tee, "  snr_db      : %.1f dB\n", cfg.snr_db)
        println(tee, "  aefbfp_preset : $(aefbfp_label)")
        println(tee, "  aefbfp_round_digits : $(aefbfp_round_label)")
        println(tee, "  aefbfp_params : $(aefbfp_params)")
        println(tee, "  production  : $(cfg.production ? 1 : 0)")
        println(tee, "  force       : $(cfg.force)")
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
                alg = build_algorithm(db, method_name;
                                      allow_untuned_aefbfp = cfg.allow_untuned_aefbfp,
                                      aefbfp_preset = cfg.aefbfp_preset,
                                      aefbfp_round_digits = cfg.aefbfp_round_digits)
                hash, hash_input = cs_config_hash(alg, case_id, cfg)
                ensure_config!(db, alg, case_id, cfg.eps, cfg.maxiter, hash, hash_input)
                prune_stale_init_rows!(db, hash, case_id, case.N, cfg.initial_points;
                                       script = "s30", production = cfg.production)

                println(tee, "  $(method_name)")

                for dataset_idx in 1:cfg.datasets
                    data_seed = dataset_seed(case, dataset_idx)
                    prob = build_problem(case; gamma = cfg.gamma, snr_db = cfg.snr_db,
                                         data_seed = data_seed, n_inits = cfg.initial_points)
                    for init in prob.initial_points
                        if !cfg.force
                            row = DBInterface.execute(db, """
                                SELECT COUNT(*) AS n
                                FROM results
                                WHERE config_hash = ? AND problem = ? AND dimension = ? AND init_point = ?
                                  AND seed_idx = ? AND script = 's30' AND production = ?
                            """, (hash, case_id, case.N, init.label, init.seed_idx, cfg.production ? 1 : 0)) |> DataFrame
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
                        objective = cs_objective(result.x, prob)
                        mse = reconstruction_mse(result.x, prob.metadata.x_star)
                        common_residual = cs_scaled_optimality_residual(result.x, prob)
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

                        insert_result!(db, hash, case_id, case.N, init.label, init.seed_idx, run_id, result_med;
                                       script = "s30", native_residual = native_residual,
                                       production = cfg.production)
                        insert_final_metrics!(db, hash, case_id, case.N, init.label, init.seed_idx;
                                              objective = objective, reconstruction_mse = mse,
                                              common_residual = common_residual,
                                              script = "s30", production = cfg.production)

                        if !isempty(history)
                            insert_history!(db, hash, case_id, case.N, init.label, init.seed_idx, history;
                                            script = "s30", production = cfg.production)
                        end

                        @printf(tee,
                                "    dataset=%2d init=%-7s conv=%-5s  iter=%5d  fe=%5d  obj=%10.3e  mse=%10.3e  res=%10.3e  cpu=%8.4f\n",
                                dataset_idx, init.label, string(result.converged), result.iterations,
                                result.f_evals, objective, mse, common_residual, med_cpu)
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
