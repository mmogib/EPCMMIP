# ============================================================================
# s20_aefbfp_parameter_search.jl
# ============================================================================
#
# Purpose
#   Tune AEFBFP for the compressed-sensing benchmark family, then store the
#   winning parameter set in the local `tuned_winners` table inside
#   `results/compressed_sensing/experiments.db`.
#
# Search design
#   - candidates: 20 by default
#   - cases: all default compressed-sensing cases
#   - dataset per case during tuning: 1 (same fixed dataset as the benchmark)
#   - initial points per case during tuning: 10 seeded random starts
#   - stopping quantity: the same native residual used later in s30_benchmark
#   - fixed benchmark settings reused from s30: eps, maxiter, gamma, SNR
#   - candidate generation: 5D Latin Hypercube over a hand-picked AEFBFP box
#
# Winner rule
#   Candidates are ranked by:
#     1. larger number of converged runs
#     2. smaller median iterations
#     3. smaller median operator evaluations
#     4. smaller median CPU time
#
# Main outputs
#   - tuning rows in `results/compressed_sensing/experiments.db`
#   - one AEFBFP winner row in the local `tuned_winners` table
#   - `results/compressed_sensing/logs/log_s20_aefbfp_parameter_search_*.txt`
#
# Execution order
#   1. Read CLI options into a compact search config
#   2. Generate an LHS matrix H on [0,1]^5
#   3. Convert each row H[i, :] into one AEFBFP parameter candidate
#   4. Run that candidate on every selected case x initial point
#   5. Score the candidate from the collected rows
#   6. Keep the best candidate and promote it into `tuned_winners`
#
# Helper map
#   - `read_search_config`      : parse and validate CLI options
#   - `lhs_unit`                : generate the unit LHS matrix H
#   - `sample_candidate_from_unit`
#                               : map one unit row to AEFBFP parameters
#   - `run_candidate_once`      : solve one case/initial-point pair
#   - `candidate_score`         : summarize one candidate's rows
#   - `promote_winner!`         : freeze the winner in the DB
#
# How to run
#   julia --project=. scripts/compressed_sensing/s20_aefbfp_parameter_search.jl
#       -> run the default search on all compressed-sensing cases
#   julia --project=. scripts/compressed_sensing/s20_aefbfp_parameter_search.jl --candidates=40
#       -> try 40 LHS candidates instead of the default 20
#   julia --project=. scripts/compressed_sensing/s20_aefbfp_parameter_search.jl --cases=M256_N512_k30,M512_N1024_k80
#       -> tune only on the listed cases, not on the full default case set
#   julia --project=. scripts/compressed_sensing/s20_aefbfp_parameter_search.jl --force
#       -> re-run rows even if matching tuning results are already in the DB
#
# ============================================================================

include(joinpath(@__DIR__, "s30_benchmark.jl"))

const SEARCH_CASES = [case.problem for case in DEFAULT_CASES]
const SEARCH_CANDIDATES = 20
const SEARCH_DATASET = 1
const SEARCH_LHS_SEED = 20260711

# AEFBFP compressed-sensing protocol: μ is searched over the requested range.
const AEFBFP_MU_RANGE = (0.1, 0.5)
const AEFBFP_TAU0_RANGE = (1.0e-4, 1.0)
# Fixed compressed-sensing sequences:
#   ξ_k = 1/k^3,  σ_k = 1/(1000k+1).
const AEFBFP_XI_EXP_RANGE = (1.01, 3.0)
const AEFBFP_SIGMA_EXP = 1.0       # retained only because AEFBFP stores this field
const AEFBFP_SIGMA_SCALE = 1.0     # ignored by :cs_inv_1000k_plus_1
const SEARCH_DIMS = 3

# ============================================================================
# Config parsing
# ============================================================================

"Read and validate CLI options for this search script."
function read_search_config(args)
    opts, flags = parse_cli(args)

    # Keep the tuning CLI intentionally small: only the core search knobs plus
    # `--force` are user-facing here.
    allowed_opts = Set(["candidates", "cases"])
    allowed_flags = Set(["force"])
    for key in keys(opts)
        key in allowed_opts ||
            throw(ArgumentError("Unsupported option '--$(key)=...'. Allowed options: --candidates, --cases"))
    end
    for flag in flags
        flag in allowed_flags ||
            throw(ArgumentError("Unsupported flag '--$(flag)'. Allowed flag: --force"))
    end

    # Step 1: read the small set of supported tuning options.
    candidates = haskey(opts, "candidates") ? parse(Int, opts["candidates"]) : SEARCH_CANDIDATES
    cases = haskey(opts, "cases") ? parse_case_list(opts["cases"]) : SEARCH_CASES

    # Step 2: keep the benchmark-side settings fixed, so every tuning run is
    # comparable and uses the same production-style stopping policy.
    datasets = 1
    eps = EPS_REF
    consec = 2
    gamma = GAMMA_REF
    snr_db = SNR_DB_REF
    initial_points = DEFAULT_INITIAL_POINTS
    maxiter = canonical_maxiter(eps)

    # Step 3: validate the final config values.
    candidates >= 1 || throw(ArgumentError("candidates must be >= 1"))
    eps > 0 || throw(ArgumentError("eps must be > 0"))
    maxiter >= 1 || throw(ArgumentError("maxiter must be >= 1"))
    consec >= 1 || throw(ArgumentError("consec must be >= 1"))
    gamma > 0 || throw(ArgumentError("gamma must be > 0"))
    isfinite(snr_db) || throw(ArgumentError("snr-db must be finite"))

    # Step 4: return one compact config object for the main search loop.
    return (
        candidates = candidates,
        cases = cases,
        datasets = datasets,
        initial_points = initial_points,
        eps = eps,
        maxiter = maxiter,
        consec = consec,
        gamma = gamma,
        snr_db = snr_db,
        force = "force" in flags,
    )
end

# ============================================================================
# Sampling helpers
# ============================================================================

"Map unit u in [0,1] to a linear range [a,b]."
@inline _lin(u, a, b) = a + u * (b - a)

"Map unit u in [0,1] to a log range [a,b] with a,b > 0."
@inline _log(u, a, b) = a * (b / a)^u

"""
    lhs_unit(rng, n, d) -> n x d Matrix

`n` = total candidate rows to generate.
`d` = number of tuned parameters, so the number of columns in the LHS matrix.

Latin Hypercube design on [0,1]^d. Each column is split into `n` equal bins,
the bins are randomly permuted, and one jittered point is drawn from each bin.
So every column gets one sample from every bin, which avoids clustering.
"""
function lhs_unit(rng, n::Int, d::Int)
    H = Matrix{Float64}(undef, n, d)
    for j in 1:d
        perm = randperm(rng, n)
        @inbounds for i in 1:n
            H[i, j] = (perm[i] - 1 + rand(rng)) / n
        end
    end
    return H
end

"""
    sample_candidate_from_unit(u) -> NamedTuple

Convert one LHS unit row `u` into a complete AEFBFP parameter set. The σ and ξ
sequences are fixed by the compressed-sensing protocol; μ and τ₀ vary.
"""
function sample_candidate_from_unit(u)
    mu    = _lin(u[1], AEFBFP_MU_RANGE[1], AEFBFP_MU_RANGE[2])
    tau_0 = _log(u[2], AEFBFP_TAU0_RANGE[1], AEFBFP_TAU0_RANGE[2])
    xi_exp = _lin(u[3], AEFBFP_XI_EXP_RANGE[1], AEFBFP_XI_EXP_RANGE[2])
    return (
        mu = mu,
        tau_0 = tau_0,
        xi_rule = :power,
        sigma_rule = :cs_inv_1000k_plus_1,
        xi_exp = xi_exp,
        sigma_exp = AEFBFP_SIGMA_EXP,
        sigma_scale = AEFBFP_SIGMA_SCALE,
    )
end

"Human-readable AEFBFP parameter string for logs; storage still keeps full precision."
pretty_params(params) = @sprintf("mu=%.3f tau0=%.3g xi_exp=%.2f sigma_exp=%.2f sigma_scale=%.3g",
                                 params.mu, params.tau_0, params.xi_exp,
                                 params.sigma_exp, params.sigma_scale)

# ============================================================================
# Candidate evaluation helpers
# ============================================================================

"""
    run_candidate_once(alg, prob, init, cfg)

Run one candidate on one fixed compressed-sensing problem and one initial
point. Returns everything the caller needs to both write the DB row and build
the candidate score summary.
"""
function run_candidate_once(alg, prob, init, cfg)
    stopping = make_stopping(prob, cfg.eps, cfg.maxiter; consec = cfg.consec)
    nrec = NativeResRecorder(prob.native_residual)

    try
        result = solve(alg, prob, copy(init.x0); stopping = stopping, observers = (nrec,))
        return (
            prob = prob,
            init = init,
            result = result,
            native_residual = nrec.value,
            error_msg = nothing,
        )
    catch err
        result = make_result(
            converged = false,
            iterations = 0,
            f_evals = 0,
            cpu_time = NaN,
            x = copy(init.x0),
            flag = :error,
        )
        return (
            prob = prob,
            init = init,
            result = result,
            native_residual = NaN,
            error_msg = sprint(showerror, err),
        )
    end
end

"""
    candidate_score(rows)

Input:
  `rows` = all run summaries produced by one sampled parameter configuration.
  Each row is expected to carry:
    - `converged`
    - `iterations`
    - `f_evals`
    - `cpu_time`

Output:
  A NamedTuple
    `(nconv, med_iter, med_eval, med_cpu)`
  where the medians are computed only from converged rows.

If no run converged, the function returns `Inf` for all median fields so this
configuration ranks below any configuration with at least one successful run.
"""
function candidate_score(rows)
    nconv = count(r -> r.converged, rows)
    finite_residuals = Float64[r.native_residual for r in rows if isfinite(r.native_residual)]
    med_residual = isempty(finite_residuals) ? Inf : median(finite_residuals)
    if nconv == 0
        return (nconv = 0, med_residual = med_residual, med_iter = Inf, med_eval = Inf, med_cpu = Inf)
    end

    conv_rows = filter(r -> r.converged, rows)
    return (
        nconv = nconv,
        med_residual = med_residual,
        med_iter = median(Float64[r.iterations for r in conv_rows]),
        med_eval = median(Float64[r.f_evals for r in conv_rows]),
        med_cpu = median(Float64[r.cpu_time for r in conv_rows]),
    )
end

"Convert one score into a sortable tuple. Better configurations get a smaller key:
 more converged runs first, then fewer iterations, fewer evaluations, and less CPU."
candidate_rank_key(score, hash::String) =
    (-score.nconv, score.med_residual, score.med_iter, score.med_eval, score.med_cpu, hash)

"JSON cannot represent `Inf` cleanly in the winner summary, so replace non-finite
 metrics with `nothing` before writing JSON."
json_safe_metric(x::Real) = isfinite(float(x)) ? float(x) : nothing

# ============================================================================
# DB promotion helper
# ============================================================================

"""
    promote_winner!(db, config_hash, summary)

Input:
  - `db`          : open SQLite handle for the compressed-sensing experiments DB
  - `config_hash` : hash of the winning AEFBFP parameter configuration
  - `summary`     : score tuple for that winner, typically containing
                    `nconv`, `med_iter`, `med_eval`, and `med_cpu`

Output:
  No returned value is used. The function's job is its side effect.

Side effect:
  Insert or replace one row in the local `tuned_winners` table so later scripts
  (especially `s30_benchmark.jl`) can load the promoted AEFBFP winner directly
  from the DB.
"""
function promote_winner!(db, config_hash::String, summary)
    ensure_local_tables!(db)
    DBInterface.execute(db, """
        INSERT OR REPLACE INTO tuned_winners (method, problem, config_hash, summary_json, created_at)
        VALUES (?, ?, ?, ?, ?)
    """, (
        "AEFBFP",
        PROBLEM_NAME,
        config_hash,
        JSON3.write(Dict(
            "nconv" => summary.nconv,
            "med_residual" => json_safe_metric(summary.med_residual),
            "med_iter" => json_safe_metric(summary.med_iter),
            "med_eval" => json_safe_metric(summary.med_eval),
            "med_cpu" => json_safe_metric(summary.med_cpu),
        )),
        Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"),
    ))
end

# ============================================================================
# Logging helpers
# ============================================================================

"Print the fixed search settings once at the top of the log."
function log_search_header(tee, cfg)
    println(tee, "="^78)
    println(tee, "  AEFBFP parameter search: $(PROBLEM_NAME)")
    println(tee, "="^78)
    println(tee, "  candidates  : $(cfg.candidates)")
    println(tee, "  cases       : $(join(cfg.cases, ", "))")
    println(tee, "  dataset     : $(cfg.datasets)  (fixed benchmark dataset)")
    println(tee, "  initial_points : $(cfg.initial_points)")
    @printf(tee, "  eps         : %.1e\n", cfg.eps)
    println(tee, "  maxiter     : $(cfg.maxiter)")
    @printf(tee, "  gamma       : %.1e\n", cfg.gamma)
    @printf(tee, "  snr_db      : %.1f dB\n", cfg.snr_db)
    println(tee, "  search      : LHS")
        println(tee, "  lhs_dims    : $(SEARCH_DIMS)  (mu, tau_0, xi_exp)")
    println(tee, "  mu_range    : [$(AEFBFP_MU_RANGE[1]), $(AEFBFP_MU_RANGE[2])]")
    println(tee, "  tau0_range  : [$(AEFBFP_TAU0_RANGE[1]), $(AEFBFP_TAU0_RANGE[2])] (log)")
        println(tee, "  xi_exp_range: [$(AEFBFP_XI_EXP_RANGE[1]), $(AEFBFP_XI_EXP_RANGE[2])]")
    println(tee, "  sigma_k     : 1/(1000k+1)")
    println(tee)
end

# ============================================================================
# Main
# ============================================================================

"""
    search_main(args = ARGS)

Input:
  - `args` : command-line arguments for the tuning script. In this simplified
             version the useful user-facing options are:
               `--candidates`
               `--cases`
               `--force`

Output:
  Returns `nothing`. The useful outputs of this function are not Julia return
  values; they are written to the DB and the log file.

Main side effects:
  - runs the AEFBFP parameter search
  - writes tuning rows into the compressed-sensing experiment DB
  - promotes the best configuration into `tuned_winners`
  - writes a log file under `results/compressed_sensing/logs`

What this function does, step by step:
  1. Read the user options and freeze them into one tuning config `cfg`
  2. Print the search settings once at the top of the log
  3. Build the LHS matrix `H`
       - rows of `H`    = sampled parameter configurations
       - columns of `H` = AEFBFP tuning dimensions
  4. For each row of `H`
       - convert the unit row into actual AEFBFP parameters
       - build an AEFBFP algorithm object from those parameters
       - run that configuration on each selected case
       - for each case, use the same fixed dataset and all fixed initial points
       - collect one summary row per run
  5. Score each sampled configuration from all of its collected rows
  6. Keep the best-scoring configuration seen so far
  7. After all rows are tested, write the winner into `tuned_winners`

Why the main helpers are used here:
  - `read_search_config`
      reads the CLI arguments and returns the final tuning settings:
      candidate count, selected cases, fixed stopping settings, and force mode
  - `log_search_header`
      prints exactly what search is about to run, so the log is self-explanatory
  - `lhs_unit`
      generates the unit LHS matrix `H`; each row is one sampled search point
      in `[0,1]^5`
  - `sample_candidate_from_unit`
      maps one unit row like `[u1,u2,u3,u4,u5]` to one real AEFBFP parameter
      tuple `(mu, tau_0, xi_exp, sigma_exp, sigma_scale, ...)`
  - `dataset_seed`
      picks the one fixed dataset used by both tuning and benchmark for a case
  - `build_problem`
      builds the compressed-sensing problem instance:
      matrix `C`, signal `x_star`, noise, measurements `y`, and initial points
  - `run_candidate_once`
      runs one AEFBFP configuration on one initial point and returns:
      solver result, native residual, and optional error text
  - `candidate_score`
      compresses all rows of one configuration into one summary score:
      `(nconv, med_iter, med_eval, med_cpu)`
  - `candidate_rank_key`
      converts the score into a sortable tuple so two configurations can be
      compared with plain tuple ordering
  - `promote_winner!`
      stores the final winning configuration in the DB so later scripts,
      especially `s30_benchmark.jl`, can load it directly
"""
function search_main(args = ARGS)
    cfg = read_search_config(args)
    logpath, tee, _ = setup_logging("s20_aefbfp_parameter_search"; logdir = LOGDIR)
    db = open_db(DB_PATH)
    ensure_local_tables!(db)

    try
        # Stage 1: print the search plan once, before any candidate runs.
        log_search_header(tee, cfg)

        # Stage 2: generate the full LHS design. Each row of H is one unit
        # vector u in [0,1]^5, and each row becomes one AEFBFP candidate.
        rng = MersenneTwister(SEARCH_LHS_SEED)
        H = lhs_unit(rng, cfg.candidates, SEARCH_DIMS)
        run_id = "s20_" * Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
        best = nothing

        # Stage 3: loop over candidates.
        for cand_idx in 1:cfg.candidates
            u = @view H[cand_idx, :]
            params = sample_candidate_from_unit(u)
            alg = AEFBFP(:P3; params...)
            hash, hash_input = make_config_hash(alg, PROBLEM_NAME, cfg.eps, cfg.maxiter)
            ensure_config!(db, alg, PROBLEM_NAME, cfg.eps, cfg.maxiter, hash, hash_input)

            rows = NamedTuple[]
            println(tee, "\n[candidate $(cand_idx)/$(cfg.candidates)] hash=$(hash)")
            println(tee, "  u      = $(collect(u))")
            println(tee, "  params = $(pretty_params(params))")

            # Stage 4: run this one candidate on the fixed dataset of every
            # selected case, across all deterministic initial points.
            for case_id in cfg.cases
                case = CASE_BY_NAME[case_id]
                data_seed = dataset_seed(case, SEARCH_DATASET)
                prob = build_problem(case; gamma = cfg.gamma, snr_db = cfg.snr_db,
                                     data_seed = data_seed, n_inits = cfg.initial_points)

                for init in prob.initial_points
                    run = run_candidate_once(alg, prob, init, cfg)

                    if run.error_msg !== nothing
                        println(tee, "  case=$(case_id) init=$(init.label) ERROR: $(run.error_msg)")
                    end

                    insert_result!(db, hash, case_id, run.prob.dim, run.init.label, run.init.seed_idx, run_id, run.result;
                                   script = "s20", native_residual = run.native_residual, production = false)

                    push!(rows, (
                        converged = run.result.converged,
                        iterations = run.result.iterations,
                        f_evals = run.result.f_evals,
                        cpu_time = run.result.cpu_time,
                        native_residual = run.native_residual,
                    ))

                    @printf(tee,
                            "  case=%-16s init=%-7s conv=%-5s  iter=%5d  fe=%5d  nat=%10.3e\n",
                            case_id, init.label, string(run.result.converged), run.result.iterations,
                            run.result.f_evals, run.native_residual)
                end
            end

            # Stage 5: score this candidate across all of its rows.
            score = candidate_score(rows)
            println(tee, "  score = $(score)")

            if best === nothing
                best = (hash = hash, params = params, score = score)
            elseif candidate_rank_key(score, hash) < candidate_rank_key(best.score, best.hash)
                best = (hash = hash, params = params, score = score)
            end
        end

        # Stage 6: freeze the winner into the local tuned_winners table.
        best === nothing && error("Search produced no candidate.")
        promote_winner!(db, best.hash, best.score)

        println(tee, "\n--- promoted winner ---")
        println(tee, "  hash   = $(best.hash)")
        println(tee, "  params = $(pretty_params(best.params))")
        println(tee, "  full   = $(best.params)")
        println(tee, "  score  = $(best.score)")
    finally
        teardown_logging(tee, logpath)
        SQLite.close(db)
    end

    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    search_main()
end
