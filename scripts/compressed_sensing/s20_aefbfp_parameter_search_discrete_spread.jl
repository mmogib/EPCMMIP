# ============================================================================
# s20_aefbfp_parameter_search_discrete_spread.jl
# ============================================================================
#
# Purpose
#   Run a separate AEFBFP tuning search for compressed sensing without touching
#   the original `s20_aefbfp_parameter_search.jl`.
#
# Search design
#   - candidate space: fully discrete
#   - default candidates evaluated: 120
#   - selection rule: spread-out greedy maximin sample from the full discrete
#     Cartesian product, so sampled candidates do not cluster
#   - cases / stopping / DB writing: same workflow as the original s20 script
#
# Discrete grids
#   - mu                            : coarse decimal grid
#   - tau_0                         : scientific/log-style grid
#   - ξ_k = 1/(k+100)^2 and σ_k = 1/(1000k+1) are fixed
#
# How to run
#   julia --project=. compressed_sensing/s20_aefbfp_parameter_search_discrete_spread.jl
#   julia --project=. compressed_sensing/s20_aefbfp_parameter_search_discrete_spread.jl --candidates=160
#   julia --project=. compressed_sensing/s20_aefbfp_parameter_search_discrete_spread.jl --force
#
# ============================================================================

include(joinpath(@__DIR__, "s20_aefbfp_parameter_search.jl"))

const DISCRETE_SEARCH_CASES = copy(SEARCH_CASES)
const DISCRETE_SEARCH_CANDIDATES = 120

const MU_LEVELS = Float64[AEFBFP_MU]
const TAU0_LEVELS = Float64[1.0e-4, 3.0e-4, 1.0e-3, 3.0e-3, 1.0e-2, 3.0e-2, 1.0e-1]
const DISCRETE_TOTAL_COMBINATIONS =
    length(MU_LEVELS) *
    length(TAU0_LEVELS)

function read_discrete_search_config(args)
    opts, flags = parse_cli(args)

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

    candidates = haskey(opts, "candidates") ? parse(Int, opts["candidates"]) : DISCRETE_SEARCH_CANDIDATES
    cases = haskey(opts, "cases") ? parse_case_list(opts["cases"]) : DISCRETE_SEARCH_CASES

    candidates >= 1 || throw(ArgumentError("candidates must be >= 1"))
    candidates <= DISCRETE_TOTAL_COMBINATIONS ||
        throw(ArgumentError("candidates must be <= $(DISCRETE_TOTAL_COMBINATIONS), got $candidates"))

    eps = EPS_REF
    maxiter = canonical_maxiter(eps)

    return (
        candidates = candidates,
        cases = cases,
        datasets = 1,
        initial_points = DEFAULT_INITIAL_POINTS,
        eps = eps,
        maxiter = maxiter,
        consec = 2,
        gamma = GAMMA_REF,
        snr_db = SNR_DB_REF,
        force = "force" in flags,
    )
end

function all_discrete_candidates()
    params = NamedTuple[]
    sizehint!(params, DISCRETE_TOTAL_COMBINATIONS)
    for mu in MU_LEVELS
        for tau_0 in TAU0_LEVELS
            push!(params, (
                mu = mu,
                tau_0 = tau_0,
                xi_rule = :cs_inv_k_plus_100_sq,
                sigma_rule = :cs_inv_1000k_plus_1,
                xi_exp = AEFBFP_XI_EXP,
                sigma_exp = AEFBFP_SIGMA_EXP,
                sigma_scale = AEFBFP_SIGMA_SCALE,
            ))
        end
    end
    return params
end

@inline normalize_linear(x, levels) = (x - first(levels)) / (last(levels) - first(levels))
@inline normalize_log(x, levels) = (log10(x) - log10(first(levels))) / (log10(last(levels)) - log10(first(levels)))

function candidate_features(params)
    feats = Matrix{Float64}(undef, length(params), 2)
    @inbounds for i in eachindex(params)
        p = params[i]
        feats[i, 1] = normalize_linear(p.mu, MU_LEVELS)
        feats[i, 2] = normalize_log(p.tau_0, TAU0_LEVELS)
    end
    return feats
end

@inline function row_distance_sq(feats::Matrix{Float64}, i::Int, j::Int)
    d2 = 0.0
    @inbounds for k in 1:size(feats, 2)
        d = feats[i, k] - feats[j, k]
        d2 += d * d
    end
    return d2
end

@inline function row_to_center_distance_sq(feats::Matrix{Float64}, i::Int)
    d2 = 0.0
    @inbounds for k in 1:size(feats, 2)
        d = feats[i, k] - 0.5
        d2 += d * d
    end
    return d2
end

function select_spread_candidate_indices(params, nselect::Int)
    nall = length(params)
    nselect <= nall || throw(ArgumentError("nselect must be <= number of candidates"))
    feats = candidate_features(params)

    first_idx = 1
    first_d2 = row_to_center_distance_sq(feats, 1)
    for i in 2:nall
        d2 = row_to_center_distance_sq(feats, i)
        if d2 < first_d2
            first_idx = i
            first_d2 = d2
        end
    end

    selected = Int[first_idx]
    min_d2 = fill(Inf, nall)

    for i in 1:nall
        min_d2[i] = row_distance_sq(feats, i, first_idx)
    end
    min_d2[first_idx] = -Inf

    while length(selected) < nselect
        next_idx = argmax(min_d2)
        push!(selected, next_idx)
        for i in 1:nall
            min_d2[i] == -Inf && continue
            d2 = row_distance_sq(feats, i, next_idx)
            d2 < min_d2[i] && (min_d2[i] = d2)
        end
        min_d2[next_idx] = -Inf
    end

    return selected
end

function log_discrete_search_header(tee, cfg)
    println(tee, "="^78)
    println(tee, "  AEFBFP discrete spread search: $(PROBLEM_NAME)")
    println(tee, "="^78)
    println(tee, "  candidates     : $(cfg.candidates)")
    println(tee, "  total combos   : $(DISCRETE_TOTAL_COMBINATIONS)")
    println(tee, "  selector       : greedy maximin spread sample (no clustering)")
    println(tee, "  cases          : $(join(cfg.cases, ", "))")
    println(tee, "  dataset        : $(cfg.datasets)  (fixed benchmark dataset)")
    println(tee, "  initial_points : $(cfg.initial_points)")
    @printf(tee, "  eps            : %.1e\n", cfg.eps)
    println(tee, "  maxiter        : $(cfg.maxiter)")
    @printf(tee, "  gamma          : %.1e\n", cfg.gamma)
    @printf(tee, "  snr_db         : %.1f dB\n", cfg.snr_db)
    println(tee, "  mu_levels         : $(MU_LEVELS)")
    println(tee, "  tau0_levels       : $(TAU0_LEVELS)")
    println(tee, "  xi_k              : 1/(k+100)^2")
    println(tee, "  sigma_k           : 1/(1000k+1)")
    println(tee)
end

function discrete_search_main(args = ARGS)
    cfg = read_discrete_search_config(args)
    logpath, tee, _ = setup_logging("s20_aefbfp_parameter_search_discrete_spread"; logdir = LOGDIR)
    db = open_db(DB_PATH)
    ensure_local_tables!(db)

    try
        log_discrete_search_header(tee, cfg)

        all_params = all_discrete_candidates()
        chosen = select_spread_candidate_indices(all_params, cfg.candidates)
        run_id = "s20d_" * Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
        best = nothing

        for (cand_pos, idx) in enumerate(chosen)
            params = all_params[idx]
            alg = AEFBFP(:P3; params...)
            hash, hash_input = make_config_hash(alg, PROBLEM_NAME, cfg.eps, cfg.maxiter)
            ensure_config!(db, alg, PROBLEM_NAME, cfg.eps, cfg.maxiter, hash, hash_input)

            rows = NamedTuple[]
            println(tee, "\n[candidate $(cand_pos)/$(cfg.candidates)] combo_idx=$(idx) hash=$(hash)")
            println(tee, "  params = $(pretty_params(params))")

            for case_id in cfg.cases
                case = CASE_BY_NAME[case_id]
                data_seed = dataset_seed(case, 1)
                prob = build_problem(case; gamma = cfg.gamma, snr_db = cfg.snr_db,
                                     data_seed = data_seed, n_inits = cfg.initial_points)

                for init in prob.initial_points
                    run = run_candidate_once(alg, prob, init, cfg)

                    if run.error_msg !== nothing
                        println(tee, "  case=$(case_id) init=$(init.label) ERROR: $(run.error_msg)")
                    end

                    insert_result!(db, hash, case_id, run.prob.dim, run.init.label, run.init.seed_idx, run_id, run.result;
                                   script = "s20_discrete_spread", native_residual = run.native_residual, production = false)

                    push!(rows, (
                        converged = run.result.converged,
                        iterations = run.result.iterations,
                        f_evals = run.result.f_evals,
                        cpu_time = run.result.cpu_time,
                    ))

                    @printf(tee,
                            "  case=%-16s init=%-7s conv=%-5s  iter=%5d  fe=%5d  nat=%10.3e\n",
                            case_id, init.label, string(run.result.converged), run.result.iterations,
                            run.result.f_evals, run.native_residual)
                end
            end

            score = candidate_score(rows)
            println(tee, "  score = $(score)")

            if best === nothing
                best = (hash = hash, params = params, score = score)
            elseif candidate_rank_key(score, hash) < candidate_rank_key(best.score, best.hash)
                best = (hash = hash, params = params, score = score)
            end
        end

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
    discrete_search_main()
end
