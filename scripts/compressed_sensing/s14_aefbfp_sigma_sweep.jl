# ============================================================================
# s14_aefbfp_sigma_sweep.jl
# ============================================================================
# Read-only anchoring sensitivity for AEFBFP compressed sensing.
# Use --grid=reviewer for c/(k+2)^0.97 with c = 0.0024, 0.024, 0.24, or
# --grid=exponents for the requested power and normalized-log exponent grids.
# ============================================================================

include(joinpath(@__DIR__, "s30_benchmark.jl"))

const DIAGNOSTIC_DIR = joinpath(RESULT_ROOT, "diagnostics")

function sweep_option(args, name::String, default::String)
    prefix = "--" * name * "="
    for arg in args
        startswith(arg, prefix) && return arg[length(prefix)+1:end]
    end
    return default
end

function sigma_sweep_config(args = ARGS)
    case_id = sweep_option(args, "case", "M256_N512_k30")
    haskey(CASE_BY_NAME, case_id) || throw(ArgumentError("Unknown --case=$case_id"))
    inits = parse(Int, sweep_option(args, "inits", "10"))
    maxiter = parse(Int, sweep_option(args, "maxiter", string(NMAX_REF)))
    eps = parse(Float64, sweep_option(args, "eps", string(EPS_REF)))
    gamma = parse(Float64, sweep_option(args, "gamma", string(GAMMA_REF)))
    snr_db = parse(Float64, sweep_option(args, "snr-db", string(SNR_DB_REF)))
    grid = sweep_option(args, "grid", "reviewer")
    grid in ("reviewer", "exponents") || throw(ArgumentError("--grid must be reviewer or exponents"))
    1 <= inits <= DEFAULT_INITIAL_POINTS || throw(ArgumentError("--inits must be between 1 and $(DEFAULT_INITIAL_POINTS)"))
    return (case_id = case_id, inits = inits, maxiter = maxiter, eps = eps,
            gamma = gamma, snr_db = snr_db, grid = grid)
end

function replace_sigma(alg::AEFBFP, rule::Symbol, scale::Float64, exponent::Float64)
    return AEFBFP(mu = alg.mu, tau_0 = alg.tau_0, xi_rule = alg.xi_rule,
                  sigma_rule = rule, xi_exp = alg.xi_exp,
                  sigma_exp = exponent, sigma_scale = scale)
end

function write_sigma_csv(path::String, rows)
    open(path, "w") do io
        println(io, "setting,sigma_rule,sigma_scale,sigma_exp,init,converged,iterations,f_evals,common_residual,flag")
        for r in rows
            @printf(io, "%s,%s,%.16e,%.16e,%s,%s,%d,%d,%.16e,%s\n",
                    r.setting, r.sigma_rule, r.sigma_scale, r.sigma_exp, r.init,
                    string(r.converged), r.iterations, r.f_evals, r.common_residual, r.flag)
        end
    end
end

function sigma_sweep_main(args = ARGS)
    cfg = sigma_sweep_config(args)
    mkpath(DIAGNOSTIC_DIR)
    logpath, tee, _ = setup_logging("s14_aefbfp_sigma_sweep"; logdir = LOGDIR)
    db = open_db(DB_PATH)
    try
        case = CASE_BY_NAME[cfg.case_id]
        prob = build_problem(case; gamma = cfg.gamma, snr_db = cfg.snr_db,
                             data_seed = dataset_seed(case, 1), n_inits = cfg.inits)
        base = build_algorithm(db, "AEFBFP")
        reviewer_settings = [
            (label = "current", rule = base.sigma_rule, scale = base.sigma_scale, exponent = base.sigma_exp),
            (label = "power_c0.0024", rule = :power, scale = 0.0024, exponent = 0.97),
            (label = "power_c0.024", rule = :power, scale = 0.024, exponent = 0.97),
            (label = "power_c0.24", rule = :power, scale = 0.24, exponent = 0.97),
        ]
        power_exponents = [0.20, 0.40, 0.60, 0.80, 0.99]
        log_exponents = [0.5, 1.0, 3.0, 5.0, 10.0]
        exponent_settings = vcat(
            [(label = @sprintf("power_p%.2f", p), rule = :power, scale = 1.0, exponent = p)
             for p in power_exponents],
            [(label = @sprintf("log_p%.1f", p), rule = :log_power, scale = 0.9, exponent = p)
             for p in log_exponents],
        )
        settings = cfg.grid == "reviewer" ? reviewer_settings : exponent_settings
        rows = NamedTuple[]

        println(tee, "AEFBFP sigma diagnostic sweep")
        println(tee, "  grid           : $(cfg.grid)")
        println(tee, "  case / starts  : $(cfg.case_id) / $(cfg.inits)")
        println(tee, "  maxiter / eps  : $(cfg.maxiter) / $(cfg.eps)")
        println(tee, "  Other AEFBFP parameters are held fixed.\n")

        for setting in settings
            alg = replace_sigma(base, setting.rule, setting.scale, setting.exponent)
            for init in prob.initial_points
                stopping = make_stopping(prob, cfg.eps, cfg.maxiter; consec = 2)
                recorder = NativeResRecorder(prob.native_residual)
                result = solve(alg, prob, copy(init.x0); stopping = stopping, observers = (recorder,))
                push!(rows, (setting = setting.label, sigma_rule = string(setting.rule),
                             sigma_scale = setting.scale, sigma_exp = setting.exponent,
                             init = init.label, converged = result.converged,
                             iterations = result.iterations, f_evals = result.f_evals,
                             common_residual = recorder.value, flag = string(result.flag)))
                @printf(tee, "  %-15s init=%-7s conv=%-5s iter=%5d residual=%10.3e\n",
                        setting.label, init.label, string(result.converged), result.iterations, recorder.value)
            end
        end

        println(tee, "\n  Summary over converged starts")
        @printf(tee, "  %-15s %9s %12s %12s\n", "setting", "success", "med iter", "med residual")
        println(tee, "  " * "-" ^ 55)
        for setting in settings
            subset = [r for r in rows if r.setting == setting.label]
            solved = [r for r in subset if r.converged]
            if isempty(solved)
                @printf(tee, "  %-15s %4d/%-4d %12s %12s\n", setting.label, 0, length(subset), "DNC", "DNC")
            else
                @printf(tee, "  %-15s %4d/%-4d %12.1f %12.3e\n", setting.label,
                        length(solved), length(subset), median(Float64[r.iterations for r in solved]),
                        median(Float64[r.common_residual for r in solved]))
            end
        end
        path = joinpath(DIAGNOSTIC_DIR, "aefbfp_sigma_sweep_$(cfg.grid)_$(cfg.case_id).csv")
        write_sigma_csv(path, rows)
        println(tee, "\n  CSV: $(path)")
    finally
        teardown_logging(tee, logpath)
        SQLite.close(db)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    sigma_sweep_main()
end
