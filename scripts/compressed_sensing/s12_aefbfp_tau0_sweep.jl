# ============================================================================
# s12_aefbfp_tau0_sweep.jl
# ============================================================================
#
# Purpose
#   A read-only, one-at-a-time AEFBFP initial-stepsize diagnostic for
#   compressed sensing.  It holds all tuned AEFBFP parameters except tau_0
#   fixed and uses the B1 common unit-step LASSO residual for stopping.
#
# Usage
#   julia --project=. scripts/compressed_sensing/s12_aefbfp_tau0_sweep.jl
#   julia --project=. scripts/compressed_sensing/s12_aefbfp_tau0_sweep.jl --tau0s=1e-4,3e-4,1e-3,3e-3 --inits=10
#
# Output
#   results/compressed_sensing/diagnostics/aefbfp_tau0_sweep_<case>.csv
#
# This script does not write to the benchmark database and must not be used to
# silently replace the predeclared benchmark setting.  It is a diagnostic OAT
# sweep; any selected setting requires a separate held-out confirmation run.
# ============================================================================

include(joinpath(@__DIR__, "s30_benchmark.jl"))

const DIAGNOSTIC_DIR = joinpath(RESULT_ROOT, "diagnostics")

function sweep_option(args, name::String, default::String)
    prefix = "--" * name * "="
    for arg in args
        if startswith(arg, prefix)
            return arg[length(prefix)+1:end]
        end
    end
    return default
end

function parse_tau0_values(text::String)
    values = Float64[]
    for item in split(text, ",")
        value = parse(Float64, strip(item))
        value > 0 || throw(ArgumentError("Every --tau0s value must be > 0"))
        push!(values, value)
    end
    isempty(values) && throw(ArgumentError("--tau0s must contain at least one value"))
    return values
end

function sweep_config(args = ARGS)
    case_id = sweep_option(args, "case", "M256_N512_k30")
    haskey(CASE_BY_NAME, case_id) || throw(ArgumentError("Unknown --case=$case_id"))
    inits = parse(Int, sweep_option(args, "inits", "10"))
    maxiter = parse(Int, sweep_option(args, "maxiter", string(NMAX_REF)))
    eps = parse(Float64, sweep_option(args, "eps", string(EPS_REF)))
    gamma = parse(Float64, sweep_option(args, "gamma", string(GAMMA_REF)))
    snr_db = parse(Float64, sweep_option(args, "snr-db", string(SNR_DB_REF)))
    tau0s = parse_tau0_values(sweep_option(args, "tau0s", "1e-4,3e-4,1e-3,3e-3"))
    1 <= inits <= DEFAULT_INITIAL_POINTS || throw(ArgumentError("--inits must be between 1 and $(DEFAULT_INITIAL_POINTS)"))
    maxiter >= 1 || throw(ArgumentError("--maxiter must be >= 1"))
    eps > 0 || throw(ArgumentError("--eps must be > 0"))
    return (case_id = case_id, inits = inits, maxiter = maxiter, eps = eps,
            gamma = gamma, snr_db = snr_db, tau0s = tau0s)
end

"Return a copy of `alg` with only its initial stepsize changed."
function replace_tau0(alg::AEFBFP, tau0::Float64)
    return AEFBFP(mu = alg.mu, tau_0 = tau0, xi_rule = alg.xi_rule,
                  sigma_rule = alg.sigma_rule, xi_exp = alg.xi_exp,
                  sigma_exp = alg.sigma_exp, sigma_scale = alg.sigma_scale)
end

function write_sweep_csv(path::String, rows)
    open(path, "w") do io
        println(io, "tau0,init,converged,iterations,f_evals,cpu_time,common_residual,flag")
        for row in rows
            @printf(io, "%.16e,%s,%s,%d,%d,%.16e,%.16e,%s\n",
                    row.tau0, row.init, string(row.converged), row.iterations,
                    row.f_evals, row.cpu_time, row.common_residual, row.flag)
        end
    end
    return path
end

function print_sweep_summary(tee, rows, tau0s)
    println(tee, "\n  Summary over converged starts")
    @printf(tee, "  %12s  %9s  %12s  %12s  %12s\n",
            "tau_0", "success", "med iter", "med F-eval", "med residual")
    println(tee, "  " * "-" ^ 65)
    for tau0 in tau0s
        subset = [row for row in rows if row.tau0 == tau0]
        solved = [row for row in subset if row.converged]
        if isempty(solved)
            @printf(tee, "  %12.4e  %4d/%-4d  %12s  %12s  %12s\n",
                    tau0, 0, length(subset), "DNC", "DNC", "DNC")
        else
            @printf(tee, "  %12.4e  %4d/%-4d  %12.1f  %12.1f  %12.3e\n",
                    tau0, length(solved), length(subset),
                    median(Float64[row.iterations for row in solved]),
                    median(Float64[row.f_evals for row in solved]),
                    median(Float64[row.common_residual for row in solved]))
        end
    end
end

function tau0_sweep_main(args = ARGS)
    cfg = sweep_config(args)
    mkpath(DIAGNOSTIC_DIR)
    logpath, tee, _ = setup_logging("s12_aefbfp_tau0_sweep"; logdir = LOGDIR)
    db = open_db(DB_PATH)
    try
        case = CASE_BY_NAME[cfg.case_id]
        prob = build_problem(case; gamma = cfg.gamma, snr_db = cfg.snr_db,
                             data_seed = dataset_seed(case, 1), n_inits = cfg.inits)
        base_alg = build_algorithm(db, "AEFBFP")
        rows = NamedTuple[]

        println(tee, "AEFBFP tau_0 diagnostic sweep")
        println(tee, "  case          : $(cfg.case_id)")
        println(tee, "  starts        : $(cfg.inits)")
        println(tee, "  maxiter / eps : $(cfg.maxiter) / $(cfg.eps)")
        println(tee, "  tau_0 values  : $(join(string.(cfg.tau0s), ", "))")
        println(tee, "  All parameters other than tau_0 are fixed at the tuned values.\n")

        for tau0 in cfg.tau0s
            alg = replace_tau0(base_alg, tau0)
            for init in prob.initial_points
                stopping = make_stopping(prob, cfg.eps, cfg.maxiter; consec = 2)
                recorder = NativeResRecorder(prob.native_residual)
                result = solve(alg, prob, copy(init.x0); stopping = stopping,
                               observers = (recorder,))
                row = (tau0 = tau0, init = init.label, converged = result.converged,
                       iterations = result.iterations, f_evals = result.f_evals,
                       cpu_time = result.cpu_time, common_residual = recorder.value,
                       flag = string(result.flag))
                push!(rows, row)
                @printf(tee, "  tau_0=%10.4e  init=%-7s  conv=%-5s  iter=%5d  F=%5d  residual=%10.3e\n",
                        tau0, init.label, string(result.converged), result.iterations,
                        result.f_evals, recorder.value)
            end
        end

        csv_path = write_sweep_csv(joinpath(DIAGNOSTIC_DIR,
                                             "aefbfp_tau0_sweep_$(cfg.case_id).csv"), rows)
        print_sweep_summary(tee, rows, cfg.tau0s)
        println(tee, "\n  CSV: $(csv_path)")
    finally
        teardown_logging(tee, logpath)
        SQLite.close(db)
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    tau0_sweep_main()
end
