# ============================================================================
# s18_irfbsm_theta_sweep.jl
# ============================================================================
#
# Purpose
#   Read-only sensitivity check for the IRFBSM relaxation parameter theta on
#   the compressed-sensing benchmark.  It uses the common L^{-1}-scaled LASSO
#   residual for stopping, exactly as s30_benchmark.jl does.
#
# Usage
#   julia --project=. scripts/compressed_sensing/s18_irfbsm_theta_sweep.jl
#   julia --project=. scripts/compressed_sensing/s18_irfbsm_theta_sweep.jl --thetas=0.6,0.8,1.0 --inits=10
#   julia --project=. scripts/compressed_sensing/s18_irfbsm_theta_sweep.jl --cases=M512_N1024_k50 --maxiter=5000
#
# Output
#   results/compressed_sensing/diagnostics/irfbsm_theta_sweep_<timestamp>.csv
#   results/compressed_sensing/logs/log_s18_irfbsm_theta_sweep_<timestamp>.txt
#
# This is a sensitivity diagnostic only.  The production benchmark uses the
# separately declared :cs_benchmark preset; :paper preserves source provenance.
# ============================================================================

include(joinpath(@__DIR__, "s30_benchmark.jl"))

const THETA_DIAGNOSTIC_DIR = joinpath(RESULT_ROOT, "diagnostics")

function theta_option(args, name::String, default::String)
    prefix = "--" * name * "="
    for arg in args
        startswith(arg, prefix) && return arg[length(prefix)+1:end]
    end
    return default
end

function parse_theta_values(text::String)
    values = Float64[]
    for item in split(text, ",")
        theta = parse(Float64, strip(item))
        0 < theta <= 1 || throw(ArgumentError("Each --thetas value must lie in (0, 1]"))
        push!(values, theta)
    end
    isempty(values) && throw(ArgumentError("--thetas must contain at least one value"))
    return unique(sort(values))
end

function theta_sweep_config(args = ARGS)
    default_cases = join([case.problem for case in DEFAULT_CASES], ",")
    case_ids = split(theta_option(args, "cases", default_cases), ",")
    all(haskey(CASE_BY_NAME, id) for id in case_ids) || throw(ArgumentError("Unknown case in --cases"))
    inits = parse(Int, theta_option(args, "inits", "10"))
    maxiter = parse(Int, theta_option(args, "maxiter", string(NMAX_REF)))
    eps_value = parse(Float64, theta_option(args, "eps", string(EPS_REF)))
    thetas = parse_theta_values(theta_option(args, "thetas", "0.6,0.8,1.0"))
    1 <= inits <= DEFAULT_INITIAL_POINTS || throw(ArgumentError("--inits must be in 1:$(DEFAULT_INITIAL_POINTS)"))
    maxiter >= 1 || throw(ArgumentError("--maxiter must be >= 1"))
    eps_value > 0 || throw(ArgumentError("--eps must be > 0"))
    return (case_ids = case_ids, inits = inits, maxiter = maxiter,
            eps = eps_value, thetas = thetas)
end

function write_theta_csv(path::String, rows)
    open(path, "w") do io
        println(io, "case,theta,init,converged,iterations,f_evals,cpu_time,common_residual,flag")
        for row in rows
            @printf(io, "%s,%.16e,%s,%s,%d,%d,%.16e,%.16e,%s\n",
                    row.case_id, row.theta, row.init, string(row.converged),
                    row.iterations, row.f_evals, row.cpu_time,
                    row.common_residual, row.flag)
        end
    end
    return path
end

function print_theta_summary(tee, rows, case_id::AbstractString, thetas)
    println(tee, "\n  Summary: $(case_id)")
    @printf(tee, "  %7s  %9s  %10s  %11s  %12s  %12s\n",
            "theta", "success", "med iter", "med F-eval", "med CPU", "med residual")
    println(tee, "  " * "-" ^ 75)
    for theta in thetas
        subset = [row for row in rows if row.case_id == case_id && row.theta == theta]
        solved = [row for row in subset if row.converged]
        if isempty(solved)
            @printf(tee, "  %7.2f  %4d/%-4d  %10s  %11s  %12s  %12s\n",
                    theta, 0, length(subset), "DNC", "DNC", "DNC", "DNC")
        else
            @printf(tee, "  %7.2f  %4d/%-4d  %10.1f  %11.1f  %12.4f  %12.3e\n",
                    theta, length(solved), length(subset),
                    median(Float64[r.iterations for r in solved]),
                    median(Float64[r.f_evals for r in solved]),
                    median(Float64[r.cpu_time for r in solved]),
                    median(Float64[r.common_residual for r in solved]))
        end
    end
end

function theta_sweep_main(args = ARGS)
    cfg = theta_sweep_config(args)
    mkpath(THETA_DIAGNOSTIC_DIR)
    logpath, tee, _ = setup_logging("s18_irfbsm_theta_sweep"; logdir = LOGDIR)
    rows = NamedTuple[]
    try
        println(tee, "IRFBSM relaxation sensitivity sweep")
        println(tee, "  theta values  : $(join(cfg.thetas, ", "))")
        println(tee, "  fixed values  : lambda_0=1.0, mu=0.9, alpha=0.04")
        println(tee, "  cases         : $(join(cfg.case_ids, ", "))")
        println(tee, "  starts        : $(cfg.inits)")
        println(tee, "  maxiter / eps : $(cfg.maxiter) / $(cfg.eps)")
        println(tee, "  stopping      : common L^{-1}-scaled LASSO residual\n")

        for case_id in cfg.case_ids
            case = CASE_BY_NAME[case_id]
            prob = build_problem(case; gamma = GAMMA_REF, snr_db = SNR_DB_REF,
                                 data_seed = dataset_seed(case, 1), n_inits = cfg.inits)
            for theta in cfg.thetas
                alg = IRFBSM(:paper; theta = theta)
                for init in prob.initial_points
                    stopping = make_stopping(prob, cfg.eps, cfg.maxiter; consec = 2)
                    result = solve(alg, prob, copy(init.x0); stopping = stopping)
                    common_residual = cs_scaled_optimality_residual(result.x, prob)
                    push!(rows, (case_id = case_id, theta = theta, init = init.label,
                                 converged = result.converged, iterations = result.iterations,
                                 f_evals = result.f_evals, cpu_time = result.cpu_time,
                                 common_residual = common_residual, flag = String(result.flag)))
                end
            end
            print_theta_summary(tee, rows, case_id, cfg.thetas)
        end

        stamp = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
        csvpath = joinpath(THETA_DIAGNOSTIC_DIR, "irfbsm_theta_sweep_$(stamp).csv")
        write_theta_csv(csvpath, rows)
        println(tee, "\n  CSV saved: $(csvpath)")
    finally
        teardown_logging(tee, logpath)
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    theta_sweep_main()
end
