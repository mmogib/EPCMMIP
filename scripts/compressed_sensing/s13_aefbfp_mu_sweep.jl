# ============================================================================
# s13_aefbfp_mu_sweep.jl
# ============================================================================
#
# Purpose
#   A read-only, one-at-a-time AEFBFP mu diagnostic for compressed sensing.
#   It holds every tuned AEFBFP parameter other than mu fixed and uses the B1
#   common unit-step LASSO residual for stopping.
#
# Usage
#   julia --project=. scripts/compressed_sensing/s13_aefbfp_mu_sweep.jl
#   julia --project=. scripts/compressed_sensing/s13_aefbfp_mu_sweep.jl --mus=0.10,0.20,0.30,0.40,0.45,0.49 --inits=10
#
# Output
#   results/compressed_sensing/diagnostics/aefbfp_mu_sweep_<case>.csv
#   results/compressed_sensing/diagnostics/aefbfp_mu_stepsize_<case>_seed<init>.pdf
#
# This is a diagnostic OAT sweep, not a final parameter-selection experiment.
# Any selected value must be confirmed on held-out instances before it is used
# in a paper-facing benchmark.
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

function parse_mu_values(text::String)
    values = Float64[]
    for item in split(text, ",")
        value = parse(Float64, strip(item))
        0 < value < 0.5 || throw(ArgumentError("Every --mus value must be in (0, 0.5)"))
        push!(values, value)
    end
    isempty(values) && throw(ArgumentError("--mus must contain at least one value"))
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
    mus = parse_mu_values(sweep_option(args, "mus", "0.10,0.20,0.30,0.40,0.45,0.49"))
    plot_init = parse(Int, sweep_option(args, "plot-init", "1"))
    1 <= inits <= DEFAULT_INITIAL_POINTS || throw(ArgumentError("--inits must be between 1 and $(DEFAULT_INITIAL_POINTS)"))
    1 <= plot_init <= inits || throw(ArgumentError("--plot-init must be between 1 and --inits"))
    maxiter >= 1 || throw(ArgumentError("--maxiter must be >= 1"))
    eps > 0 || throw(ArgumentError("--eps must be > 0"))
    return (case_id = case_id, inits = inits, maxiter = maxiter, eps = eps,
            gamma = gamma, snr_db = snr_db, mus = mus, plot_init = plot_init)
end

"Return a copy of `alg` with only the min-rule factor mu changed."
function replace_mu(alg::AEFBFP, mu::Float64)
    return AEFBFP(mu = mu, tau_0 = alg.tau_0, xi_rule = alg.xi_rule,
                  sigma_rule = alg.sigma_rule, xi_exp = alg.xi_exp,
                  sigma_exp = alg.sigma_exp, sigma_scale = alg.sigma_scale)
end

function write_sweep_csv(path::String, rows)
    open(path, "w") do io
        println(io, "mu,init,converged,iterations,f_evals,cpu_time,common_residual,flag")
        for row in rows
            @printf(io, "%.16e,%s,%s,%d,%d,%.16e,%.16e,%s\n",
                    row.mu, row.init, string(row.converged), row.iterations,
                    row.f_evals, row.cpu_time, row.common_residual, row.flag)
        end
    end
    return path
end

function print_sweep_summary(tee, rows, mus)
    println(tee, "\n  Summary over converged starts")
    @printf(tee, "  %8s  %9s  %12s  %12s  %12s\n",
            "mu", "success", "med iter", "med F-eval", "med residual")
    println(tee, "  " * "-" ^ 61)
    for mu in mus
        subset = [row for row in rows if row.mu == mu]
        solved = [row for row in subset if row.converged]
        if isempty(solved)
            @printf(tee, "  %8.3f  %4d/%-4d  %12s  %12s  %12s\n",
                    mu, 0, length(subset), "DNC", "DNC", "DNC")
        else
            @printf(tee, "  %8.3f  %4d/%-4d  %12.1f  %12.1f  %12.3e\n",
                    mu, length(solved), length(subset),
                    median(Float64[row.iterations for row in solved]),
                    median(Float64[row.f_evals for row in solved]),
                    median(Float64[row.common_residual for row in solved]))
        end
    end
end

"Plot the stepsize recorded after each iteration for every tested mu value."
function write_mu_stepsize_plot(path::String, trajectories, case_id::String, init_idx::Int)
    plt = plot(xlabel = "Iteration", ylabel = "tau_next", yscale = :log10,
               grid = true, legend = :best,
               title = "AEFBFP stepsize trajectories: $(case_id), seed $(init_idx)")
    for mu in sort(collect(keys(trajectories)))
        history = trajectories[mu]
        ks = [record.k for record in history]
        taus = [max(record.step_size, eps(Float64)) for record in history]
        plot!(plt, ks, taus; lw = 2.0, label = @sprintf("mu = %.2f", mu))
    end
    savefig(plt, path)
    return path
end

function mu_sweep_main(args = ARGS)
    cfg = sweep_config(args)
    mkpath(DIAGNOSTIC_DIR)
    logpath, tee, _ = setup_logging("s13_aefbfp_mu_sweep"; logdir = LOGDIR)
    db = open_db(DB_PATH)
    try
        case = CASE_BY_NAME[cfg.case_id]
        prob = build_problem(case; gamma = cfg.gamma, snr_db = cfg.snr_db,
                             data_seed = dataset_seed(case, 1), n_inits = cfg.inits)
        base_alg = build_algorithm(db, "AEFBFP")
        rows = NamedTuple[]
        trajectories = Dict{Float64,Vector{IterRecord}}()

        println(tee, "AEFBFP mu diagnostic sweep")
        println(tee, "  case          : $(cfg.case_id)")
        println(tee, "  starts        : $(cfg.inits)")
        println(tee, "  maxiter / eps : $(cfg.maxiter) / $(cfg.eps)")
        println(tee, "  mu values     : $(join(string.(cfg.mus), ", "))")
        println(tee, "  plot start    : seed$(cfg.plot_init)")
        println(tee, "  All parameters other than mu are fixed at the tuned values.\n")

        for mu in cfg.mus
            alg = replace_mu(base_alg, mu)
            for init in prob.initial_points
                stopping = make_stopping(prob, cfg.eps, cfg.maxiter; consec = 2)
                recorder = NativeResRecorder(prob.native_residual)
                result = if init.seed_idx == cfg.plot_init
                    history_callback = HistoryCallback()
                    trial_result = solve(alg, prob, copy(init.x0); stopping = stopping,
                                         observers = (recorder, history_callback))
                    trajectories[mu] = history_callback.history
                    trial_result
                else
                    solve(alg, prob, copy(init.x0); stopping = stopping,
                          observers = (recorder,))
                end
                row = (mu = mu, init = init.label, converged = result.converged,
                       iterations = result.iterations, f_evals = result.f_evals,
                       cpu_time = result.cpu_time, common_residual = recorder.value,
                       flag = string(result.flag))
                push!(rows, row)
                @printf(tee, "  mu=%7.3f  init=%-7s  conv=%-5s  iter=%5d  F=%5d  residual=%10.3e\n",
                        mu, init.label, string(result.converged), result.iterations,
                        result.f_evals, recorder.value)
            end
        end

        csv_path = write_sweep_csv(joinpath(DIAGNOSTIC_DIR,
                                             "aefbfp_mu_sweep_$(cfg.case_id).csv"), rows)
        plot_path = write_mu_stepsize_plot(joinpath(DIAGNOSTIC_DIR,
                                                     "aefbfp_mu_stepsize_$(cfg.case_id)_seed$(cfg.plot_init).pdf"),
                                            trajectories, cfg.case_id, cfg.plot_init)
        print_sweep_summary(tee, rows, cfg.mus)
        println(tee, "\n  CSV: $(csv_path)")
        println(tee, "  stepsize plot: $(plot_path)")
    finally
        teardown_logging(tee, logpath)
        SQLite.close(db)
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    mu_sweep_main()
end
