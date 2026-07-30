# ============================================================================
# s11_aefbfp_stepsize_diagnostics.jl
# ============================================================================
#
# Purpose
#   Diagnose AEFBFP's adaptive stepsize on one reproducible compressed-sensing
#   run.  This is a read-only diagnostic: it neither alters benchmark rows nor
#   selects a parameter setting.  It writes the complete tau trajectory and a
#   summary of which branch of the min-rule was active.
#
# Usage
#   julia --project=. scripts/compressed_sensing/s11_aefbfp_stepsize_diagnostics.jl
#   julia --project=. scripts/compressed_sensing/s11_aefbfp_stepsize_diagnostics.jl --case=M256_N512_k30 --init=1 --maxiter=5000 --every=100
#
# Outputs
#   results/compressed_sensing/diagnostics/aefbfp_tau_<case>_seed<init>.csv
#   results/compressed_sensing/diagnostics/aefbfp_tau_<case>_seed<init>.pdf
#
# The iteration below mirrors solve(::AEFBFP, ...) in src/algorithm.jl.  Keep
# it synchronized with that solver if the method is changed.
# ============================================================================

include(joinpath(@__DIR__, "s30_benchmark.jl"))

const DIAGNOSTIC_DIR = joinpath(RESULT_ROOT, "diagnostics")

function diagnostic_option(args, name::String, default::String)
    prefix = "--" * name * "="
    for arg in args
        startswith(arg, prefix) && return arg[length(prefix)+1:end]
    end
    return default
end

function diagnostic_config(args = ARGS)
    case_id = diagnostic_option(args, "case", "M256_N512_k30")
    haskey(CASE_BY_NAME, case_id) || throw(ArgumentError("Unknown --case=$case_id"))
    init_idx = parse(Int, diagnostic_option(args, "init", "1"))
    maxiter = parse(Int, diagnostic_option(args, "maxiter", string(NMAX_REF)))
    every = parse(Int, diagnostic_option(args, "every", "1"))
    eps = parse(Float64, diagnostic_option(args, "eps", string(EPS_REF)))
    gamma = parse(Float64, diagnostic_option(args, "gamma", string(GAMMA_REF)))
    snr_db = parse(Float64, diagnostic_option(args, "snr-db", string(SNR_DB_REF)))
    init_idx >= 1 || throw(ArgumentError("--init must be >= 1"))
    maxiter >= 1 || throw(ArgumentError("--maxiter must be >= 1"))
    every >= 1 || throw(ArgumentError("--every must be >= 1"))
    eps > 0 || throw(ArgumentError("--eps must be > 0"))
    return (case_id = case_id, init_idx = init_idx, maxiter = maxiter, every = every,
            eps = eps, gamma = gamma, snr_db = snr_db)
end

"Run the AEFBFP update while recording both candidates of its min-rule."
function run_stepsize_diagnostic(alg::AEFBFP, prob::TestProblem, x0::Vector{Float64};
                                 maxiter::Int, eps::Float64, consec::Int = 2)
    x_anchor = copy(x0)
    x_curr = copy(x0)
    z_prev = copy(x0)                    # z_{-1} = x_0
    Bz_prev = prob.B(z_prev)             # initialization F-evaluation
    tau_curr = alg.tau_0
    f_evals = 1
    below_count = 0
    records = NamedTuple[]
    converged = false

    for k in 0:maxiter-1
        sigma_k = _epcm_sigma(alg.sigma_rule, alg.sigma_scale, alg.sigma_exp, k)
        w_k = sigma_k .* x_anchor .+ (1 - sigma_k) .* x_curr
        z_k = prob.resolvent_A(w_k .- tau_curr .* Bz_prev, tau_curr)
        Bz_k = prob.B(z_k)
        f_evals += 1
        x_next = z_k .+ tau_curr .* (Bz_prev .- Bz_k)

        delta_z = z_prev .- z_k
        delta_B = Bz_prev .- Bz_k
        norm_delta_B = norm(delta_B)
        xi_k = _epcm_delta(alg.xi_rule, alg.xi_exp, k)
        increment_candidate = tau_curr + xi_k
        ratio_candidate = norm_delta_B > 0 ?
            alg.mu * norm(delta_z) / norm_delta_B : Inf
        ratio_branch_active = ratio_candidate <= increment_candidate
        tau_next = min(ratio_candidate, increment_candidate)
        branch = ratio_branch_active ? "ratio" : "increment"

        residual = prob.native_residual(x_next, x_curr)
        if isfinite(residual) && residual < eps
            below_count += 1
            converged = below_count >= consec
        else
            below_count = 0
        end

        push!(records, (
            k = k + 1,
            f_evals = f_evals,
            tau_k = tau_curr,
            tau_next = tau_next,
            sigma_k = sigma_k,
            xi_k = xi_k,
            ratio_candidate = ratio_candidate,
            increment_candidate = increment_candidate,
            branch = branch,
            common_residual = residual,
        ))

        converged && break
        x_curr = x_next
        z_prev = z_k
        Bz_prev = Bz_k
        tau_curr = tau_next
    end
    return records, converged
end

function write_stepsize_csv(path::String, records)
    open(path, "w") do io
        println(io, "k,f_evals,tau_k,tau_next,sigma_k,xi_k,ratio_candidate,increment_candidate,branch,common_residual")
        for r in records
            @printf(io, "%d,%d,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%s,%.16e\n",
                    r.k, r.f_evals, r.tau_k, r.tau_next, r.sigma_k, r.xi_k,
                    r.ratio_candidate, r.increment_candidate, r.branch, r.common_residual)
        end
    end
    return path
end

function write_stepsize_plot(path::String, records, case_id::String, init_idx::Int)
    ks = [r.k for r in records]
    tau = [max(r.tau_k, eps(Float64)) for r in records]
    ratio = [max(r.ratio_candidate, eps(Float64)) for r in records]
    increment = [max(r.increment_candidate, eps(Float64)) for r in records]
    plt = plot(ks, tau; yscale = :log10, label = L"\tau_k", lw = 2.4,
               xlabel = "Iteration", ylabel = "Stepsize", grid = true,
               title = "AEFBFP stepsize diagnostic: $(case_id), seed $(init_idx)")
    plot!(plt, ks, ratio; label = "ratio candidate", lw = 1.5, ls = :dash)
    plot!(plt, ks, increment; label = "increment candidate", lw = 1.5, ls = :dot)
    savefig(plt, path)
    return path
end

function diagnostics_main(args = ARGS)
    cfg = diagnostic_config(args)
    mkpath(DIAGNOSTIC_DIR)
    logpath, tee, _ = setup_logging("s11_aefbfp_stepsize_diagnostics"; logdir = LOGDIR)
    db = open_db(DB_PATH)
    try
        case = CASE_BY_NAME[cfg.case_id]
        prob = build_problem(case; gamma = cfg.gamma, snr_db = cfg.snr_db,
                             data_seed = dataset_seed(case, 1), n_inits = cfg.init_idx)
        init = prob.initial_points[cfg.init_idx]
        alg = build_algorithm(db, "AEFBFP")
        records, converged = run_stepsize_diagnostic(alg, prob, copy(init.x0);
                                                      maxiter = cfg.maxiter, eps = cfg.eps)
        isempty(records) && error("Diagnostic produced no iteration records")

        println(tee, "\n  Iteration stepsize table")
        @printf(tee, "  %6s  %12s  %12s  %12s  %12s  %-10s  %12s\n",
                "iter", "tau_k", "ratio", "increment", "tau_next", "min branch", "residual")
        println(tee, "  " * "-" ^ 94)
        for r in records
            if r.k == 1 || r.k % cfg.every == 0 || r.k == last(records).k
                @printf(tee, "  %6d  %12.4e  %12.4e  %12.4e  %12.4e  %-10s  %12.4e\n",
                        r.k, r.tau_k, r.ratio_candidate, r.increment_candidate,
                        r.tau_next, r.branch, r.common_residual)
            end
        end

        total_iterations = length(records)
        ratio_count = count(r -> r.branch == "ratio", records)
        increment_count = count(r -> r.branch == "increment", records)
        ratio_percent = 100 * ratio_count / total_iterations
        increment_percent = 100 * increment_count / total_iterations
        stem = "aefbfp_tau_$(cfg.case_id)_seed$(cfg.init_idx)"
        csv_path = write_stepsize_csv(joinpath(DIAGNOSTIC_DIR, stem * ".csv"), records)
        pdf_path = write_stepsize_plot(joinpath(DIAGNOSTIC_DIR, stem * ".pdf"), records,
                                       cfg.case_id, cfg.init_idx)
        final = last(records)

        println(tee, "AEFBFP stepsize diagnostic")
        println(tee, "  case/init       : $(cfg.case_id) / $(init.label)")
        println(tee, "  print interval  : every $(cfg.every) iteration(s)")
        println(tee, "  total iterations: $(total_iterations)")
        println(tee, "  converged       : $(converged)")
        println(tee, "  branch summary")
        @printf(tee, "    ratio     : %d / %d  (%.2f%%)\n",
                ratio_count, total_iterations, ratio_percent)
        @printf(tee, "    increment : %d / %d  (%.2f%%)\n",
                increment_count, total_iterations, increment_percent)
        @printf(tee, "  tau_1 / tau_end : %.6e / %.6e\n", first(records).tau_k, final.tau_next)
        @printf(tee, "  final residual  : %.6e\n", final.common_residual)
        println(tee, "  CSV             : $(csv_path)")
        println(tee, "  plot            : $(pdf_path)")
    finally
        teardown_logging(tee, logpath)
        SQLite.close(db)
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    diagnostics_main()
end
