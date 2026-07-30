# Read-only AEFBFP increment-tail sensitivity for compressed sensing.
# Compares xi_k=(k+1)^(-p), p in {1.11, 2.0, current exponent}.

include(joinpath(@__DIR__, "s30_benchmark.jl"))

const DIAGNOSTIC_DIR = joinpath(RESULT_ROOT, "diagnostics")

function option(args, name, default)
    prefix = "--" * name * "="
    for arg in args
        startswith(arg, prefix) && return arg[length(prefix)+1:end]
    end
    default
end

function replace_xi(alg::AEFBFP, exponent::Float64)
    AEFBFP(mu=alg.mu, tau_0=alg.tau_0, xi_rule=:power,
           sigma_rule=alg.sigma_rule, xi_exp=exponent,
           sigma_exp=alg.sigma_exp, sigma_scale=alg.sigma_scale)
end

function xi_sweep_main(args=ARGS)
    case_id = option(args, "case", "M256_N512_k30")
    haskey(CASE_BY_NAME, case_id) || error("Unknown --case=$case_id")
    inits = parse(Int, option(args, "inits", "10"))
    maxiter = parse(Int, option(args, "maxiter", string(NMAX_REF)))
    eps = parse(Float64, option(args, "eps", string(EPS_REF)))
    1 <= inits <= DEFAULT_INITIAL_POINTS || error("--inits must be 1:$(DEFAULT_INITIAL_POINTS)")
    exponents = [1.11, 2.0]
    logpath, tee, _ = setup_logging("s15_aefbfp_xi_sweep"; logdir=LOGDIR)
    db = open_db(DB_PATH)
    try
        case = CASE_BY_NAME[case_id]
        prob = build_problem(case; gamma=GAMMA_REF, snr_db=SNR_DB_REF,
                             data_seed=dataset_seed(case, 1), n_inits=inits)
        base = build_algorithm(db, "AEFBFP")
        push!(exponents, base.xi_exp)
        exponents = unique(exponents)
        rows = NamedTuple[]
        println(tee, "AEFBFP xi exponent sweep: $(case_id), $(inits) starts, maxiter=$(maxiter)")
        for p in exponents
            alg = replace_xi(base, p)
            for init in prob.initial_points
                rec = NativeResRecorder(prob.native_residual)
                result = solve(alg, prob, copy(init.x0);
                               stopping=make_stopping(prob, eps, maxiter; consec=2), observers=(rec,))
                push!(rows, (p=p, init=init.label, converged=result.converged,
                             iterations=result.iterations, f_evals=result.f_evals,
                             residual=rec.value, flag=string(result.flag)))
                @printf(tee, "  p=%4.2f init=%-7s conv=%-5s iter=%5d residual=%10.3e\n",
                        p, init.label, string(result.converged), result.iterations, rec.value)
            end
        end
        println(tee, "\n  Summary")
        @printf(tee, "  %6s %9s %12s %12s\n", "p", "success", "med iter", "med residual")
        for p in exponents
            sub = [r for r in rows if r.p == p]
            solved = [r for r in sub if r.converged]
            if isempty(solved)
                @printf(tee, "  %6.2f %4d/%-4d %12s %12.3e\n", p, 0, length(sub), "DNC", median([r.residual for r in sub]))
            else
                @printf(tee, "  %6.2f %4d/%-4d %12.1f %12.3e\n", p, length(solved), length(sub), median(Float64[r.iterations for r in solved]), median(Float64[r.residual for r in solved]))
            end
        end
        mkpath(DIAGNOSTIC_DIR)
        path = joinpath(DIAGNOSTIC_DIR, "aefbfp_xi_sweep_$(case_id).csv")
        open(path, "w") do io
            println(io, "xi_exp,init,converged,iterations,f_evals,common_residual,flag")
            for r in rows
                @printf(io, "%.16e,%s,%s,%d,%d,%.16e,%s\n", r.p, r.init, string(r.converged), r.iterations, r.f_evals, r.residual, r.flag)
            end
        end
        println(tee, "  CSV: $(path)")
    finally
        teardown_logging(tee, logpath)
        SQLite.close(db)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    xi_sweep_main()
end
