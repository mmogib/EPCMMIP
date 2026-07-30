# Compare unit-step and 1/L-scaled LASSO residuals without changing stopping.

include(joinpath(@__DIR__, "s30_benchmark.jl"))
const DIAGNOSTIC_DIR = joinpath(RESULT_ROOT, "diagnostics")

function main(args=ARGS)
    maxiter = 5000
    eps = 1e-5
    case = CASE_BY_NAME["M256_N512_k30"]
    mkpath(DIAGNOSTIC_DIR)
    logpath, tee, _ = setup_logging("s17_compare_residual_scales"; logdir=LOGDIR)
    db = open_db(DB_PATH)
    try
        prob = build_problem(case; gamma=GAMMA_REF, snr_db=SNR_DB_REF,
                             data_seed=dataset_seed(case, 1), n_inits=10)
        rows = NamedTuple[]
        println(tee, "Residual-scale comparison: unit stopping remains active; both residuals are recorded at termination.")
        for T in METHOD_TYPES
            method = name(T)
            alg = build_algorithm(db, method)
            for init in prob.initial_points
                rec = NativeResRecorder(prob.native_residual)
                result = solve(alg, prob, copy(init.x0);
                               stopping=make_stopping(prob, eps, maxiter; consec=2), observers=(rec,))
                unit = cs_optimality_residual(result.x, prob)
                scaled = cs_optimality_residual(result.x, prob; lambda=inv(prob.metadata.L))
                push!(rows, (method=method, init=init.label, iterations=result.iterations,
                             unit=unit, scaled=scaled, unit_hit=unit <= eps,
                             scaled_hit=scaled <= eps, flag=string(result.flag)))
            end
        end
        println(tee, "\n  Method       unit<=1e-5  scaled<=1e-5  median unit  median scaled")
        println(tee, "  ------------------------------------------------------------------")
        for T in METHOD_TYPES
            method = name(T); sub = [r for r in rows if r.method == method]
            @printf(tee, "  %-12s %4d/10       %4d/10       %10.3e   %10.3e\n", method,
                    count(r -> r.unit_hit, sub), count(r -> r.scaled_hit, sub),
                    median([r.unit for r in sub]), median([r.scaled for r in sub]))
        end
        path = joinpath(DIAGNOSTIC_DIR, "residual_scale_comparison_$(case.problem).csv")
        open(path, "w") do io
            println(io, "method,init,iterations,unit_residual,l_scaled_residual,unit_hit,scaled_hit,flag")
            for r in rows
                @printf(io, "%s,%s,%d,%.16e,%.16e,%s,%s,%s\n", r.method,r.init,r.iterations,r.unit,r.scaled,string(r.unit_hit),string(r.scaled_hit),r.flag)
            end
        end
        println(tee, "\n  CSV: $(path)")
    finally
        teardown_logging(tee, logpath); SQLite.close(db)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__; main(); end
