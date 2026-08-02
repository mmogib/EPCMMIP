# Numerical-only smoke test for the saddle-point family.

include(joinpath(@__DIR__, "s30_benchmark.jl"))

function saddle_smoke_main()
    runtime = configure_reproducible_runtime!()
    prob = build_random_game(20; instance_index = 1, n_inits = 2)
    u = prob.initial_points[1].x0
    v = prob.initial_points[2].x0
    @assert abs(dot(prob.B(u), u)) <= 1.0e-11
    @assert norm(prob.B(u) - prob.B(v)) <= prob.metadata.L * norm(u - v) + 1.0e-11
    @assert prob.native_residual(u, Float64[]) >= 0
    @assert game_duality_gap(prob, u) >= -1.0e-12

    degenerate = build_duplicated_identity_game(20; n_inits = 3)
    for init in degenerate.initial_points
        target = degenerate_solution_projection(init.x0, 20)
        @assert degenerate.native_residual(target, Float64[]) <= 1.0e-13
        @assert game_duality_gap(degenerate, target) <= 1.0e-13
        @assert degenerate_projection_certificate(init.x0, target, 20) <= 1.0e-13
    end

    for method in ("AEFBFP", "HFB")
        alg = build_game_algorithm(method, prob)
        result = solve(alg, prob, copy(u);
                       stopping = (MaxIterStopping(5), NanStopping()), observers = ())
        @assert result.iterations == 5
        @assert all(isfinite, result.x)
        println("$method: iter=$(result.iterations), f_evals=$(result.f_evals), natural=$(prob.native_residual(result.x, Float64[]))")
    end
    println("saddle-point smoke passed on Julia $(runtime.julia_version), threads=$(runtime.julia_threads), BLAS=$(runtime.blas_threads)")
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    saddle_smoke_main()
end
