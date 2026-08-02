using Test
using LinearAlgebra

include(joinpath(@__DIR__, "..", "src", "includes.jl"))
include(joinpath(@__DIR__, "..", "scripts", "saddle_point", "problem_definition.jl"))
include(joinpath(@__DIR__, "..", "scripts", "saddle_point", "halpern_forward_backward.jl"))
include(joinpath(@__DIR__, "..", "scripts", "saddle_point", "diagnostics.jl"))

@testset "saddle-point problem invariants" begin
    p = build_random_game(12; instance_index = 1, n_inits = 3)
    @test p.metadata.m == 12
    @test p.metadata.n == 12
    @test length(p.initial_points) == 3
    @test length(unique(p.metadata.seeds.starts_x)) == 3
    @test length(unique(p.metadata.seeds.starts_y)) == 3
    @test p.metadata.hashes == build_random_game(12; instance_index = 1, n_inits = 3).metadata.hashes
    @test all(isapprox(sum(ip.x0[1:12]), 1.0; atol = 1e-14) for ip in p.initial_points)
    @test all(isapprox(sum(ip.x0[13:24]), 1.0; atol = 1e-14) for ip in p.initial_points)

    u = p.initial_points[1].x0
    Bu = p.B(u)
    @test abs(dot(Bu, u)) <= 1e-12
    v = p.initial_points[2].x0
    @test norm(p.B(u) - p.B(v)) <= p.metadata.L * norm(u - v) + 1e-12
    @test p.native_residual(u, Float64[]) >= 0.0
    @test game_duality_gap(p, u) >= -1e-12
end

@testset "duplicated-identity anchor projection" begin
    p = build_duplicated_identity_game(10; n_inits = 3)
    q = p.metadata.q
    infeasible = vcat(zeros(q + 1), ones(q))
    @test game_duality_gap(p, infeasible) >= 0.0
    for ip in p.initial_points
        target = degenerate_solution_projection(ip.x0, q)
        x = target[1:q+1]
        y = target[q+2:end]
        @test all(x .>= 0.0)
        @test all(y .>= 0.0)
        @test isapprox(sum(x), 1.0; atol = 1e-14)
        @test isapprox(sum(y), 1.0; atol = 1e-14)
        @test all(isapprox.(x[2:q], 1 / q; atol = 1e-14))
        @test isapprox(x[1] + x[q+1], 1 / q; atol = 1e-14)
        @test all(isapprox.(y, 1 / q; atol = 1e-14))
        @test game_duality_gap(p, target) <= 1e-14
        @test p.native_residual(target, Float64[]) <= 1e-14
        @test degenerate_projection_certificate(ip.x0, target, q) <= 1e-14
    end
end

@testset "game-local Halpern forward-backward and diagnostics" begin
    p = build_random_game(10; instance_index = 2, n_inits = 1)
    alg = HalpernForwardBackward(0.99 / p.metadata.L)
    natural = GameNaturalHistory(p.native_residual)
    result = solve(alg, p, copy(p.initial_points[1].x0);
                   stopping = (MaxIterStopping(4), NanStopping()),
                   observers = (natural,))
    @test result.iterations == 4
    @test result.f_evals == 5
    @test length(natural.records) == 4
    @test all(record.residual >= 0.0 for record in natural.records)

    tau = AEFBFPTauObserver(1.11, 0.05)
    state = SolverState(:AEFBFP, zeros(2))
    state.step_size = 0.05
    on_event!(tau, state, :init)
    state.k = 1
    state.step_size = 0.05 + 1.0
    on_event!(tau, state, :iter)
    state.k = 2
    state.step_size = 0.25
    on_event!(tau, state, :iter)
    @test tau.records[1].tau_index == 0
    @test tau.records[2].tau_index == 1
    @test !tau.records[2].ratio_branch
    @test tau.records[3].ratio_branch
    @test min_branch_fraction(tau) == 0.5
end
