using Test

include(joinpath(@__DIR__, "..", "scripts", "optimal_control",
                 "double_integrator_control", "s30_benchmark.jl"))
include(joinpath(@__DIR__, "..", "scripts", "optimal_control",
                 "harmonic_oscillator", "problem_definition.jl"))

@testset "optimal-control definitive protocol" begin
    @test isdefined(Main, :OC_MANUSCRIPT_AEFBFP_PARAMS)
    @test isdefined(Main, :OC_DEFAULT_REPS)

    if isdefined(Main, :OC_MANUSCRIPT_AEFBFP_PARAMS)
        alg = build_algorithm(nothing, DOUBLE_INTEGRATOR_SPEC, "AEFBFP")
        @test alg.mu == 0.32
        @test alg.tau_0 == 0.05
        @test alg.xi_exp == 1.11
        @test alg.sigma_exp == 0.97
        @test alg.sigma_scale == 0.024
    end

    if isdefined(Main, :OC_DEFAULT_REPS)
        @test read_benchmark_config(String[]).reps == 3
        @test read_benchmark_config(["--quick"]).reps == 1
    end

    sample_result(t; flag = :converged, iterations = 7, f_evals = 9) = make_result(
        converged = flag == :converged,
        iterations = iterations,
        f_evals = f_evals,
        cpu_time = t,
        x = [1.0, 2.0],
        flag = flag,
        residual = 1.0e-8,
        scaled_residual = 2.0e-8,
    )
    repeated = [sample_result(0.30), sample_result(0.10), sample_result(0.20)]
    @test benchmark_result_signature(repeated[1]) == (:converged, 7, 9)
    stored = median_cpu_result(repeated)
    @test stored.cpu_time == 0.20
    @test benchmark_result_signature(stored) == (:converged, 7, 9)
    inconsistent = [sample_result(0.10), sample_result(0.20; iterations = 8)]
    @test_throws ErrorException median_cpu_result(inconsistent)

    cfg = read_benchmark_config(String[])
    hash, hash_input = oc_config_hash(build_algorithm(nothing, DOUBLE_INTEGRATOR_SPEC, "AEFBFP"),
                                      DOUBLE_INTEGRATOR_SPEC, cfg)
    @test length(hash) == 12
    @test occursin("protocol=oc_manuscript_v1", hash_input)
    @test occursin("reps=3", hash_input)
    @test occursin("consec=2", hash_input)

    for builder in (build_double_integrator_problem, build_harmonic_problem)
        p1 = builder(50; n_inits = 10)
        p2 = builder(50; n_inits = 10)
        @test length(p1.initial_points) == 10
        @test haskey(p1.metadata, :seeds)
        @test haskey(p1.metadata, :hashes)
        if haskey(p1.metadata, :seeds) && haskey(p1.metadata, :hashes)
            @test length(unique(p1.metadata.seeds.starts)) == 10
            @test p1.metadata.hashes == p2.metadata.hashes
        end
        @test all(p1.initial_points[i].x0 == p2.initial_points[i].x0 for i in 1:10)
        @test p1.native_residual(p1.initial_points[1].x0, Float64[]) >= 0.0
    end
end
