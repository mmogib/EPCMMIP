using Test

include(joinpath(@__DIR__, "..", "scripts", "compressed_sensing", "s30_benchmark.jl"))

@testset "compressed-sensing manuscript protocol API" begin
    @test isdefined(Main, :build_manuscript_problem)
    @test isdefined(Main, :MANUSCRIPT_AEFBFP_PARAMS)

    if isdefined(Main, :build_manuscript_problem)
        case = DEFAULT_CASES[1]
        prob1 = build_manuscript_problem(case; case_index = 1, gamma = 1.0e-3, n_inits = 10)
        prob2 = build_manuscript_problem(case; case_index = 1, gamma = 1.0e-3, n_inits = 10)
        meta = prob1.metadata

        @test norm(meta.C * meta.C' - I, Inf) <= 1.0e-10
        @test count(!iszero, meta.x_star) == case.k
        @test all(abs.(meta.x_star[meta.x_star .!= 0.0]) .== 1.0)
        @test meta.y == meta.C * meta.x_star + meta.noise
        @test prob1.native_residual(ones(case.N), zeros(case.N)) == sqrt(case.N)
        @test prob1.native_residual(ones(case.N), Float64[]) == Inf
        @test length(prob1.initial_points) == 10
        @test all(ip -> norm(ip.x0) > 0.0, prob1.initial_points)
        @test prob1.initial_points[1].x0 != prob1.initial_points[2].x0
        @test meta.hashes == prob2.metadata.hashes
        @test all(prob1.initial_points[i].x0 == prob2.initial_points[i].x0 for i in 1:10)
        seed_values = [meta.seeds.matrix, meta.seeds.support, meta.seeds.signs,
                       meta.seeds.noise, meta.seeds.starts...]
        @test length(unique(seed_values)) == length(seed_values)
    end

    if isdefined(Main, :MANUSCRIPT_AEFBFP_PARAMS)
        alg = build_algorithm(nothing, "AEFBFP")
        @test alg.mu == 0.32
        @test alg.tau_0 == 0.05
        @test alg.xi_exp == 1.11
        @test alg.sigma_exp == 0.97
        @test alg.sigma_scale == 0.024
        @test build_algorithm(nothing, "RFBSM").theta == 1.0
        @test build_algorithm(nothing, "IRFBSM").theta == 1.0
        cfg = read_benchmark_config(String[])
        hash, input = cs_config_hash(alg, DEFAULT_CASES[1].problem, cfg)
        @test length(hash) == 12
        @test occursin("reps=3", input)
        @test occursin("warmup=2", input)
    end

    @test isdefined(Main, :manuscript_problem_start)
    if isdefined(Main, :manuscript_problem_start)
        selected = manuscript_problem_start(DEFAULT_CASES[3].problem, 4)
        direct = build_manuscript_problem(DEFAULT_CASES[3];
                                          case_index = 3,
                                          gamma = GAMMA_REF,
                                          n_inits = DEFAULT_INITIAL_POINTS)
        @test selected.case_index == 3
        @test selected.prob.metadata.protocol == "manuscript_v1"
        @test selected.prob.metadata.hashes == direct.metadata.hashes
        @test selected.init.label == "seed4"
        @test selected.init.seed_idx == 4
        @test array_sha256(selected.init.x0) == direct.metadata.hashes.starts[4]
        @test_throws ArgumentError manuscript_problem_start(DEFAULT_CASES[3].problem, 0)
        @test_throws ArgumentError manuscript_problem_start(DEFAULT_CASES[3].problem, 11)
        @test_throws ArgumentError manuscript_problem_start("unknown_case", 1)
    end

    plot_source = read(joinpath(@__DIR__, "..", "scripts", "compressed_sensing",
                                "s70_figures_tables.jl"), String)
    @test occursin("manuscript_problem_start", plot_source)
    @test !occursin("resolved_aefbfp_params(db;", plot_source)
    @test !occursin("prob = build_problem(case;", plot_source)
end
