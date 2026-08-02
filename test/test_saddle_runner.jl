using Test

include(joinpath(@__DIR__, "..", "scripts", "saddle_point", "s70_tables.jl"))

@testset "saddle-point benchmark protocol" begin
    cfg = read_game_config(String[])
    @test cfg.reps == 3
    @test cfg.initial_points == 10
    @test cfg.eps == 1.0e-6
    @test cfg.maxiter == 10_000
    @test cfg.consec == 2
    @test cfg.instances == ["random_100", "random_500", "random_1000", "duplicated_identity_q100"]
    @test cfg.methods == ["AEFBFP", "VAFBS", "MDITSM", "RFBSM", "IRFBSM", "IFRAB", "HFB"]
    @test read_game_config(["--quick"]).reps == 1
    @test read_game_config(["--pilot-1000"]).initial_points == 1
    @test read_game_config(["--pilot-1000"]).instances == ["random_1000"]

    prob = build_random_game(10; instance_index = 1, n_inits = 1)
    @test build_game_algorithm("AEFBFP", prob).mu == 0.32
    @test build_game_algorithm("RFBSM", prob) == RFBSM(:paper)
    hfb = build_game_algorithm("HFB", prob)
    @test hfb.lambda == 0.99 / prob.metadata.L

    hash, input = game_config_hash(build_game_algorithm("AEFBFP", prob), prob,
                                   read_game_config(["--quick"]))
    @test length(hash) == 12
    @test occursin("protocol=game_manuscript_v1", input)
    @test occursin(prob.metadata.hashes.matrix, input)

    result(t; iterations = 4) = make_result(
        converged = false, iterations = iterations, f_evals = 5,
        cpu_time = t, x = copy(prob.initial_points[1].x0), flag = :maxiter)
    @test game_median_cpu_result([result(0.3), result(0.1), result(0.2)]).cpu_time == 0.2
    @test_throws ErrorException game_median_cpu_result([result(0.1), result(0.2; iterations = 5)])

    capped = DataFrame(method = ["X", "X"], problem = ["P", "P"],
                       converged = [0, 0], iterations = [10, 10],
                       f_evals = [11, 11], cpu_time = [0.1, 0.2],
                       duality_gap = [0.2, 0.4],
                       min_branch_fraction = Union{Missing,Float64}[missing, missing])
    capped_stats = game_table_stats(capped, "X", "P")
    @test capped_stats.success == 0
    @test isinf(capped_stats.iter)
    @test capped_stats.gap ≈ 0.3
end
