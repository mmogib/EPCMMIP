using Test

const RUNNER_TEST_ROOT = normpath(joinpath(@__DIR__, ".."))
const TIMING_RUNNERS = (
    cs = joinpath(RUNNER_TEST_ROOT, "scripts", "compressed_sensing",
                  "s35_timing_v2.jl"),
    di = joinpath(RUNNER_TEST_ROOT, "scripts", "optimal_control",
                  "double_integrator_control", "s35_timing_v2.jl"),
    ho = joinpath(RUNNER_TEST_ROOT, "scripts", "optimal_control",
                  "harmonic_oscillator", "s35_timing_v2.jl"),
    game = joinpath(RUNNER_TEST_ROOT, "scripts", "saddle_point",
                    "s35_timing_v2.jl"),
)
const TIMING_TABLES = (
    joinpath(RUNNER_TEST_ROOT, "scripts", "compressed_sensing", "s70_tables.jl"),
    joinpath(RUNNER_TEST_ROOT, "scripts", "optimal_control",
             "double_integrator_control", "s70_tables.jl"),
    joinpath(RUNNER_TEST_ROOT, "scripts", "optimal_control",
             "harmonic_oscillator", "s70_tables.jl"),
    joinpath(RUNNER_TEST_ROOT, "scripts", "saddle_point", "s70_tables.jl"),
)

@testset "timing-v2 family runner protocol" begin
    for path in values(TIMING_RUNNERS)
        @test isfile(path)
        isfile(path) || continue
        source = read(path, String)
        @test occursin("run_timing_v2", source)
        @test occursin("warmups = 2", source)
        @test occursin("repetitions = 3", source)
        @test occursin("min_batch_seconds = quick ? 0.005 : 0.1", source)
        @test occursin("monitor_residual = false", source)
        @test occursin("record_elapsed = false", source)
        @test occursin("observers = ()", source)
        @test occursin("insert_timing_v2_repetition!", source)
        @test occursin("write_run_manifest", source)
        @test !occursin(r"(?m)^\s*(using|import)\s+(Plots|LaTeXStrings)", source)
    end

    if all(isfile, values(TIMING_RUNNERS))
        cs = read(TIMING_RUNNERS.cs, String)
        @test occursin("DEFAULT_CASES", cs)
        @test occursin("DEFAULT_INITIAL_POINTS", cs)
        @test occursin("current_method_hashes", cs)

        for path in (TIMING_RUNNERS.di, TIMING_RUNNERS.ho)
            source = read(path, String)
            @test occursin("OC_DEFAULT_DIMS", source)
            @test occursin("OC_BENCH_INITS", source)
            @test occursin("oc_config_hash", source)
        end

        game = read(TIMING_RUNNERS.game, String)
        @test occursin("GAME_INSTANCE_NAMES", game)
        @test occursin("GAME_METHOD_NAMES", game)
        @test occursin("game_config_hash", game)
        @test occursin("reps = 3", game)
    end

    for path in TIMING_TABLES
        source = read(path, String)
        @test occursin("timing_v2_protocol_hash", source)
        @test occursin("min_batch_seconds = 0.1", source)
        @test occursin("timing_ms", source)
        @test occursin("legacy_cpu_time", source)
        @test occursin("CPU (ms)", source)
    end
end
