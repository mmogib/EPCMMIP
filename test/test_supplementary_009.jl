using Test

const SUPPLEMENTARY_SOURCES = Dict(
    "cs" => joinpath(@__DIR__, "..", "scripts", "compressed_sensing", "s40_xi_matched.jl"),
    "oc_common" => joinpath(@__DIR__, "..", "scripts", "optimal_control", "s40_xi_matched_common.jl"),
    "di" => joinpath(@__DIR__, "..", "scripts", "optimal_control", "double_integrator_control", "s40_xi_matched.jl"),
    "ho" => joinpath(@__DIR__, "..", "scripts", "optimal_control", "harmonic_oscillator", "s40_xi_matched.jl"),
    "game" => joinpath(@__DIR__, "..", "scripts", "saddle_point", "s40_xi_matched.jl"),
)

@testset "009 supplementary runner contracts" begin
    for (_, path) in SUPPLEMENTARY_SOURCES
        @test isfile(path)
        isfile(path) || continue
        source = read(path, String)
        @test !occursin("using Plots", source)
        @test !occursin("using StatsPlots", source)
        @test occursin("s40_xi_matched", source)
    end

    for family in ("cs", "oc_common", "game")
        path = SUPPLEMENTARY_SOURCES[family]
        isfile(path) || continue
        source = read(path, String)
        @test occursin("xi_exp = 2.0", source)
        @test occursin("reps = 3", source)
        @test occursin("n_inits = 10", source)
        @test occursin("retain_supplementary_repetitions!", source)
        @test occursin("write_run_manifest", source)
    end

    game_path = SUPPLEMENTARY_SOURCES["game"]
    if isfile(game_path)
        game_source = read(game_path, String)
        @test occursin("maxiter = 100_000", game_source)
        @test occursin("anchor_diagnostics", game_source)
        @test occursin("insert_anchor_metrics!", game_source)
    end

    common_path = SUPPLEMENTARY_SOURCES["oc_common"]
    if isfile(common_path)
        oc_source = read(common_path, String)
        @test occursin("OC_EPS_REF", oc_source)
        @test occursin("OC_NMAX_REF", oc_source)
        @test occursin("NativeResRecorder", oc_source)
    end
end
