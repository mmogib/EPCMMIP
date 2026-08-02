using Test

const TEST_JCODE_ROOT = normpath(joinpath(@__DIR__, ".."))
const OC_PROTOCOL_PATH = joinpath(TEST_JCODE_ROOT, "scripts", "optimal_control",
                                  "manuscript_protocol.jl")
const DI_S30_PATH = joinpath(TEST_JCODE_ROOT, "scripts", "optimal_control",
                             "double_integrator_control", "s30_benchmark.jl")
const DI_S70_PATH = joinpath(TEST_JCODE_ROOT, "scripts", "optimal_control",
                             "double_integrator_control", "s70_figures_tables.jl")
const HO_S30_PATH = joinpath(TEST_JCODE_ROOT, "scripts", "optimal_control",
                             "harmonic_oscillator", "s30_benchmark.jl")
const HO_S70_PATH = joinpath(TEST_JCODE_ROOT, "scripts", "optimal_control",
                             "harmonic_oscillator", "s70_figures_tables.jl")
const CS_S70_PATH = joinpath(TEST_JCODE_ROOT, "scripts", "compressed_sensing",
                             "s70_figures_tables.jl")
const GAME_S71_PATH = joinpath(TEST_JCODE_ROOT, "scripts", "saddle_point",
                               "s71_figures.jl")

@testset "manuscript figure handoff" begin
    @test isfile(OC_PROTOCOL_PATH)
    if isfile(OC_PROTOCOL_PATH)
        include(OC_PROTOCOL_PATH)
        @test OC_MANUSCRIPT_AEFBFP_PARAMS == (
            mu = 0.32,
            tau_0 = 0.05,
            xi_rule = :power,
            sigma_rule = :power,
            xi_exp = 1.11,
            sigma_exp = 0.97,
            sigma_scale = 0.024,
        )
        protocol_source = read(OC_PROTOCOL_PATH, String)
        @test !occursin(r"(?m)^\s*(using|import)\s+(Plots|LaTeXStrings)", protocol_source)
    end

    include_token = "include(joinpath(@__DIR__, \"..\", \"manuscript_protocol.jl\"))"
    for path in (DI_S30_PATH, DI_S70_PATH, HO_S30_PATH, HO_S70_PATH)
        @test occursin(include_token, read(path, String))
    end

    stale_values = ("0.32475054644276846", "0.052386951978823273",
                    "1.1090105055152015", "0.9709015323685187",
                    "0.02390913974996533")
    for path in (DI_S70_PATH, HO_S70_PATH)
        source = read(path, String)
        @test all(value -> !occursin(value, source), stale_values)
        report_tail = last(split(source, "function read_report_config(args)"; limit = 2))
        @test !occursin("cfg.allow_untuned_aefbfp", report_tail)
        @test !occursin("cfg.aefbfp_preset", report_tail)
        @test !occursin("cfg.aefbfp_round_digits", report_tail)
        @test occursin("build_report_algorithm(method_name)", source)
    end

    for path in (CS_S70_PATH, DI_S70_PATH, HO_S70_PATH, GAME_S71_PATH)
        source = read(path, String)
        @test occursin("savefig(deepcopy(plt)", source)
        @test !occursin("savefig(plt,", source)
    end

    di_source = read(DI_S70_PATH, String)
    @test occursin("const DI_TIME_TICKS", di_source)
    @test occursin("ylims = (1.0e-6, 1.0e1)", di_source)
    @test occursin("hline!(plt, [1.0e-5]", di_source)

    ho_source = read(HO_S70_PATH, String)
    @test occursin("const HO_TIME_TICKS", ho_source)
    @test occursin("ylims = (1.0e-6, 1.0e0)", ho_source)
    @test occursin("hline!(plt, [1.0e-5]", ho_source)

    game_source = read(GAME_S71_PATH, String)
    @test occursin("size = (900, 620)", game_source)
    @test occursin("dpi = 220", game_source)
    @test occursin("xlabel = \"Event index\"", game_source)
    @test occursin("markeralpha = 0.45", game_source)
    @test occursin("tau_padding", game_source)

    cs_source = read(CS_S70_PATH, String)
    @test occursin("joinpath(FIGDIR, \"cs_signal_panels\")", cs_source)
    @test occursin("y = Cx* + ε", cs_source)
    @test !occursin("y = Ax* + ε", cs_source)
    @test occursin(raw"k=$(case.k)", cs_source)
    @test !occursin(raw"K=$(case.k)", cs_source)
    @test occursin(raw"start $(dataset_idx)", cs_source)

    for source in (di_source, ho_source)
        @test !occursin(r"xlims\s*=\s*\(0,\s*250\)", source)
        @test occursin(raw"start $(seed_idx)", source)
        @test occursin(raw"start $(rep.init.seed_idx)", source)
    end

    @test occursin("random-500, start 1", game_source)
    @test occursin("local_figdir(spec)", di_source)
    @test occursin("local_figdir(spec)", ho_source)
    @test occursin("joinpath(GAME_FIGDIR", game_source)
end
