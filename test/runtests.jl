using Test

const JCODE_TEST_ROOT = normpath(joinpath(@__DIR__, ".."))
const REPRO_SOURCE = joinpath(JCODE_TEST_ROOT, "src", "reproducibility.jl")

@testset "reproducibility source" begin
    @test isfile(REPRO_SOURCE)
end

if isfile(REPRO_SOURCE)
    include(joinpath(JCODE_TEST_ROOT, "src", "deps.jl"))
    include(REPRO_SOURCE)
    include(joinpath(JCODE_TEST_ROOT, "src", "resolvents.jl"))

    @testset "runtime fingerprint" begin
        info = configure_reproducible_runtime!()
        @test info.julia_version == "1.12.6"
        @test info.julia_threads == 1
        @test info.blas_threads == 1
        @test !isempty(info.blas_config)
    end

    @testset "realized-array hashes" begin
        A = reshape(Float64.(1:6), 2, 3)
        h = array_sha256(A)
        @test length(h) == 64
        @test h == array_sha256(copy(A))
        A[1] += 1.0
        @test h != array_sha256(A)
        @test h != array_sha256(reshape(copy(A), 3, 2))
    end

    @testset "run manifest API" begin
        @test isdefined(Main, :file_sha256)
        @test isdefined(Main, :write_run_manifest)
        if isdefined(Main, :file_sha256) && isdefined(Main, :write_run_manifest)
            mktempdir() do dir
                source = joinpath(dir, "source.txt")
                write(source, "manifest-test")
                @test length(file_sha256(source)) == 64
                @test file_sha256(joinpath(dir, "missing")) === nothing

                target = joinpath(dir, "run_manifest.json")
                write_run_manifest(target;
                    runtime = (julia_version = "1.12.6",),
                    protocol = (name = "test",),
                    seeds = (matrix = 1,),
                    hashes = (matrix = "abc",),
                    parameters = (mu = 0.32,),
                    project_manifest = source,
                )
                parsed = JSON3.read(read(target, String))
                @test parsed.runtime.julia_version == "1.12.6"
                @test parsed.manifest_sha256 == file_sha256(source)
            end
        end
    end


    @testset "simplex projection API" begin
        @test isdefined(Main, :project_simplex)
        @test isdefined(Main, :project_product_simplex)
        if isdefined(Main, :project_simplex) && isdefined(Main, :project_product_simplex)
            p = project_simplex([0.2, 0.3, 0.5])
            @test p ≈ [0.2, 0.3, 0.5]
            @test project_simplex([2.0, -1.0, 0.0]) ≈ [1.0, 0.0, 0.0]

            v = [-0.7, 0.1, 2.0, 0.4]
            p = project_simplex(v)
            @test all(p .>= 0.0)
            @test sum(p) ≈ 1.0 atol = 1.0e-14
            for e in eachindex(p)
                vertex = zeros(length(p)); vertex[e] = 1.0
                @test dot(v - p, vertex - p) <= 1.0e-12
            end

            u = [2.0, -1.0, 0.0, -3.0, 4.0]
            pp = project_product_simplex(u, 3)
            @test pp[1:3] ≈ [1.0, 0.0, 0.0]
            @test pp[4:5] ≈ [0.0, 1.0]
            @test_throws ArgumentError project_product_simplex(u, 0)
            @test_throws ArgumentError project_product_simplex(u, length(u))
        end
    end
end
