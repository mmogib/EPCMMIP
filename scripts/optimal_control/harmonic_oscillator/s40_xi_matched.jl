# 009 xi-matched AEFBFP companion: harmonic oscillator, K=50,100,200. No plotting.

include(joinpath(@__DIR__, "s30_benchmark.jl"))
include(joinpath(@__DIR__, "..", "s40_xi_matched_common.jl"))

if abspath(PROGRAM_FILE) == @__FILE__
    oc_xi_matched_main(HARMONIC_OSCILLATOR_SPEC, collect(OC_DEFAULT_DIMS)) # s40_xi_matched
end
