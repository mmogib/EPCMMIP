# 009 xi-matched AEFBFP companion: double integrator, K=100. No plotting.

include(joinpath(@__DIR__, "s30_benchmark.jl"))
include(joinpath(@__DIR__, "..", "s40_xi_matched_common.jl"))

if abspath(PROGRAM_FILE) == @__FILE__
    oc_xi_matched_main(DOUBLE_INTEGRATOR_SPEC, [100]) # s40_xi_matched
end
