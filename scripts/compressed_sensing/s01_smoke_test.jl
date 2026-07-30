# ============================================================================
# s01_smoke_test.jl
# ============================================================================
#
# Purpose
#   Quick sanity check for the compressed-sensing benchmark pipeline.
#
# What it does
#   - runs one compressed-sensing case
#   - uses two methods only (AEFBFP and IFRAB)
#   - uses one dataset and one timing repetition
#   - writes rows into the non-production tier
#   - allows untuned AEFBFP so the smoke test can run before s20
#
# How to run
#   julia --project=. scripts/compressed_sensing/s01_smoke_test.jl
#
# ============================================================================

include(joinpath(@__DIR__, "s30_benchmark.jl"))

function smoke_main()
    return benchmark_main([
        "--quick",
        "--force",
        "--allow-untuned-aefbfp",
        "--datasets=1",
        "--reps=1",
        "--cases=M256_N512_k30",
        "--methods=AEFBFP,IFRAB",
    ]; script_name = "s01_smoke_test")
end

if abspath(PROGRAM_FILE) == @__FILE__
    smoke_main()
end
