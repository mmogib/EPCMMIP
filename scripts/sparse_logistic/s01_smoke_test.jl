# ============================================================================
# s01_smoke_test.jl
# ============================================================================
#
# Purpose
#   Quick DB-backed sanity check for the sparse-logistic benchmark pipeline.
#
# What it does
#   - runs one sparse-logistic case
#   - uses two methods only (AEFBFP and IFRAB)
#   - uses one dataset and one timing repetition
#   - writes rows into the non-production tier
#
# How to run
#   julia --project=. scripts/sparse_logistic/s01_smoke_test.jl
#
# ============================================================================

include(joinpath(@__DIR__, "s30_benchmark.jl"))

function smoke_main()
    return benchmark_main([
        "--quick",
        "--force",
        "--datasets=1",
        "--reps=1",
        "--cases=m128_n64",
        "--methods=AEFBFP,IFRAB",
    ]; script_name = "s01_smoke_test")
end

if abspath(PROGRAM_FILE) == @__FILE__
    smoke_main()
end
