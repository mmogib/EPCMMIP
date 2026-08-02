# External monitoring-free timing pass for definitive double-integrator OC.

include(joinpath(@__DIR__, "s30_benchmark.jl"))
include(joinpath(@__DIR__, "..", "s35_timing_v2_common.jl"))

# Protocol anchors: OC_DEFAULT_DIMS, OC_BENCH_INITS, current_benchmark_hashes,
# oc_config_hash, run_timing_v2, warmups = 2, repetitions = 3,
# min_batch_seconds = quick ? 0.005 : 0.1, observers = (),
# monitor_residual = false, record_elapsed = false, timing_signature,
# insert_timing_v2_repetition!, write_run_manifest.

if abspath(PROGRAM_FILE) == @__FILE__
    oc_timing_v2_main(DOUBLE_INTEGRATOR_SPEC,
                      "optimal_control_double_integrator")
end
