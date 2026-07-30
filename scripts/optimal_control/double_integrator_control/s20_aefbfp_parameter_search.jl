include(joinpath(@__DIR__, "s30_benchmark.jl"))

if abspath(PROGRAM_FILE) == @__FILE__
    println("No local AEFBFP tuning is used for $(DOUBLE_INTEGRATOR_SPEC.problem_name).")
    println("This benchmark uses the fixed AEFBFP preset chosen for the optimal-control experiments.")
end
