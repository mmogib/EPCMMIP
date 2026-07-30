# Independent verification of the compressed-sensing fixed-point residual.
# Solves the same LASSO instance with FISTA at the known safe step 1/L, then
# evaluates both the B1 unit-step residual and the 1/L-scaled residual.

include(joinpath(@__DIR__, "s30_benchmark.jl"))

function fista_verify(prob::TestProblem; maxiter::Int=100_000, report_every::Int=5_000)
    L = prob.metadata.L
    x = zeros(Float64, prob.dim)
    y = copy(x)
    t = 1.0
    for k in 1:maxiter
        x_next = prob.resolvent_A(y .- prob.B(y) ./ L, inv(L))
        t_next = (1 + sqrt(1 + 4t^2)) / 2
        y = x_next .+ ((t - 1) / t_next) .* (x_next .- x)
        x = x_next
        if k == 1 || k % report_every == 0
            unit = cs_optimality_residual(x, prob)
            scaled = cs_optimality_residual(x, prob; lambda=inv(L))
            @printf("  FISTA k=%6d  unit=%.4e  scaled=%.4e\n", k, unit, scaled)
        end
    end
    return x
end

function main()
    case = CASE_BY_NAME["M256_N512_k30"]
    prob = build_problem(case; gamma=GAMMA_REF, snr_db=SNR_DB_REF,
                         data_seed=dataset_seed(case, 1), n_inits=1)
    println("Residual verification on M256_N512_k30, dataset 1")
    @printf("  L = %.8e, safe FISTA step = %.8e\n", prob.metadata.L, inv(prob.metadata.L))
    x = fista_verify(prob)
    unit = cs_optimality_residual(x, prob)
    scaled = cs_optimality_residual(x, prob; lambda=inv(prob.metadata.L))
    @printf("  Final unit-step residual   = %.12e\n", unit)
    @printf("  Final 1/L-step residual    = %.12e\n", scaled)
    scaled < 1e-5 || error("Reference FISTA did not reach the conventional 1/L-scaled residual tolerance")
    println("  PASS: the 1/L-scaled fixed-point residual reaches 1e-5.")
    println("  NOTE: with L >> 1, the raw unit-step residual is about L times stricter near a solution.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
