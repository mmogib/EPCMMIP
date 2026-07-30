function build_double_integrator_problem(K::Int; n_inits::Int = 10)
    n_inits >= 1 || throw(ArgumentError("build_double_integrator_problem: n_inits must be >= 1, got $n_inits"))
    K >= 10 || throw(ArgumentError("build_double_integrator_problem: K must be >= 10, got $K"))

    h = 2.0 / K
    c = Float64[-h * (K - j) for j in 0:K-1]
    L = 2.0 * h * K
    M_factor = 2.0 * h

    B_fn = let M_factor = M_factor, c = c
        z -> begin
            coeff = M_factor * sum(z)
            return coeff .+ c
        end
    end

    resolvent_A_fn = (z, ρ) -> clipping_box(z, -1.0, 1.0)
    native_residual_fn = let B_fn = B_fn
        (z, z_prev) -> begin
            Bz = B_fn(z)
            return 0.5 * sum(abs2, z .- clipping_box(z .- Bz, -1.0, 1.0))
        end
    end

    switch = round(Int, 0.6 * K)
    exact_control = vcat(fill(1.0, switch), fill(-1.0, K - switch))

    initial_points = InitialPoint[]
    for seed_idx in 1:n_inits
        rng = Xoshiro(UInt64(31_000_000 + 1_000 * K + seed_idx))
        z0 = -1.0 .+ 2.0 .* rand(rng, K)
        push!(initial_points, InitialPoint("seed$seed_idx", seed_idx, z0))
    end

    metadata = (
        L = L,
        K = K,
        h = h,
        c = c,
        M_factor = M_factor,
    )

    return TestProblem(
        31,
        "DoubleIntegrator_Local",
        K,
        B_fn,
        resolvent_A_fn,
        native_residual_fn,
        exact_control,
        initial_points,
        metadata,
    )
end

function double_integrator_state(control::Vector{Float64}, prob::TestProblem)
    h = prob.metadata.h
    K = length(control)
    t = collect(0.0:h:(K * h))
    w1 = zeros(Float64, K + 1)
    w2 = zeros(Float64, K + 1)
    for j in 1:K
        w1[j + 1] = w1[j] + h * w2[j]
        w2[j + 1] = w2[j] + h * control[j]
    end
    return (t = t, state1 = w1, state2 = w2)
end

const DOUBLE_INTEGRATOR_SPEC = (
    key = :double_integrator_control,
    problem_name = "double_integrator_control",
    display_name = "double-integrator optimal-control problem",
    control_label = "z",
    state_labels = ("w_1", "w_2"),
    final_time = 2.0,
    build_problem = build_double_integrator_problem,
    state_trajectory = double_integrator_state,
)
