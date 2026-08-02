function harmonic_exact_control(K::Int, h::Float64)
    t = collect(0.0:h:(3 * pi - h))
    return [cos(tj) >= 0 ? 1.0 : -1.0 for tj in t]
end

function harmonic_gradient(K::Int, h::Float64)
    Ad = [1.0 h; -h 1.0]
    Bd = [0.0, h]
    lambda = [0.0, 1.0]
    c = zeros(Float64, K)
    for j in K:-1:1
        c[j] = dot(Bd, lambda)
        lambda = Ad' * lambda
    end
    return c
end

function build_harmonic_problem(K::Int; n_inits::Int = 10)
    K >= 10 || throw(ArgumentError("K must be >= 10, got $K"))
    n_inits >= 1 || throw(ArgumentError("n_inits must be >= 1, got $n_inits"))

    T = 3 * pi
    h = T / K
    c = harmonic_gradient(K, h)
    exact_control = harmonic_exact_control(K, h)

    B_fn = let c = copy(c)
        z -> copy(c)
    end

    resolvent_A_fn = (z, rho) -> clipping_box(z, -1.0, 1.0)

    native_residual_fn = let B_fn = B_fn
        (z, z_prev) -> begin
            Bz = B_fn(z)
            return 0.5 * sum(abs2, z .- clipping_box(z .- Bz, -1.0, 1.0))
        end
    end

    initial_points = InitialPoint[]
    start_seeds = Int[]
    for seed_idx in 1:n_inits
        seed = 32_000_000 + 1_000 * K + seed_idx
        push!(start_seeds, seed)
        rng = Xoshiro(UInt64(seed))
        z0 = -1.0 .+ 2.0 .* rand(rng, K)
        push!(initial_points, InitialPoint("seed$seed_idx", seed_idx, z0))
    end


    seeds = (starts = start_seeds,)
    hashes = (
        operator_c = array_sha256(c),
        exact_control = array_sha256(exact_control),
        starts = [array_sha256(ip.x0) for ip in initial_points],
    )

    metadata = (
        L = 0.0,
        K = K,
        T = T,
        h = h,
        c = c,
        seeds = seeds,
        hashes = hashes,
    )

    return TestProblem(
        33,
        "HarmonicOscillator_Local",
        K,
        B_fn,
        resolvent_A_fn,
        native_residual_fn,
        exact_control,
        initial_points,
        metadata,
    )
end

function harmonic_state(control::Vector{Float64}, prob::TestProblem)
    h = prob.metadata.h
    K = length(control)
    t = collect(0.0:h:(K * h))
    x1 = zeros(Float64, K + 1)
    x2 = zeros(Float64, K + 1)
    for j in 1:K
        x1[j + 1] = x1[j] + h * x2[j]
        x2[j + 1] = x2[j] + h * (-x1[j] + control[j])
    end
    return (t = t, state1 = x1, state2 = x2)
end

const HARMONIC_OSCILLATOR_SPEC = (
    key = :harmonic_oscillator,
    problem_name = "harmonic_oscillator",
    display_name = "harmonic-oscillator optimal-control problem",
    control_label = "p",
    state_labels = ("x_1", "x_2"),
    final_time = 3 * pi,
    build_problem = build_harmonic_problem,
    state_trajectory = harmonic_state,
)
