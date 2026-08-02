# Deterministic zero-sum matrix-game instances for the non-cocoercive test.

const GAME_SEED_BASE = 2_026_080_100_000
const GAME_RANDOM_SIZES = (100, 500, 1000)
const GAME_DEGENERATE_Q = 100

function game_named_seeds(instance_index::Int, n_inits::Int)
    base = GAME_SEED_BASE + 100_000 * instance_index
    return (
        matrix = base + 1,
        starts_x = [base + 1_000 + 2 * i for i in 1:n_inits],
        starts_y = [base + 2_000 + 2 * i for i in 1:n_inits],
    )
end

function dirichlet_one(seed::Int, dim::Int)
    rng = Xoshiro(UInt64(seed))
    values = randexp(rng, dim)
    return values ./ sum(values)
end

function game_operator(K::Matrix{Float64})
    m, n = size(K)
    return u -> begin
        length(u) == n + m || throw(DimensionMismatch("Expected $(n + m) game coordinates, got $(length(u))"))
        x = @view u[1:n]
        y = @view u[n+1:n+m]
        return vcat(K' * y, -(K * x))
    end
end

function build_game_problem(K::Matrix{Float64}, problem_id::Int, problem_name::String;
                            instance_kind::Symbol, instance_index::Int,
                            n_inits::Int = 10, matrix_seed::Union{Nothing,Int} = nothing,
                            q::Union{Nothing,Int} = nothing)
    n_inits >= 1 || throw(ArgumentError("n_inits must be >= 1"))
    m, n = size(K)
    B_fn = game_operator(K)
    resolvent_fn = (u, rho) -> project_product_simplex(u, n)
    native_fn = let B_fn = B_fn, n = n
        (u, u_prev) -> norm(u .- project_product_simplex(u .- B_fn(u), n))
    end

    seed_index = instance_kind === :random ? instance_index : 4
    all_seeds = game_named_seeds(seed_index, n_inits)
    initial_points = InitialPoint[]
    starts_x = Vector{String}()
    starts_y = Vector{String}()
    starts_u = Vector{String}()
    for i in 1:n_inits
        x0 = dirichlet_one(all_seeds.starts_x[i], n)
        y0 = dirichlet_one(all_seeds.starts_y[i], m)
        u0 = vcat(x0, y0)
        push!(initial_points, InitialPoint("seed$i", i, u0))
        push!(starts_x, array_sha256(x0))
        push!(starts_y, array_sha256(y0))
        push!(starts_u, array_sha256(u0))
    end

    L = opnorm(K)
    L > 0 || error("Game matrix must be nonzero.")
    exact = if instance_kind === :duplicated_identity
        c = 1.0 / something(q)
        vcat(c / 2, fill(c, something(q) - 1), c / 2, fill(c, something(q)))
    else
        zeros(Float64, n + m)
    end
    seeds = (
        matrix = matrix_seed,
        starts_x = all_seeds.starts_x,
        starts_y = all_seeds.starts_y,
    )
    hashes = (
        matrix = array_sha256(K),
        starts_x = starts_x,
        starts_y = starts_y,
        starts = starts_u,
    )
    metadata = (
        K = K,
        m = m,
        n = n,
        L = L,
        instance_kind = instance_kind,
        instance_index = instance_index,
        q = q,
        seeds = seeds,
        hashes = hashes,
    )
    return TestProblem(problem_id, problem_name, n + m, B_fn, resolvent_fn,
                       native_fn, exact, initial_points, metadata)
end

function build_random_game(size::Int; instance_index::Int, n_inits::Int = 10)
    size >= 2 || throw(ArgumentError("Random game size must be >= 2"))
    seeds = game_named_seeds(instance_index, n_inits)
    rng = Xoshiro(UInt64(seeds.matrix))
    K = 2.0 .* rand(rng, size, size) .- 1.0
    return build_game_problem(K, 40 + instance_index, "random_$(size)";
                              instance_kind = :random,
                              instance_index = instance_index,
                              n_inits = n_inits,
                              matrix_seed = seeds.matrix)
end

function build_duplicated_identity_game(q::Int = GAME_DEGENERATE_Q; n_inits::Int = 10)
    q >= 2 || throw(ArgumentError("q must be >= 2"))
    K = hcat(Matrix{Float64}(I, q, q), Matrix{Float64}(I, q, q)[:, 1])
    return build_game_problem(K, 44, "duplicated_identity_q$(q)";
                              instance_kind = :duplicated_identity,
                              instance_index = 4,
                              n_inits = n_inits,
                              matrix_seed = nothing,
                              q = q)
end

function build_game_instance(name::AbstractString; n_inits::Int = 10)
    for (index, size) in enumerate(GAME_RANDOM_SIZES)
        name == "random_$(size)" &&
            return build_random_game(size; instance_index = index, n_inits = n_inits)
    end
    name == "duplicated_identity_q$(GAME_DEGENERATE_Q)" &&
        return build_duplicated_identity_game(GAME_DEGENERATE_Q; n_inits = n_inits)
    throw(ArgumentError("Unknown game instance '$name'."))
end

function game_duality_gap(prob::TestProblem, u::AbstractVector{<:Real})
    K = prob.metadata.K
    m, n = size(K)
    # Several splitting methods expose an unconstrained principal iterate.
    # The matrix-game duality gap is defined on Delta_n x Delta_m, so evaluate
    # it at the canonical feasible shadow P_C(u).
    feasible = project_product_simplex(u, n)
    x = @view feasible[1:n]
    y = @view feasible[n+1:n+m]
    return maximum(K * x) - minimum(K' * y)
end

function degenerate_solution_projection(u0::AbstractVector{<:Real}, q::Int)
    length(u0) == 2q + 1 || throw(DimensionMismatch("Expected $(2q + 1) coordinates."))
    c = 1.0 / q
    a = clamp((u0[1] - u0[q + 1] + c) / 2, 0.0, c)
    b = c - a
    return vcat(a, fill(c, q - 1), b, fill(c, q))
end

function degenerate_solution(q::Int, split::Real)
    c = 1.0 / q
    a = Float64(split)
    0.0 <= a <= c || throw(ArgumentError("split must lie in [0, 1/q]"))
    return vcat(a, fill(c, q - 1), c - a, fill(c, q))
end

"Exact maximum projection variational-inequality residual over the saddle segment."
function degenerate_projection_certificate(u0::AbstractVector{<:Real}, ubar::AbstractVector{<:Real}, q::Int)
    direction = Float64.(u0 .- ubar)
    endpoint_0 = degenerate_solution(q, 0.0)
    endpoint_1 = degenerate_solution(q, 1.0 / q)
    return max(dot(direction, endpoint_0 .- ubar),
               dot(direction, endpoint_1 .- ubar))
end
