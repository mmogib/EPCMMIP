# ============================================================================
# s00_problem_smoke.jl
# ============================================================================
#
# Purpose
#   Lightweight smoke test for the sparse logistic problem definition.
#
# What it does
#   - builds `SparseLogistic_L1` via `get_sparse_logistic_problem`
#   - checks dimension/metadata consistency
#   - evaluates the forward operator `B(x0)`
#   - evaluates the resolvent `J^A_lambda(x0)` at lambda = 1
#   - evaluates the native residual at the initial point
#
# How to run
#   julia --project=. scripts/sparse_logistic/s00_problem_smoke.jl
#
# ============================================================================

include(joinpath(@__DIR__, "..", "..", "src", "includes.jl"))

const SMOKE_DIM = 128
const SMOKE_NSEEDS = 2
const SMOKE_RHO_SCALE = 5.0e-3
const SMOKE_LAMBDA = 1.0

function _check(condition::Bool, msg::AbstractString)
    condition || error(msg)
    return nothing
end

function smoke_main(; dim::Int = SMOKE_DIM, n_seeds::Int = SMOKE_NSEEDS,
                    rho_scale::Float64 = SMOKE_RHO_SCALE, lambda::Float64 = SMOKE_LAMBDA)
    println("==============================================================")
    println("Sparse Logistic Problem Smoke Test")
    println("==============================================================")
    println("project root : ", JCODE_ROOT)
    println("script path  : ", @__FILE__)
    println("dim          : ", dim)
    println("n_seeds      : ", n_seeds)
    println("rho_scale    : ", rho_scale)
    println("lambda       : ", lambda)
    println()

    prob = get_sparse_logistic_problem(dim = dim, n_seeds = n_seeds, rho_scale = rho_scale)
    x0 = copy(first(prob.initial_points).x0)

    println("Problem summary")
    println("  name        : ", prob.name)
    println("  dim         : ", prob.dim)
    println("  init points : ", length(prob.initial_points))
    println("  metadata.N  : ", prob.metadata.N)
    println("  metadata.m  : ", prob.metadata.m)
    println("  metadata.rho: ", prob.metadata.rho)
    println("  metadata.L  : ", prob.metadata.L)
    println()

    _check(prob.name == "SparseLogistic_L1", "unexpected problem name: $(prob.name)")
    _check(prob.dim == dim, "problem dim mismatch: expected $dim, got $(prob.dim)")
    _check(length(prob.initial_points) == n_seeds, "initial-point count mismatch")
    _check(length(x0) == dim, "x0 length mismatch")
    _check(size(prob.metadata.X, 1) == prob.metadata.N, "metadata.X row count mismatch")
    _check(size(prob.metadata.X, 2) == prob.metadata.m, "metadata.X column count mismatch")
    _check(size(prob.metadata.K) == size(prob.metadata.X), "metadata.K size mismatch")
    _check(length(prob.metadata.b) == prob.metadata.N, "metadata.b length mismatch")
    _check(isfinite(prob.metadata.rho) && prob.metadata.rho > 0.0, "rho must be finite and positive")
    _check(isfinite(prob.metadata.L) && prob.metadata.L > 0.0, "L must be finite and positive")

    Bx0 = prob.B(x0)
    prox_x0 = prob.resolvent_A(x0, lambda)
    native0 = prob.native_residual(x0, x0)
    fixed_point_residual = norm(x0 .- prob.resolvent_A(x0 .- lambda .* Bx0, lambda))

    println("Operator checks")
    println("  ||x0||              : ", norm(x0))
    println("  ||B(x0)||           : ", norm(Bx0))
    println("  ||J^A_lambda(x0)||  : ", norm(prox_x0))
    println("  native residual     : ", native0)
    println("  fixed-point residual: ", fixed_point_residual)
    println()

    _check(length(Bx0) == dim, "B(x0) length mismatch")
    _check(length(prox_x0) == dim, "resolvent output length mismatch")
    _check(all(isfinite, Bx0), "B(x0) contains non-finite values")
    _check(all(isfinite, prox_x0), "resolvent output contains non-finite values")
    _check(isfinite(native0) && native0 >= 0.0, "native residual must be finite and nonnegative")
    _check(isfinite(fixed_point_residual) && fixed_point_residual >= 0.0,
           "fixed-point residual must be finite and nonnegative")

    println("Smoke test status: PASS")
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    smoke_main()
end
