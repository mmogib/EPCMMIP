# types.jl — Core types for solver results and history tracking
#
# Defines the data structures that ALL solvers return and that the DB layer
# reads and writes. Keep types.jl and benchmark.jl in sync: if you add a field
# to SolverResult, add the corresponding DB column too.

# ============================================================================
# Iteration Record (for per-iteration history tracking)
# ============================================================================

"""
    IterRecord

One row of per-iteration convergence data. Stored in `SolverResult.history`
and bulk-inserted into the `history` DB table when `track=true`.
"""
struct IterRecord
    # ── Common fields ────────────────────────────────────────────────────
    k::Int                      # iteration number
    f_evals::Int                # cumulative function evaluations of B
    elapsed::Float64            # cumulative wall-clock time (seconds)

    # ── Inclusion-problem tracking fields ────────────────────────────────
    residual::Float64           # problem-specific residual: ||x_n - x*||,
                                # ||x_{n+1} - x_n||, or 0.5||w_n - J^A(w_n - B w_n)||^2
    step_size::Float64          # lambda_n / tau_n (adaptive step size)
end

# ============================================================================
# Solver Result
# ============================================================================

"""
    SolverResult

Returned by every solver. Common fields are required; project-specific fields
are tailored to the monotone-inclusion comparison framework.

## Solver contract (see jcode/CLAUDE.md)
Every solver function must:
  1. Return a `SolverResult`
  2. Accept `track::Bool=false` keyword (enable per-iteration history)
  3. Accept `callback=nothing` keyword (live progress updates)
     Callback signature: `callback(k::Int, metric::Float64, maxiter::Int)`
"""
struct SolverResult
    # ── Common fields (do not remove) ────────────────────────────────────
    converged::Bool
    iterations::Int
    f_evals::Int                # B-evaluations (single-valued operator)
    cpu_time::Float64
    x::Vector{Float64}          # final iterate
    flag::Symbol                # :converged, :maxiter, :linesearch_failed, :error, ...
    history::Vector{IterRecord} # empty if track=false

    # ── Inclusion-problem fields ─────────────────────────────────────────
    residual::Float64           # final residual (problem-dependent stopping quantity)
end

# ============================================================================
# Constructors
# ============================================================================

"""
    make_result(; converged, iterations, f_evals, cpu_time, x, flag, history=[], residual=NaN)

Keyword constructor for `SolverResult`. Use in error/catch paths to create a
failed result without running the solver:

    catch ex
        result = make_result(converged=false, iterations=0, f_evals=0,
                             cpu_time=0.0, x=copy(x0), flag=:error)
    end
"""
function make_result(;
        converged::Bool,
        iterations::Int,
        f_evals::Int,
        cpu_time::Float64,
        x::Vector{Float64},
        flag::Symbol,
        history::Vector{IterRecord} = IterRecord[],
        residual::Float64 = NaN,
    )
    return SolverResult(
        converged, iterations, f_evals, cpu_time, x, flag, history,
        residual,
    )
end

# ============================================================================
# Solver Version Constants (declare in each solver file)
# ============================================================================
#
# Every solver file must declare these two constants:
#
#   const {SOLVER}_VERSION  = "1.0.0"
#   const {SOLVER}_DEFAULTS = (param1=val1, param2=val2, ...)
#
# VERSION follows semver: bug fix → PATCH, logic change → MINOR.
# DEFAULTS is the NamedTuple that is both hashed AND splatted to the solver.
# Changing defaults does NOT require a version bump (params are in the hash).
#
# Example (to be filled in after discussion):
#   const EPCM_VERSION  = "1.0.0"
#   const EPCM_DEFAULTS = (lambda0=1.0, mu=0.5, alpha_n = n -> 1.0/(n+1))
