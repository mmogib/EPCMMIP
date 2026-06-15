# types.jl — Core types for solver results and history tracking.
#
# Defines the data structures that ALL solvers return and that the DB layer
# reads and writes. Keep types.jl and benchmark.jl in sync: adding a field
# here means adding the corresponding DB column too.

# ============================================================================
# Iteration Record (per-iteration history)
# ============================================================================

"""
    IterRecord

One row of per-iteration convergence data. Pushed into `HistoryCallback.history`
when a `HistoryCallback` is in the solver's observers tuple, then bulk-inserted
into the `history` DB table after the solve completes.

Carries BOTH residual forms per the B1 post-review decision: the stopping rule
(`ResidualStopping`) picks one; the figure scripts can show either or both.
"""
struct IterRecord
    k::Int                      # iteration number (1-based, fired at :iter)
    f_evals::Int                # cumulative B-evaluations
    elapsed::Float64            # cumulative wall-clock seconds since solve start

    residual::Float64           # R_n  = ||p_n - J^A_{ρ_n}(p_n - ρ_n B p_n)||
    scaled_residual::Float64    # R̃_n = R_n / ρ_n  (natural-residual limit as ρ_n → 0)
    step_size::Float64          # ρ_n / λ_n / τ_n  (method's current step)
end

# ============================================================================
# Solver Result
# ============================================================================

"""
    SolverResult

Returned by every solver. The benchmark wrapper consumes this plus, optionally,
the `value` field of a `NativeResRecorder` observer to obtain the native
problem-specific residual for reporting.

## Solver contract (see jcode/CLAUDE.md + script_port_plan.md §5.1)
Every `solve_X` in `algorithm.jl`:
  1. Returns a `SolverResult`.
  2. Signature `solve_X(prob, x0; stopping::Tuple, observers::Tuple=(), params...)`.
  3. Declares `const X_VERSION` and `const X_PARAMS_BY_PROBLEM` (no Function values).
  4. Records both `residual` and `scaled_residual` per iteration (state fields
     + IterRecord fields). The stopping rule chooses one — driven by B1 pilot.
  5. There is NO `track::Bool` kwarg. History is collected iff a
     `HistoryCallback` is present in `observers`.
"""
struct SolverResult
    # ── Common fields ────────────────────────────────────────────────────
    converged::Bool
    iterations::Int
    f_evals::Int                # B-evaluations (single-valued operator)
    cpu_time::Float64
    x::Vector{Float64}          # final iterate
    flag::Symbol                # one of :converged, :maxiter, :nan, :stagnation, :error
    history::Vector{IterRecord} # empty unless a HistoryCallback was used

    # ── Residuals at termination ─────────────────────────────────────────
    residual::Float64           # R_n at termination
    scaled_residual::Float64    # R̃_n at termination
end

# ============================================================================
# Constructors
# ============================================================================

"""
    make_result(; converged, iterations, f_evals, cpu_time, x, flag,
                  history=IterRecord[], residual=NaN, scaled_residual=NaN)

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
        scaled_residual::Float64 = NaN,
    )
    return SolverResult(
        converged, iterations, f_evals, cpu_time, x, flag, history,
        residual, scaled_residual,
    )
end

# ============================================================================
# Solver registry — see algorithm_types.jl
# ============================================================================
#
# Each solver is a concrete subtype of `AbstractAlgorithm` defined in
# `algorithm_types.jl` (Step 4.5). That file declares:
#
#   - The struct (parameters as fields, `Float64` / `Symbol` only).
#   - `<ALG>_PRESETS::Dict{Symbol,NamedTuple}` for `:P1, :P2, :P3` (per-problem
#     tuned defaults, overwritten by `s25_promote_best_params.jl` after LHS)
#     and `:paper` (source-paper defaults; used by `s28_validate_baselines.jl`).
#   - `name(::Type{T})::String` and `version(::Type{T})::VersionNumber`.
#
# `algorithm.jl` (Steps 7, 10) holds only the `solve` method bodies. The hash
# `make_config_hash(alg::AbstractAlgorithm, prob_id, eps, maxiter)` pulls
# `name(typeof(alg))`, `string(version(typeof(alg)))`, and per-field values
# via `fieldnames` + `getfield`. Version bumps OR field-value changes
# produce a different hash → fresh DB rows; old runs preserve historical state.
