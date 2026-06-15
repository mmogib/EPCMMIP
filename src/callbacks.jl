# callbacks.jl — Event-dispatched callback system for solvers.
#
# Defines the abstract callback hierarchy, the shared `SolverState` struct,
# and the dispatch contracts used by all five solvers in `algorithm.jl`.
#
# Design (script_port_plan.md §4 / Q3 + post-review revisions):
#
#  - Solvers loop their iterations; at three lifecycle events
#    (`:init`, `:iter`, `:terminate`) they fire two kinds of callbacks:
#
#      * **Stopping callbacks** (`AbstractStoppingCallback`) — return
#        `(halt::Bool, reason::Symbol)`. First-halt wins. Checked once
#        per `:iter` event, in input order.
#
#      * **Observer callbacks** (`AbstractObserverCallback`) — fire
#        `on_event!` for side effects (history collection, progress
#        display, logging, native-residual recording). Fired in input
#        order at every event.
#
#  - **Mutation contract**: the solver OWNS `state::SolverState`. Stoppers
#    and observers are READ-ONLY on `state` and write only to their own
#    fields. The solver mutates `state.k`, `state.residual`, etc.;
#    callbacks read.
#
#  - **Type stability**: stoppers and observers are passed as `Tuple`s
#    (not `Vector`s) so dispatch in the solver hot loop is concrete.
#    Solver signature: `solve_X(prob, x0; stopping::Tuple, observers::Tuple=(), params...)`.
#
#  - **De-bounce**: `ResidualStopping` and `NativeResStopping` require
#    `consec` consecutive iters below threshold (default 2). Avoids
#    false stops from `ρ_n`-jump artefacts in adaptive-step methods.
#
#  - **Two residuals** (per B1 post-review): `state` carries both
#    `residual` (unnormalised `R_n = ||p - J^A_ρ(p - ρ B p)||`) AND
#    `scaled_residual` (`R̃_n = R_n / ρ`). `ResidualStopping` checks
#    one or the other via its `field::Symbol` arg. B1 pilot (Step 9
#    in notes/next_steps.md) decides the project default.
#
#  - **`flag::Symbol` enum** (recorded in DB):
#       `:converged`   — stopping callback halted with this reason.
#       `:maxiter`     — `MaxIterStopping` tripped.
#       `:nan`         — `NanStopping` detected a non-finite residual.
#       `:stagnation`  — reserved for solver internals (no built-in stopper).
#       `:error`       — exception caught at outer benchmark layer.

# ============================================================================
# Abstract types
# ============================================================================

"""
    AbstractCallback

Root abstract type for all callbacks. Direct subtypes:
- `AbstractStoppingCallback` — implements `check_stop`.
- `AbstractObserverCallback` — implements `on_event!`.
"""
abstract type AbstractCallback end

"Stopping callbacks halt the iteration. Implement `check_stop(cb, state)`."
abstract type AbstractStoppingCallback <: AbstractCallback end

"Observer callbacks fire for side effects only. Implement `on_event!(cb, state, event)`."
abstract type AbstractObserverCallback <: AbstractCallback end

# ============================================================================
# SolverState
# ============================================================================

"""
    SolverState

Solver-owned state passed read-only to callbacks each event. Fields:

| Field             | Meaning                                                          |
|-------------------|------------------------------------------------------------------|
| `k`               | Iteration count (0 at `:init`, 1 at first `:iter`).              |
| `x`               | Current principal iterate `p_n` (Vector{Float64}).               |
| `x_prev`          | Previous iterate `p_{n-1}` (empty Vector{Float64} at `:init`).   |
| `residual`        | Unnormalised `R_n = ‖p − J^A_ρ(p − ρ B p)‖` (NaN at `:init`).    |
| `scaled_residual` | Normalised `R̃_n = R_n / ρ_n` (NaN at `:init`).                  |
| `step_size`       | Current step `ρ_n` / `λ_n` (NaN at `:init`).                     |
| `f_evals`         | Cumulative B-evaluations.                                        |
| `elapsed`         | Cumulative wall-clock seconds since solve start.                 |
| `flag`            | Sentinel `:running` while iterating; set to a flag-enum value at `:terminate`. |
| `method`          | Solver tag (e.g. `:EPCM`).                                       |

## Solver responsibility — write order per `:iter` event

The solver MUST follow this order each iteration before invoking the
stopping/observer tuples:
  1. Set `state.x_prev = state.x` (or copy/swap as appropriate).
  2. Update `state.x` to the new iterate `p_{n+1}`.
  3. Update `state.f_evals` and `state.elapsed` cumulatively.
  4. Update `state.step_size` to the method's current `ρ_n`.
  5. **Update BOTH `state.residual` and `state.scaled_residual`** —
     `NanStopping` short-circuits otherwise, and `ResidualStopping` would
     read stale values.
  6. Increment `state.k`.
  7. Fire observers (`:iter` event).
  8. Invoke the stopping tuple via `check_stop`; first-halt wins, sets
     `state.flag`, breaks the loop.
  9. At `:terminate`, observers fire one final time with the final state.
"""
mutable struct SolverState
    k::Int
    x::Vector{Float64}
    x_prev::Vector{Float64}
    residual::Float64
    scaled_residual::Float64
    step_size::Float64
    f_evals::Int
    elapsed::Float64
    flag::Symbol
    method::Symbol
end

"""
    SolverState(method::Symbol, x0::Vector{Float64})

Construct a fresh state at the `:init` event. `x_prev` is empty (solver
fills on first `:iter`); all residuals / step start at NaN; `flag` is
`:running`.
"""
function SolverState(method::Symbol, x0::Vector{Float64})
    return SolverState(
        0,                  # k
        copy(x0),           # x
        Float64[],          # x_prev (empty until iter 1)
        NaN,                # residual
        NaN,                # scaled_residual
        NaN,                # step_size
        0,                  # f_evals
        0.0,                # elapsed
        :running,           # flag
        method,             # method
    )
end

# ============================================================================
# Dispatch contracts (no-op defaults)
# ============================================================================

"""
    check_stop(cb::AbstractStoppingCallback, state::SolverState) -> (halt::Bool, reason::Symbol)

Stopping check, invoked by the solver in input order across a `Tuple`
of stoppers at every `:iter` event, AFTER the solver has updated all
state fields for the current iteration (see `SolverState` docstring,
"Solver responsibility"). Returns `(true, reason)` to halt and set
`state.flag = reason`, or `(false, :none)` to continue. **First-halt
wins** — once any stopper returns `(true, _)` the solver breaks the
loop and skips the remaining stoppers. Default implementation: never halt.
"""
check_stop(::AbstractStoppingCallback, ::SolverState) = (false, :none)

"""
    on_event!(cb::AbstractObserverCallback, state::SolverState, event::Symbol) -> Nothing

Observer fire. Invoked by the solver in input order across a `Tuple`
of observers at every event. `event ∈ {:init, :iter, :terminate}`.
Side effects only; the observer must NOT mutate `state` (read any field;
write only to its own fields). Default implementation: no-op.
"""
on_event!(::AbstractObserverCallback, ::SolverState, ::Symbol) = nothing

# ============================================================================
# Built-in stopping callbacks
# ============================================================================

"""
    MaxIterStopping(maxiter::Int)

Halt when `state.k >= maxiter`. Immutable; no internal counter needed.
"""
struct MaxIterStopping <: AbstractStoppingCallback
    maxiter::Int
end

function check_stop(cb::MaxIterStopping, state::SolverState)
    return state.k >= cb.maxiter ? (true, :maxiter) : (false, :none)
end

"""
    ResidualStopping(tol::Real; field::Symbol=:residual, consec::Int=2)

Halt with reason `:converged` when `state.<field> < tol` (strictly less
than) for `consec` consecutive `:iter` events. `field` is one of
`:residual` (unnormalised `R_n`) or `:scaled_residual` (`R̃_n`).
De-bounce protects against `ρ_n`-jump artefacts in adaptive-step solvers.

**Default `field = :residual`** (the unnormalised R_n) per the B1 pilot
decision — see `notes/plan_review_findings.md`. The unscaled rule matches
the manuscript's §Stopping criteria definition and produces a threshold
1e-6 that is achievable in principle and roughly consistent with the
source-paper native thresholds for our P1/P2/P3 conditioning. Pass
`field=:scaled_residual` to halt on the step-size-invariant
R̃_n = R_n / ρ_n instead.
"""
mutable struct ResidualStopping <: AbstractStoppingCallback
    tol::Float64
    field::Symbol
    consec::Int
    below_count::Int

    function ResidualStopping(tol::Real; field::Symbol=:residual, consec::Int=2)
        field in (:residual, :scaled_residual) ||
            throw(ArgumentError("ResidualStopping field must be :residual or :scaled_residual, got :$field"))
        consec >= 1 || throw(ArgumentError("ResidualStopping consec must be >= 1, got $consec"))
        tol > 0 || throw(ArgumentError("ResidualStopping tol must be > 0, got $tol"))
        return new(convert(Float64, tol), field, consec, 0)
    end
end

function check_stop(cb::ResidualStopping, state::SolverState)
    val = getproperty(state, cb.field)
    if isfinite(val) && val < cb.tol
        cb.below_count += 1
        cb.below_count >= cb.consec && return (true, :converged)
    else
        cb.below_count = 0
    end
    return (false, :none)
end

"""
    NativeResStopping(native_fn, tol::Real; consec::Int=2)

Halt with reason `:converged` when `native_fn(state.x, state.x_prev) < tol`
for `consec` consecutive iters. Intended for ablation runs and
`s28_validate_baselines.jl` where the stopping rule is each source paper's
native quantity (P1 `‖x_n − x*‖`, P2 `‖x_{n+1} − x_n‖`, P3 `0.5‖z − J^A(z − Bz)‖²`).

The standard project stopping rule is `ResidualStopping`; use this only
when reproducing source-paper protocols.
"""
mutable struct NativeResStopping{F} <: AbstractStoppingCallback
    native_fn::F
    tol::Float64
    consec::Int
    below_count::Int

    function NativeResStopping{F}(native_fn::F, tol::Float64; consec::Int=2) where {F}
        consec >= 1 || throw(ArgumentError("NativeResStopping consec must be >= 1, got $consec"))
        tol > 0 || throw(ArgumentError("NativeResStopping tol must be > 0, got $tol"))
        return new{F}(native_fn, tol, consec, 0)
    end
end

NativeResStopping(native_fn::F, tol::Real; consec::Int=2) where {F} =
    NativeResStopping{F}(native_fn, convert(Float64, tol); consec=consec)

function check_stop(cb::NativeResStopping, state::SolverState)
    val = cb.native_fn(state.x, state.x_prev)
    if isfinite(val) && val < cb.tol
        cb.below_count += 1
        cb.below_count >= cb.consec && return (true, :converged)
    else
        cb.below_count = 0
    end
    return (false, :none)
end

"""
    NanStopping()

Halt with reason `:nan` if either `state.residual` or `state.scaled_residual`
is non-finite — covers both `NaN` and `±Inf`. Overflow on P2 with B(x)=2x+b
or on adaptive-step methods that overshoot can produce `Inf` residuals; this
stopper catches both failure modes. The reason `:nan` is retained for
historical reasons even though `Inf` triggers it too.

Cheap; include in every stopping tuple to catch divergence early.
"""
struct NanStopping <: AbstractStoppingCallback end

function check_stop(::NanStopping, state::SolverState)
    return (!isfinite(state.residual) || !isfinite(state.scaled_residual)) ?
           (true, :nan) : (false, :none)
end

# ============================================================================
# Built-in observer callbacks
# ============================================================================

"""
    HistoryCallback()

Captures per-iteration data into `cb.history::Vector{IterRecord}`. Fires
on `:iter` only. Include in `observers` to enable history collection;
omit in OAT/LHS tuning runs (s10/s20) where history is wasted.
"""
mutable struct HistoryCallback <: AbstractObserverCallback
    history::Vector{IterRecord}
end

HistoryCallback() = HistoryCallback(IterRecord[])

function on_event!(cb::HistoryCallback, state::SolverState, event::Symbol)
    if event === :iter
        push!(cb.history, IterRecord(
            state.k,
            state.f_evals,
            state.elapsed,
            state.residual,
            state.scaled_residual,
            state.step_size,
        ))
    end
    return nothing
end

"""
    ProgressCallback(prog; method="", prob_name="", init_label="", every=100)

Updates an outer `ProgressMeter.Progress` bar's `showvalues` from inside
each solve. Does NOT advance the bar's position — the outer benchmark loop
does that after each completed solve.

The bar's redraw is throttled internally by `ProgressMeter`'s `dt` knob, but
the `update!` call itself allocates a fresh `Vector{Tuple}` for `showvalues`
on each invocation. We additionally gate via `every::Int=100` so cheap
problems (e.g. P1 @ m=5 with thousands of iterations/sec) aren't dominated
by this allocation. Bump `every` lower (e.g. 1) for live diagnostics on
expensive problems where each iteration takes ≥ 10 ms.

**Note**: the implementation reads `cb.prog.counter` to pass the *current*
bar position back to `ProgressMeter.update!` (intentional — keeps the
position unchanged while refreshing `showvalues`). This is a documented
dependency on a `ProgressMeter` internal field. If a future ProgressMeter
release renames it, update both the field access and `Project.toml`'s
compat bound on `ProgressMeter`.

`prog` is held by reference; one instance typically follows one outer
work-item bar across many solver calls (mutate `method`/`prob_name`/
`init_label` between solves).
"""
mutable struct ProgressCallback{P} <: AbstractObserverCallback
    prog::P
    method::String
    prob_name::String
    init_label::String
    every::Int
end

function ProgressCallback(prog::P;
        method::String = "",
        prob_name::String = "",
        init_label::String = "",
        every::Int = 100,
    ) where {P}
    every >= 1 || throw(ArgumentError("ProgressCallback every must be >= 1, got $every"))
    return ProgressCallback{P}(prog, method, prob_name, init_label, every)
end

function on_event!(cb::ProgressCallback, state::SolverState, event::Symbol)
    if event === :iter && (state.k == 1 || state.k % cb.every == 0)
        ProgressMeter.update!(cb.prog, cb.prog.counter;
            showvalues = [
                ("method", cb.method),
                ("prob",   cb.prob_name),
                ("init",   cb.init_label),
                ("k",      state.k),
                ("Rn",     @sprintf("%.3e", state.residual)),
                ("Rt_n",   @sprintf("%.3e", state.scaled_residual)),
                ("rho_n",  @sprintf("%.3e", state.step_size)),
            ])
    end
    return nothing
end

"""
    LogTableCallback(io::IO=stdout; every::Int=100)

Prints a one-line table row every `every` iterations, plus a header at
`:init` and a final-state line at `:terminate`. Columns: `k`, `R_n`,
`R̃_n`, `ρ_n`, `elapsed (s)`. Throttled by `every` to keep `println`
overhead from dominating CPU time on cheap problems (P1@m=5 would
otherwise be dominated by I/O).
"""
struct LogTableCallback <: AbstractObserverCallback
    io::IO
    every::Int

    function LogTableCallback(io::IO=stdout; every::Int=100)
        every >= 1 || throw(ArgumentError("LogTableCallback every must be >= 1, got $every"))
        return new(io, every)
    end
end

function on_event!(cb::LogTableCallback, state::SolverState, event::Symbol)
    if event === :init
        @printf(cb.io, "  %5s  %12s  %12s  %12s  %10s\n",
                "k", "R_n", "Rt_n", "rho_n", "elapsed")
        @printf(cb.io, "  %s\n", "-" ^ 62)
    elseif event === :iter && (state.k == 1 || state.k % cb.every == 0)
        @printf(cb.io, "  %5d  %12.4e  %12.4e  %12.4e  %10.3f\n",
                state.k, state.residual, state.scaled_residual,
                state.step_size, state.elapsed)
    elseif event === :terminate
        @printf(cb.io, "  %s\n", "-" ^ 62)
        @printf(cb.io, "  %5s  %12.4e  %12.4e  %12.4e  %10.3f  [%s]\n",
                "fin", state.residual, state.scaled_residual,
                state.step_size, state.elapsed, state.flag)
    end
    return nothing
end

"""
    NativeResRecorder(native_fn)

Records `native_fn(state.x, state.x_prev)` at `:terminate`. After the
solver returns, the benchmark wrapper reads `cb.value` to get the final
native quantity (P1's `‖x − x*‖`, P2's `‖x − x_prev‖`, P3's `Tol_n`)
for the manuscript's per-paper reporting column.

The solver's own stopping check uses `ResidualStopping` (universal
residual); this observer is purely for reporting.
"""
mutable struct NativeResRecorder{F} <: AbstractObserverCallback
    native_fn::F
    value::Float64
end

NativeResRecorder(native_fn::F) where {F} = NativeResRecorder{F}(native_fn, NaN)

function on_event!(cb::NativeResRecorder, state::SolverState, event::Symbol)
    if event === :terminate
        cb.value = cb.native_fn(state.x, state.x_prev)
    end
    return nothing
end
