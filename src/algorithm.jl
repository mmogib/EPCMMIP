# algorithm.jl — `solve` method bodies for each AbstractAlgorithm subtype.
#
# Currently contains:
#   - solve(::EPCM, prob, x0; stopping, observers)  (Step 7)
#
# Step 10 will add solve methods for MTTM, IMTTM, IFRAB, SFRBM.
#
# Every body honours the 9-step "Solver responsibility" contract documented
# in `callbacks.jl`'s `SolverState` docstring. In particular:
#   - state.x_prev is updated BEFORE state.x is overwritten.
#   - state.residual and state.scaled_residual are updated BEFORE invoking
#     the stopping tuple (otherwise NanStopping short-circuits or
#     ResidualStopping reads stale values).

# ============================================================================
# Rule dispatchers — Symbol tags → numeric values per iteration
# ============================================================================
# Each algorithm's struct has Symbol-typed rule fields (e.g. delta_rule,
# sigma_rule). The solver resolves these to Float64 values at each iter via
# the helpers below. Adding a new rule = add a new Symbol case here.

"""
    _epcm_delta(rule::Symbol, exp::Float64, k::Int) -> Float64

EPCM's δ_k step-growth budget (parameter restriction: ∑ δ_k < ∞ ⟹ τ_k bounded;
Step 11-2 parametrization). Rules:
- `:power` → δ_k = 1/(k+1)^`exp` (admissible iff `exp > 1`; default `exp=2`
  reproduces the manuscript's 1/(k+1)², headroom Π(1+δ_k) ≈ 3.68×).
- `:zero`  → δ_k ≡ 0 (monotone-shrink baseline; `exp` ignored).
The `exp > 1` admissibility is validated once at `solve` entry, not here.
"""
@inline function _epcm_delta(rule::Symbol, exp::Float64, k::Int)
    rule === :power && return 1.0 / (k + 1.0)^exp
    rule === :zero  && return 0.0
    throw(ArgumentError("Unknown EPCM delta_rule :$rule; known: :power, :zero"))
end

"""
    _epcm_sigma(rule::Symbol, scale::Float64, exp::Float64, k::Int) -> Float64

EPCM's σ_k Halpern-anchoring weight (parameter restriction: σ_k∈(0,1), σ_k→0,
∑ σ_k = ∞; Step 11-2 parametrization). Rules:
- `:power` → σ_k = `scale` / (k+2)^`exp`. Admissible iff `exp ∈ (0,1]` (so
  ∑ diverges) and σ_0 = `scale`/2^`exp` < 1. Default `scale=1, exp=1`
  reproduces the manuscript's 1/(k+2) (Lieder-optimal Halpern weight).
The admissibility is validated once at `solve` entry, not here.
"""
@inline function _epcm_sigma(rule::Symbol, scale::Float64, exp::Float64, k::Int)
    rule === :power && return scale / (k + 2.0)^exp
    throw(ArgumentError("Unknown EPCM sigma_rule :$rule; known: :power"))
end

"""
    _mttm_alpha(rule::Symbol, n::Int) -> Float64

MTTM's α_n sequence (Gibali2018 Condition 3: α_n → 0, ∑ α_n = ∞). Index `n`
is 1-based (`n = state.k + 1`), matching Gibali2018's numerical choice
α_k = 1/k (paper page 19). Rules:
- `:inv_n` → α_n = 1/n (manuscript §Parameter choices line 1607)

At n = 1, α_1 = 1 and β_1 = 0, so the first MTTM update is
x_{next} = (1−α_1−β_1)x_0 + β_1 z = 0. This degenerate first step is faithful
to Gibali2018 (the convexity weight 1−α_1−β_1 = 0 stays valid). On Problem 1
(x* = 0) it makes MTTM reach the solution on its first computed iterate — a
parameter artifact, not an algorithmic advantage; noted for s28/s30 reporting.
"""
@inline function _mttm_alpha(rule::Symbol, n::Int)
    rule === :inv_n && return 1.0 / n
    throw(ArgumentError("Unknown MTTM alpha_rule :$rule; known: :inv_n"))
end

"""
    _mttm_beta(rule::Symbol, n::Int) -> Float64

MTTM's β_n sequence (Gibali2018 Condition 3: {β_n} ⊂ (a, 1−α_n)). Index `n`
is 1-based. Rules:
- `:n_minus_1_over_2n` → β_n = (n−1)/(2n) (manuscript §Parameter choices line 1607)
"""
@inline function _mttm_beta(rule::Symbol, n::Int)
    rule === :n_minus_1_over_2n && return (n - 1) / (2n)
    throw(ArgumentError("Unknown MTTM beta_rule :$rule; known: :n_minus_1_over_2n"))
end

"""
    _imttm_alpha(rule::Symbol, n::Int) -> Float64

IMTTM's α_n sequence (Tan2022a (C1): α_n ∈ (0,1), α_n → 0, ∑ α_n = ∞). Index
`n` is 1-based (`n = state.k + 1`). Rules:
- `:inv_n_plus_1` → α_n = 1/(n+1) (manuscript §Parameter choices line 1607)
"""
@inline function _imttm_alpha(rule::Symbol, n::Int)
    rule === :inv_n_plus_1 && return 1.0 / (n + 1)
    throw(ArgumentError("Unknown IMTTM alpha_rule :$rule; known: :inv_n_plus_1"))
end

"""
    _imttm_beta(rule::Symbol, α_n::Float64) -> Float64

IMTTM's β_n sequence (Tan2022a (C5): {β_n} ⊂ (a, 1−α_n)). Depends on the
already-computed α_n. Rules:
- `:tan` → β_n = 0.5·(1 − α_n) (manuscript §Parameter choices line 1607)
"""
@inline function _imttm_beta(rule::Symbol, α_n::Float64)
    rule === :tan && return 0.5 * (1 - α_n)
    throw(ArgumentError("Unknown IMTTM beta_rule :$rule; known: :tan"))
end

"""
    _imttm_epsilon(rule::Symbol, n::Int) -> Float64

IMTTM's ε_n sequence — the inertial summability cap (Tan2022a Remark 3.1(i):
ε_n/α_n → 0, hence θ_n‖x_n − x_{n−1}‖/α_n → 0). Index `n` is 1-based. Rules:
- `:hundred_inv_sq` → ε_n = 100/(n+1)² (manuscript §Parameter choices line 1607)
"""
@inline function _imttm_epsilon(rule::Symbol, n::Int)
    rule === :hundred_inv_sq && return 100.0 / (n + 1)^2
    throw(ArgumentError("Unknown IMTTM epsilon_rule :$rule; known: :hundred_inv_sq"))
end

"""
    _sfrbm_beta(rule::Symbol, k::Int) -> Float64

SFRBM's β_k Halpern-anchoring sequence (Yao2024 Alg 2: β_k ∈ (0,1), β_k → 0,
∑ β_k = ∞). Index `k` is 0-based (`k = state.k`), matching Yao's k = 0, 1, 2, …
(NB: 0-based, unlike MTTM/IMTTM's 1-based n). Rules:
- `:yao` → β_k = 1/(5000(k+1)) (manuscript §Parameter choices line 1623)
"""
@inline function _sfrbm_beta(rule::Symbol, k::Int)
    rule === :yao && return 1.0 / (5000 * (k + 1))
    throw(ArgumentError("Unknown SFRBM beta_rule :$rule; known: :yao"))
end

"""
    _ifrab_sigma(rule::Symbol, n::Int) -> Float64

IFRAB's σ_n anchoring sequence (Izuchukwu2023 Thm 4.4: σ_n ∈ (0,1), σ_n → 0,
∑ σ_n = ∞). Index `n` is 1-based (`n = state.k + 1`), matching Algorithm 4.5's
n ≥ 1. Rules:
- `:izuchukwu_sigma` → σ_n = 0.005/(3n + 25000) (manuscript §Parameter choices line 1614)
"""
@inline function _ifrab_sigma(rule::Symbol, n::Int)
    rule === :izuchukwu_sigma && return 0.005 / (3n + 25000)
    throw(ArgumentError("Unknown IFRAB sigma_rule :$rule; known: :izuchukwu_sigma"))
end

"""
    _ifrab_c(rule::Symbol, n::Int) -> Float64

IFRAB's c_n stepsize-growth budget (Izuchukwu2023 Alg 4.5: c_n ∈ [0,∞),
∑ c_n < ∞, so the non-monotone step δ_{n+1} ≤ δ_n + c_n stays bounded;
Remark 3.2). Index `n` is 1-based. Rules:
- `:inv_n_sq_plus_1` → c_n = 1/(n² + 1) (manuscript §Parameter choices line 1615)
"""
@inline function _ifrab_c(rule::Symbol, n::Int)
    rule === :inv_n_sq_plus_1 && return 1.0 / (n^2 + 1)
    throw(ArgumentError("Unknown IFRAB c_rule :$rule; known: :inv_n_sq_plus_1"))
end

"""
    _ifrab_vartheta(rule::Symbol, vartheta_bar::Float64, n::Int) -> Float64

IFRAB's variable inertial parameter ϑ_n (Izuchukwu2023 Thm 4.4 requires
0 ≤ ϑ_n ≤ ϑ_{n+1} ≤ ϑ̄). Index `n` is 1-based. Rules:
- `:n_over_n_plus_1` → ϑ_n = ϑ̄ · n/(n+1) (manuscript §Parameter choices line 1619; B2 decision)

The n/(n+1) form is monotonically increasing in n and bounded above by ϑ̄,
satisfying the nesting 0 ≤ ϑ_n ≤ ϑ_{n+1} ≤ ϑ̄ without a tuned cutoff.
"""
@inline function _ifrab_vartheta(rule::Symbol, vartheta_bar::Float64, n::Int)
    rule === :n_over_n_plus_1 && return vartheta_bar * n / (n + 1)
    throw(ArgumentError("Unknown IFRAB vartheta_rule :$rule; known: :n_over_n_plus_1"))
end

# ============================================================================
# EPCM — Extrapolation–Projection–Contraction (Algorithm 1 of the manuscript)
# ============================================================================
#
# Iteration (k = 0, 1, 2, …, manuscript Algorithm 1 lines 533-605):
#
#   Step 1 (Extrapolation):       w_k = σ_k x_0 + (1−σ_k) x_k
#   Step 2 (Resolvent):           z_k = J^A_{τ_k}(w_k − τ_k B(z_{k−1}))
#   Step 3 (Forward correction):  v_k = z_k + τ_k (B(z_{k−1}) − B(z_k))
#   Step 4 (Direction):           u_k = w_k − v_k                    [= w_k − z_k − τ_k(Bz_{k−1}−Bz_k)]
#   Step 5 (Coefficient θ_k):     ⟨v_k − z_k, u_k⟩ / ‖u_k‖²  if u_k ≠ 0, else 0
#   Step 6 (Iterate update):      x_{k+1} = v_k − ζ θ_k u_k
#   Step 7 (Stepsize update):     case split on ‖Bz_{k−1} − Bz_k‖ vs ϑ_0 τ_k ‖z_{k−1} − z_k‖
#
# Initial: x_0 given by the caller; **z_{-1} = x_0** by convention (Issue 2).
# Per-iter cost: 1 new B-eval for z_k (B(z_{k-1}) cached) + 1 extra for the
# universal residual at x_{k+1}. Steady-state 2 B-evals/iter.
#
# μ note (Issue 1, refined): the proof-side constant μ ∈ (1, 2) does NOT appear
# in the algorithm body NOR on the EPCM struct. Its existence is guaranteed by
# the strict admissibility constraint ϑ_0 < √(γ/(2(2γ+1))): with strictness,
# any μ ∈ (1, √(γ/(2(2γ+1)))/ϑ_0) satisfies μ·ϑ_0 < √(γ/(2(2γ+1))). The
# convergence proof picks such a μ (manuscript §Convergence, just before
# eq:tau_bound); the iteration output does not depend on which admissible
# μ the proof picks. We validate the strict ϑ_0 bound at solve entry.

"""
    solve(alg::EPCM, prob::TestProblem, x0::Vector{Float64};
          stopping::Tuple, observers::Tuple=()) -> SolverResult

Run the EPCM iteration on `prob` starting from `x0`. Returns a `SolverResult`
populated from the final state plus, if a `HistoryCallback` is in `observers`,
the per-iteration record vector.

Validates the manuscript's parameter constraints (Algorithm 1 input, lines
537-544) before iterating. Throws `ArgumentError` on violation.
"""
function solve(alg::EPCM, prob::TestProblem, x0::Vector{Float64};
               stopping::Tuple,
               observers::Tuple = ())

    # ── Parameter validation (manuscript Algorithm 1 input constraints) ──
    alg.gamma > 0 ||
        throw(ArgumentError("EPCM: γ must be > 0, got $(alg.gamma)"))
    alg.vartheta_0 > 0 ||
        throw(ArgumentError("EPCM: ϑ_0 must be > 0, got $(alg.vartheta_0)"))
    (0 < alg.vartheta_1 < alg.vartheta_0) ||
        throw(ArgumentError("EPCM: require 0 < ϑ_1 < ϑ_0, got ϑ_1=$(alg.vartheta_1), ϑ_0=$(alg.vartheta_0)"))
    bound_theta0 = sqrt(alg.gamma / (2 * (2 * alg.gamma + 1)))
    alg.vartheta_0 < bound_theta0 ||
        throw(ArgumentError("EPCM: strict admissibility — ϑ_0 < √(γ/(2(2γ+1))) = $bound_theta0, " *
                            "got ϑ_0 = $(alg.vartheta_0). Strictness leaves room for an admissible μ ∈ (1, $bound_theta0/$(alg.vartheta_0)) in the convergence proof."))
    (0 < alg.zeta < 2 / (2 + alg.gamma)) ||
        throw(ArgumentError("EPCM: ζ must be in (0, 2/(2+γ)) = (0, $(2/(2+alg.gamma))), got $(alg.zeta)"))
    alg.tau_0 > 0 ||
        throw(ArgumentError("EPCM: τ_0 must be > 0, got $(alg.tau_0)"))

    # σ_k = sigma_scale/(k+2)^sigma_exp must satisfy σ_k ∈ (0,1), σ_k → 0,
    # Σσ_k = ∞ (Algorithm 1 input). Σ diverges iff sigma_exp ≤ 1; σ_k ∈ (0,1)
    # iff σ_0 = sigma_scale/2^sigma_exp < 1 (σ_k is decreasing for sigma_exp > 0).
    if alg.sigma_rule === :power
        (0 < alg.sigma_exp <= 1) ||
            throw(ArgumentError("EPCM: sigma_exp must be in (0, 1] for Σσ_k = ∞, got $(alg.sigma_exp)"))
        alg.sigma_scale > 0 ||
            throw(ArgumentError("EPCM: sigma_scale must be > 0, got $(alg.sigma_scale)"))
        σ0 = alg.sigma_scale / 2.0^alg.sigma_exp
        σ0 < 1 ||
            throw(ArgumentError("EPCM: require σ_0 = sigma_scale/2^sigma_exp < 1 (so σ_k ∈ (0,1)), got σ_0 = $σ0"))
    end
    # δ_k = 1/(k+1)^delta_exp must satisfy δ_k ≥ 0, Σδ_k < ∞ (Algorithm 1 input).
    # Σ converges iff delta_exp > 1 (delta_exp = 1 is the divergent harmonic boundary).
    if alg.delta_rule === :power
        alg.delta_exp > 1 ||
            throw(ArgumentError("EPCM: delta_exp must be > 1 for Σδ_k < ∞, got $(alg.delta_exp)"))
    end

    # ── State setup ──────────────────────────────────────────────────────
    state = SolverState(:EPCM, x0)
    t0    = time()

    # x_0 is the INITIAL iterate (constant across iters, read in Step 1).
    # z_{-1} := x_0 by convention (Issue 2). x_k flows through `state.x`
    # directly — no separate `x_curr` shadow variable.
    x_0     = copy(x0)
    z_prev  = copy(x0)       # z_{k-1}, starts as z_{-1} = x_0

    # Preload B(z_{-1}) — counted as the first f_eval
    Bz_prev = prob.B(z_prev)
    state.f_evals = 1

    # Stepsize
    τ_curr = alg.tau_0       # τ_k, starts as τ_0
    state.step_size = τ_curr

    # Fire :init observers (k=0, state.x = x_0, residuals = NaN, step_size = τ_0)
    for cb in observers
        on_event!(cb, state, :init)
    end

    # ── Main loop ────────────────────────────────────────────────────────
    # On entry to iter k: state.x = x_k (set by Step 6 of the previous iter,
    # or initially x_0). state.x_prev is updated at the top of each iter to
    # snapshot x_k BEFORE state.x is reassigned to x_{k+1}.
    while true
        # Step 1 of the "Solver responsibility" contract: snapshot x_k.
        state.x_prev = copy(state.x)

        # ── Step 1: Extrapolation ────────────────────────────────────────
        # state.x is x_k at this point (not yet updated this iter).
        σ_k = _epcm_sigma(alg.sigma_rule, alg.sigma_scale, alg.sigma_exp, state.k)
        w_k = σ_k .* x_0 .+ (1 - σ_k) .* state.x

        # ── Step 2: Resolvent step (uses cached Bz_prev = B(z_{k-1})) ────
        z_k = prob.resolvent_A(w_k .- τ_curr .* Bz_prev, τ_curr)

        # New B-eval: B(z_k)
        Bz_k = prob.B(z_k)
        state.f_evals += 1

        # ── Step 3: Forward correction ───────────────────────────────────
        v_k = z_k .+ τ_curr .* (Bz_prev .- Bz_k)

        # ── Step 4: Direction u_k = w_k − v_k ────────────────────────────
        u_k = w_k .- v_k

        # ── Step 5 (algorithm): Coefficient θ_k ──────────────────────────
        # θ_k is CLAMPED at 0 via max{0,·}: this guarantees θ_k ≥ 0, which the
        # projection–contraction descent (manuscript Lemma, part (ii)) requires
        # — the bound ‖x_{k+1}−x†‖² ≤ ‖v_k−x†‖² − ζ(2−ζ)θ_k²‖u_k‖² needs θ_k ≥ 0.
        # When the raw ratio ⟨v_k−z_k,u_k⟩/‖u_k‖² < 0, the PC correction is
        # skipped (θ_k=0 ⟹ x_{k+1}=v_k, the plain FRB step). Original (unclamped)
        # EPCM is recovered whenever the raw ratio ≥ 0. See manuscript Step 5
        # (max{0,·}) and notes/algorithm_improvement.md §24.3.
        norm_u_sq = sum(abs2, u_k)
        θ_k = norm_u_sq > 0 ? max(0.0, dot(v_k .- z_k, u_k) / norm_u_sq) : 0.0

        # ── Step 6 (algorithm): x_{k+1} = v_k − ζ θ_k u_k ────────────────
        x_next = v_k .- (alg.zeta * θ_k) .* u_k

        # ── Step 7 (algorithm): Stepsize update ──────────────────────────
        Δz       = z_prev .- z_k
        ΔBz      = Bz_prev .- Bz_k
        norm_Δz  = norm(Δz)
        norm_ΔBz = norm(ΔBz)
        τ_next = if norm_ΔBz > alg.vartheta_0 * τ_curr * norm_Δz
            # "If" branch: shrink. norm_ΔBz > 0 here, safe to divide.
            alg.vartheta_1 * norm_Δz / norm_ΔBz
        else
            # "Otherwise" branch: grow by (1 + δ_k).
            δ_k = _epcm_delta(alg.delta_rule, alg.delta_exp, state.k)
            (1.0 + δ_k) * τ_curr
        end

        # ── Update SolverState per the "Solver responsibility" contract ──
        # Step 2: x ← x_{k+1}.
        state.x = x_next
        # Step 3: elapsed (f_evals already updated inline above).
        state.elapsed = time() - t0
        # Step 4: step_size ← ρ_n = τ_{k+1}.
        state.step_size = τ_next

        # Step 5: BOTH residuals at the new principal iterate x_{k+1} with
        # ρ = τ_{k+1}. Costs 1 extra B-eval (B(x_{k+1})) per iter (Issue 4).
        Bxnext        = prob.B(x_next)
        state.f_evals += 1
        rho           = τ_next
        prox_arg      = x_next .- rho .* Bxnext
        prox_val      = prob.resolvent_A(prox_arg, rho)
        Rn            = norm(x_next .- prox_val)
        state.residual        = Rn
        state.scaled_residual = rho > 0 ? Rn / rho : NaN

        # Step 6: increment k AFTER the residual block (per contract).
        state.k += 1

        # Step 7: fire :iter observers (state fully updated for this iter).
        for cb in observers
            on_event!(cb, state, :iter)
        end

        # Step 8: check stopping callbacks; first-halt wins.
        halted = false
        for cb in stopping
            should_stop, reason = check_stop(cb, state)
            if should_stop
                state.flag = reason
                halted     = true
                break
            end
        end
        halted && break

        # ── Rotate iteration state for the next loop ─────────────────────
        # (state.x is already x_{k+1}; no separate x_curr to rotate.)
        z_prev  = z_k          # z_{k-1} for next iter
        Bz_prev = Bz_k         # B(z_{k-1}) for next iter (cached, no re-eval)
        τ_curr  = τ_next
    end

    # Fire :terminate observers (final state, e.g. NativeResRecorder reads x, x_prev)
    for cb in observers
        on_event!(cb, state, :terminate)
    end

    # ── Build SolverResult ───────────────────────────────────────────────
    # Extract history from a HistoryCallback if one is present in observers
    history = IterRecord[]
    for cb in observers
        if cb isa HistoryCallback
            history = cb.history
            break
        end
    end

    return make_result(
        converged       = state.flag === :converged,
        iterations      = state.k,
        f_evals         = state.f_evals,
        cpu_time        = state.elapsed,
        x               = state.x,
        flag            = state.flag,
        history         = history,
        residual        = state.residual,
        scaled_residual = state.scaled_residual,
    )
end

# ============================================================================
# MTTM — Mann Tseng-Type Method (Gibali2018 Algorithm 1)
# ============================================================================
#
# Iteration (n = 1, 2, …; current iterate x_n, current step λ_n; manuscript
# lines 1510-1525). Gibali2018's Algorithm 1 (eqs 3-5, 20) writes the
# single-valued operator as A and the set-valued one as B; the manuscript and
# this project use the opposite convention (A set-valued, B single-valued), so
# the iterate below matches the manuscript verbatim:
#
#   y_n     = J^A_{λ_n}(x_n − λ_n B x_n)
#   z_n     = y_n − λ_n (B y_n − B x_n)
#   x_{n+1} = (1 − α_n − β_n) x_n + β_n z_n
#   λ_{n+1} = min{ μ‖x_n − y_n‖ / ‖B x_n − B y_n‖, λ_n }   if B x_n ≠ B y_n
#             λ_n                                            otherwise
#
# Parameters (manuscript line 1607, = Gibali2018 page 19): λ_0 = 7.55,
# μ = 0.85, α_n = 1/n, β_n = (n−1)/(2n). The α/β index n is 1-based
# (n = state.k + 1); see the n = 1 degeneracy note on `_mttm_alpha`.
#
# B-eval accounting: B(x_n) is cached across iterations — the residual eval at
# x_{n+1} (with ρ = λ_{n+1}) doubles as B(x_n) for the next iteration. Per-iter
# cost: 1 fresh B(y_n) + 1 fresh B(x_{n+1}) = 2 B-evals/iter steady-state, plus
# 1 preload B(x_0). This mirrors EPCM's residual convention exactly so the
# universal residual ‖p − J^A_ρ(p − ρ B p)‖ is defined identically across
# methods.

"""
    solve(alg::MTTM, prob::TestProblem, x0::Vector{Float64};
          stopping::Tuple, observers::Tuple=()) -> SolverResult

Run the Mann Tseng-type method (Gibali2018 Algorithm 1) on `prob` from `x0`.
Returns a `SolverResult`; per-iteration history is captured iff a
`HistoryCallback` is in `observers`.

Validates the Gibali2018 Initialization constraints (λ_0 > 0, μ ∈ (0,1))
before iterating. Throws `ArgumentError` on violation.
"""
function solve(alg::MTTM, prob::TestProblem, x0::Vector{Float64};
               stopping::Tuple,
               observers::Tuple = ())

    # ── Parameter validation (Gibali2018 Algorithm 1 initialization) ──────
    alg.lambda_0 > 0 ||
        throw(ArgumentError("MTTM: λ_0 must be > 0, got $(alg.lambda_0)"))
    (0 < alg.mu < 1) ||
        throw(ArgumentError("MTTM: μ must be in (0, 1), got $(alg.mu)"))

    # ── State setup ───────────────────────────────────────────────────────
    state = SolverState(:MTTM, x0)
    t0    = time()

    # Cached B(x_n); preloaded as B(x_0) — counted as the first f_eval.
    Bx_curr       = prob.B(state.x)
    state.f_evals = 1

    # Stepsize: λ_k, starts at λ_0.
    λ_curr          = alg.lambda_0
    state.step_size = λ_curr

    # Fire :init observers (k=0, state.x = x_0, residuals = NaN, step = λ_0).
    for cb in observers
        on_event!(cb, state, :init)
    end

    # ── Main loop ─────────────────────────────────────────────────────────
    while true
        # Contract step 1: snapshot x_n before state.x is reassigned.
        state.x_prev = copy(state.x)

        # 1-based parameter index (Gibali2018 α_k = 1/k, β_k = (k−1)/2k).
        n   = state.k + 1
        α_n = _mttm_alpha(alg.alpha_rule, n)
        β_n = _mttm_beta(alg.beta_rule, n)

        # ── Resolvent (forward–backward) step, uses cached Bx_curr ───────
        y_n  = prob.resolvent_A(state.x .- λ_curr .* Bx_curr, λ_curr)
        By_n = prob.B(y_n)
        state.f_evals += 1

        # ── Tseng correction + Mann average ──────────────────────────────
        z_n    = y_n .- λ_curr .* (By_n .- Bx_curr)
        x_next = (1 - α_n - β_n) .* state.x .+ β_n .* z_n

        # ── Adaptive stepsize update (Gibali2018 eq 20) ──────────────────
        Δx      = state.x .- y_n
        ΔB      = Bx_curr .- By_n
        norm_ΔB = norm(ΔB)
        λ_next  = norm_ΔB > 0 ? min(alg.mu * norm(Δx) / norm_ΔB, λ_curr) : λ_curr

        # ── Update SolverState per the "Solver responsibility" contract ──
        state.x         = x_next
        state.elapsed   = time() - t0
        state.step_size = λ_next

        # Universal residual at x_{n+1} with ρ = λ_{n+1}; the B(x_next) eval
        # is reused as Bx_curr next iteration (no extra eval).
        Bxnext        = prob.B(x_next)
        state.f_evals += 1
        rho           = λ_next
        prox_val      = prob.resolvent_A(x_next .- rho .* Bxnext, rho)
        Rn            = norm(x_next .- prox_val)
        state.residual        = Rn
        state.scaled_residual = rho > 0 ? Rn / rho : NaN

        # Contract step 6: increment k AFTER the residual block.
        state.k += 1

        # Contract step 7: fire :iter observers (state fully updated).
        for cb in observers
            on_event!(cb, state, :iter)
        end

        # Contract step 8: check stopping callbacks; first-halt wins.
        halted = false
        for cb in stopping
            should_stop, reason = check_stop(cb, state)
            if should_stop
                state.flag = reason
                halted     = true
                break
            end
        end
        halted && break

        # ── Rotate iteration state for the next loop ─────────────────────
        Bx_curr = Bxnext       # B(x_{n+1}) cached as next-iter B(x_n)
        λ_curr  = λ_next
    end

    # Fire :terminate observers (final state).
    for cb in observers
        on_event!(cb, state, :terminate)
    end

    # ── Build SolverResult ───────────────────────────────────────────────
    history = IterRecord[]
    for cb in observers
        if cb isa HistoryCallback
            history = cb.history
            break
        end
    end

    return make_result(
        converged       = state.flag === :converged,
        iterations      = state.k,
        f_evals         = state.f_evals,
        cpu_time        = state.elapsed,
        x               = state.x,
        flag            = state.flag,
        history         = history,
        residual        = state.residual,
        scaled_residual = state.scaled_residual,
    )
end

# ============================================================================
# IMTTM — Inertial Mann Tseng-Type Method (Tan2022a Algorithm 3.3)
# ============================================================================
#
# Iteration (n = 1, 2, …; iterates x_{n−1}, x_n; current step λ_n; manuscript
# lines 1527-1551). Tan2022a uses the opposite operator convention (their A is
# single-valued, B set-valued); the block below matches the manuscript /
# project convention (A set-valued, B single-valued):
#
#   w_n     = x_n + θ_n (x_n − x_{n−1})                    [inertial extrapolation]
#   y_n     = J^A_{λ_n}(w_n − λ_n B w_n)
#   z_n     = y_n − λ_n (B y_n − B w_n)
#   x_{n+1} = (1 − α_n − β_n) w_n + β_n z_n                [Mann average about w_n]
#   θ_n     = min{ ε_n / ‖x_n − x_{n−1}‖, θ }   if x_n ≠ x_{n−1}, else θ
#   λ_{n+1} = min{ μ‖w_n − y_n‖ / ‖B w_n − B y_n‖, λ_n }  if B w_n ≠ B y_n, else λ_n
#
# z_n uses B w_n (NOT B x_n). The printed Algorithm-3.3 box in Tan2022a writes
# "Ax_n" in the correction term, but that contradicts Tan's own convergence
# proof (Lemma 3.5, eq 39, p.5399), which expands z_n with Aw_n (= Bw_n in our
# convention) and is consistent with the w_n-based stepsize (eq 38) and the
# sibling Algorithms 3.1/3.2. We implement the proof-consistent Bw_n and
# corrected the manuscript (line 1532, Bx_n→Bw_n). See notes/plan_review_findings.md.
#
# Parameters (manuscript line 1607): μ=0.5, α_n=1/(n+1), β_n=0.5(1−α_n),
# ε_n=100/(n+1)², θ=0.5; λ_0=1 (P1: λ_0=0.01). Index n is 1-based
# (n = state.k + 1). Tan seeds x_0, x_1 arbitrarily; given one initial point we
# set x_0 = x_1 = x0, so the first inertial term θ_1(x_1 − x_0) = 0.
#
# B-eval accounting: the inertial w_n changes every iteration, so B(w_n) cannot
# be cached across iters (unlike MTTM's B(x_n)). Per-iter cost: B(w_n) + B(y_n)
# + B(x_{n+1}) for the universal residual = 3 B-evals/iter. No preload.

"""
    solve(alg::IMTTM, prob::TestProblem, x0::Vector{Float64};
          stopping::Tuple, observers::Tuple=()) -> SolverResult

Run the inertial Mann Tseng-type method (Tan2022a Algorithm 3.3) on `prob`
from `x0`. Returns a `SolverResult`; per-iteration history is captured iff a
`HistoryCallback` is in `observers`.

Validates the Tan2022a Algorithm 3.3 Initialization constraints (λ_0 > 0,
θ > 0, μ ∈ (0,1)). Throws `ArgumentError` on violation.
"""
function solve(alg::IMTTM, prob::TestProblem, x0::Vector{Float64};
               stopping::Tuple,
               observers::Tuple = ())

    # ── Parameter validation (Tan2022a Algorithm 3.3 initialization) ──────
    alg.lambda_0 > 0 ||
        throw(ArgumentError("IMTTM: λ_0 must be > 0, got $(alg.lambda_0)"))
    alg.theta > 0 ||
        throw(ArgumentError("IMTTM: θ must be > 0, got $(alg.theta)"))
    (0 < alg.mu < 1) ||
        throw(ArgumentError("IMTTM: μ must be in (0, 1), got $(alg.mu)"))

    # ── State setup ───────────────────────────────────────────────────────
    state = SolverState(:IMTTM, x0)
    t0    = time()

    # x_{n−1}; Tan seeds x_0 = x_1 = x0, so x_older starts at x_0 = x0 while
    # state.x = x_1 = x0 (the first inertial term is therefore 0).
    x_older = copy(x0)

    # Stepsize: λ_k, starts at λ_0.
    λ_curr          = alg.lambda_0
    state.step_size = λ_curr

    # Fire :init observers (k=0, state.x = x_1, residuals = NaN, step = λ_0).
    for cb in observers
        on_event!(cb, state, :init)
    end

    # ── Main loop ─────────────────────────────────────────────────────────
    while true
        # x_n; its array is never mutated in place (all updates allocate fresh),
        # so aliasing it into x_older below is safe.
        x_curr = state.x

        # Contract step 1: snapshot x_n as the previous principal iterate.
        state.x_prev = copy(x_curr)

        # 1-based parameter index (Tan2022a α_n = 1/(n+1), ε_n = 100/(n+1)²).
        n   = state.k + 1
        α_n = _imttm_alpha(alg.alpha_rule, n)
        β_n = _imttm_beta(alg.beta_rule, α_n)
        ε_n = _imttm_epsilon(alg.epsilon_rule, n)

        # ── Inertial extrapolation w_n = x_n + θ_n(x_n − x_{n−1}) ─────────
        d_inert = x_curr .- x_older
        norm_d  = norm(d_inert)
        θ_n     = norm_d > 0 ? min(ε_n / norm_d, alg.theta) : alg.theta
        w_n     = x_curr .+ θ_n .* d_inert

        # ── Resolvent (forward–backward) step about w_n ──────────────────
        Bw_n = prob.B(w_n)
        state.f_evals += 1
        y_n  = prob.resolvent_A(w_n .- λ_curr .* Bw_n, λ_curr)
        By_n = prob.B(y_n)
        state.f_evals += 1

        # ── Tseng correction (proof-consistent Bw_n) + Mann average ──────
        z_n    = y_n .- λ_curr .* (By_n .- Bw_n)
        x_next = (1 - α_n - β_n) .* w_n .+ β_n .* z_n

        # ── Adaptive stepsize update (Tan2022a eq 38) ────────────────────
        Δw      = w_n .- y_n
        ΔB      = Bw_n .- By_n
        norm_ΔB = norm(ΔB)
        λ_next  = norm_ΔB > 0 ? min(alg.mu * norm(Δw) / norm_ΔB, λ_curr) : λ_curr

        # ── Update SolverState per the "Solver responsibility" contract ──
        state.x         = x_next
        state.elapsed   = time() - t0
        state.step_size = λ_next

        # Universal residual at x_{n+1} with ρ = λ_{n+1}.
        Bxnext        = prob.B(x_next)
        state.f_evals += 1
        rho           = λ_next
        prox_val      = prob.resolvent_A(x_next .- rho .* Bxnext, rho)
        Rn            = norm(x_next .- prox_val)
        state.residual        = Rn
        state.scaled_residual = rho > 0 ? Rn / rho : NaN

        # Contract step 6: increment k AFTER the residual block.
        state.k += 1

        # Contract step 7: fire :iter observers.
        for cb in observers
            on_event!(cb, state, :iter)
        end

        # Contract step 8: check stopping; first-halt wins.
        halted = false
        for cb in stopping
            should_stop, reason = check_stop(cb, state)
            if should_stop
                state.flag = reason
                halted     = true
                break
            end
        end
        halted && break

        # ── Rotate iteration state for the next loop ─────────────────────
        x_older = x_curr       # x_{n−1} ← x_n (array never mutated; alias safe)
        λ_curr  = λ_next
    end

    # Fire :terminate observers.
    for cb in observers
        on_event!(cb, state, :terminate)
    end

    # ── Build SolverResult ───────────────────────────────────────────────
    history = IterRecord[]
    for cb in observers
        if cb isa HistoryCallback
            history = cb.history
            break
        end
    end

    return make_result(
        converged       = state.flag === :converged,
        iterations      = state.k,
        f_evals         = state.f_evals,
        cpu_time        = state.elapsed,
        x               = state.x,
        flag            = state.flag,
        history         = history,
        residual        = state.residual,
        scaled_residual = state.scaled_residual,
    )
end

# ============================================================================
# SFRBM — Strongly-Convergent FRB with Momentum (Yao2024 Algorithm 2)
# ============================================================================
#
# Iteration (k = 0, 1, …; iterates x_k, auxiliary z_k; steps λ_k, λ_{k−1};
# manuscript lines 1572-1595 = Yao2024 Alg 2 eq 41-42 verbatim — no A↔B issue,
# Yao already uses A set-valued / B single-valued in (1)):
#
#   y_k     = x_k/(1+θ) + θ z_k/(1+θ)
#   x_{k+1} = J^A_{λ_k}( β_k x_0 + (1−β_k) y_k − λ_k B x_k
#                        − λ_{k−1}(1−β_k)(B x_k − B x_{k−1}) )
#   z_{k+1} = (1−β_k)θ/(1+θ) · x_k + (1−β_k)/(1+θ) · z_k + β_k x_0
#   λ_{k+1} = min{ μ‖x_k − x_{k+1}‖ / ‖B x_k − B x_{k+1}‖, λ_k }  if B x_k ≠ B x_{k+1}, else λ_k
#
# Initialization (Yao2024 Alg 2): x_{-1} = x_0 = z_0 (all seeded equal — we set
# them to the single given x0), λ_{-1} = λ_0 > 0, θ > 0, and μ ∈ (0,1) with the
# JOINT bound 2μ < 1/(1+θ). With x_{-1} = x_0 the k=0 reflection term
# λ_{-1}(1−β_0)(Bx_0 − Bx_{-1}) vanishes (Bx_{-1} = Bx_0).
#
# Parameters (manuscript line 1623): λ_{-1} = λ_0 = μ = 1e-4, θ = 10,
# β_k = 1/(5000(k+1)). Index k is 0-based (k = state.k), matching Yao.
#
# B-eval accounting: the FRB reflection reuses cached B x_k, B x_{k−1}; only
# B(x_{k+1}) is fresh each iter, and it serves the stepsize, the universal
# residual, AND the next iter's B x_k. So 1 B-eval/iter steady-state (cheapest
# of the five), plus 1 preload B(x_0). No separate B(y_k) (y_k feeds only the
# resolvent argument, not B).

"""
    solve(alg::SFRBM, prob::TestProblem, x0::Vector{Float64};
          stopping::Tuple, observers::Tuple=()) -> SolverResult

Run the strongly-convergent forward-reflected-backward method with momentum
(Yao2024 Algorithm 2) on `prob` from `x0`. Returns a `SolverResult`;
per-iteration history is captured iff a `HistoryCallback` is in `observers`.

Validates the Yao2024 Algorithm 2 Initialization constraints (λ_{-1} > 0,
λ_0 > 0, θ > 0, μ > 0 with 2μ < 1/(1+θ)). Throws `ArgumentError` on violation.
"""
function solve(alg::SFRBM, prob::TestProblem, x0::Vector{Float64};
               stopping::Tuple,
               observers::Tuple = ())

    # ── Parameter validation (Yao2024 Algorithm 2 initialization) ─────────
    alg.theta > 0 ||
        throw(ArgumentError("SFRBM: θ must be > 0, got $(alg.theta)"))
    alg.lambda_0 > 0 ||
        throw(ArgumentError("SFRBM: λ_0 must be > 0, got $(alg.lambda_0)"))
    alg.lambda_minus1 > 0 ||
        throw(ArgumentError("SFRBM: λ_{-1} must be > 0, got $(alg.lambda_minus1)"))
    alg.mu > 0 ||
        throw(ArgumentError("SFRBM: μ must be > 0, got $(alg.mu)"))
    # Joint admissibility: 2μ < 1/(1+θ) (Yao2024 Alg 2 init; implies μ < 1/2 < 1).
    bound_mu = 1.0 / (1.0 + alg.theta)
    (2 * alg.mu < bound_mu) ||
        throw(ArgumentError("SFRBM: require 2μ < 1/(1+θ) = $bound_mu (Yao2024 Alg 2 init), " *
                            "got 2μ = $(2 * alg.mu)"))

    # ── State setup ───────────────────────────────────────────────────────
    state = SolverState(:SFRBM, x0)
    t0    = time()

    # Constant momentum coefficients (θ fixed across iters).
    θ       = alg.theta
    inv_1pθ = 1.0 / (1.0 + θ)          # 1/(1+θ)
    θ_1pθ   = θ * inv_1pθ              # θ/(1+θ)

    # Anchor x_0 (constant) and auxiliary z_0 = x_0.
    x_anchor = copy(x0)
    z_curr   = copy(x0)               # z_k, starts as z_0 = x_0

    # Cached B-values: Bx_k preloaded as B(x_0); Bx_{k−1} := Bx_{-1} = Bx_0.
    Bx_curr       = prob.B(state.x)
    Bx_prev       = Bx_curr           # alias safe (never mutated in place)
    state.f_evals = 1

    # Steps: λ_k starts at λ_0; λ_{k−1} starts at λ_{-1}.
    λ_curr          = alg.lambda_0
    λ_prev          = alg.lambda_minus1
    state.step_size = λ_curr

    # Fire :init observers (k=0, state.x = x_0, residuals = NaN, step = λ_0).
    for cb in observers
        on_event!(cb, state, :init)
    end

    # ── Main loop ─────────────────────────────────────────────────────────
    while true
        # x_k; its array is never mutated in place, so aliasing is safe.
        x_k = state.x

        # Contract step 1: snapshot x_k as the previous principal iterate.
        state.x_prev = copy(x_k)

        # 0-based index (Yao β_k = 1/(5000(k+1)), k = 0, 1, …).
        k   = state.k
        β_k = _sfrbm_beta(alg.beta_rule, k)

        # ── Momentum combination y_k ─────────────────────────────────────
        y_k = inv_1pθ .* x_k .+ θ_1pθ .* z_curr

        # ── Anchored forward-reflected-backward resolvent step ───────────
        # refl = β_k x_0 + (1−β_k)y_k − λ_k B x_k − λ_{k−1}(1−β_k)(B x_k − B x_{k−1})
        refl   = β_k .* x_anchor .+ (1 - β_k) .* y_k .-
                 λ_curr .* Bx_curr .-
                 (λ_prev * (1 - β_k)) .* (Bx_curr .- Bx_prev)
        x_next = prob.resolvent_A(refl, λ_curr)

        # ── Auxiliary update z_{k+1} ─────────────────────────────────────
        z_next = ((1 - β_k) * θ_1pθ) .* x_k .+
                 ((1 - β_k) * inv_1pθ) .* z_curr .+
                 β_k .* x_anchor

        # ── One fresh B-eval: B(x_{k+1}) (stepsize + residual + next cache) ─
        Bxnext        = prob.B(x_next)
        state.f_evals += 1

        # ── Adaptive stepsize update (Yao2024 eq 42) ─────────────────────
        Δx      = x_k .- x_next
        ΔB      = Bx_curr .- Bxnext
        norm_ΔB = norm(ΔB)
        λ_next  = norm_ΔB > 0 ? min(alg.mu * norm(Δx) / norm_ΔB, λ_curr) : λ_curr

        # ── Update SolverState per the "Solver responsibility" contract ──
        state.x         = x_next
        state.elapsed   = time() - t0
        state.step_size = λ_next

        # Universal residual at x_{k+1} with ρ = λ_{k+1}, reusing Bxnext.
        rho           = λ_next
        prox_val      = prob.resolvent_A(x_next .- rho .* Bxnext, rho)
        Rn            = norm(x_next .- prox_val)
        state.residual        = Rn
        state.scaled_residual = rho > 0 ? Rn / rho : NaN

        # Contract step 6: increment k AFTER the residual block.
        state.k += 1

        # Contract step 7: fire :iter observers.
        for cb in observers
            on_event!(cb, state, :iter)
        end

        # Contract step 8: check stopping; first-halt wins.
        halted = false
        for cb in stopping
            should_stop, reason = check_stop(cb, state)
            if should_stop
                state.flag = reason
                halted     = true
                break
            end
        end
        halted && break

        # ── Rotate iteration state for the next loop ─────────────────────
        z_curr  = z_next       # z_k ← z_{k+1}
        Bx_prev = Bx_curr      # B x_{k−1} ← B x_k
        Bx_curr = Bxnext       # B x_k ← B x_{k+1} (cached; no re-eval)
        λ_prev  = λ_curr       # λ_{k−1} ← λ_k
        λ_curr  = λ_next       # λ_k ← λ_{k+1}
    end

    # Fire :terminate observers.
    for cb in observers
        on_event!(cb, state, :terminate)
    end

    # ── Build SolverResult ───────────────────────────────────────────────
    history = IterRecord[]
    for cb in observers
        if cb isa HistoryCallback
            history = cb.history
            break
        end
    end

    return make_result(
        converged       = state.flag === :converged,
        iterations      = state.k,
        f_evals         = state.f_evals,
        cpu_time        = state.elapsed,
        x               = state.x,
        flag            = state.flag,
        history         = history,
        residual        = state.residual,
        scaled_residual = state.scaled_residual,
    )
end

# ============================================================================
# IFRAB — Variable-Inertial Forward-Reflected-Anchored-Backward (Izuchukwu2023 Alg 4.5)
# ============================================================================
#
# Iteration (n = 1, 2, …; iterates w_{n−1}, w_n; steps δ_{n−1}, δ_n; anchor x̂;
# manuscript lines 1553-1570 = Izuchukwu2023 Alg 4.5 + stepsize eq 3.2).
# Izuchukwu uses S set-valued / T single-valued; their S = our A, their T = our
# B, so the block matches the manuscript / project convention directly:
#
#   w_{n+1} = J^A_{δ_n}( σ_n x̂ + (1−σ_n)(w_n + ϑ_n(w_n − w_{n−1}))
#                        − δ_n B w_n − δ_{n−1}(1−σ_n)(B w_n − B w_{n−1}) )
#   δ_{n+1} = min{ r̄‖w_n − w_{n+1}‖ / ‖B w_n − B w_{n+1}‖, δ_n + c_n }  if B w_n ≠ B w_{n+1}
#             δ_n + c_n                                                   otherwise
#
# Note the stepsize is NON-MONOTONE: it may GROW, capped at δ_n + c_n with
# ∑ c_n < ∞ (Remark 3.2 ⇒ δ_n converges). Contrast MTTM/IMTTM/SFRBM whose
# adaptive steps only shrink (min{·, λ_n}).
#
# Initialization (Izuchukwu2023 Alg 4.5): arbitrary anchor x̂ and iterates
# w_0, w_1; given one initial point we set x̂ = w_0 = w_1 = x0. Then Bw_0 = Bw_1,
# so at n=1 both the inertial term ϑ_1(w_1 − w_0) and the reflection-memory
# term δ_0(1−σ_1)(Bw_1 − Bw_0) vanish. δ_n starts at δ_1, δ_{n−1} at δ_0.
# Index n is 1-based (n = state.k + 1, matching Alg 4.5's n ≥ 1). The strong
# limit is P_{(A+B)^{-1}(0)} x̂ — the projection of the anchor onto the solution set.
#
# Parameters (manuscript lines 1609-1622): δ_0=0.1, δ_1=0.3, r̄=0.3,
# σ_n=0.005/(3n+25000), c_n=1/(n²+1), ϑ_n=ϑ̄·n/(n+1) with ϑ̄=0.04.
#
# B-eval accounting: the FRB reflection reuses cached B w_n, B w_{n−1}; only
# B(w_{n+1}) is fresh each iter, serving the stepsize, the universal residual,
# AND the next iter's B w_n. So 1 B-eval/iter steady-state, plus 1 preload B(x0)
# (which seeds both Bw_0 and Bw_1).

"""
    solve(alg::IFRAB, prob::TestProblem, x0::Vector{Float64};
          stopping::Tuple, observers::Tuple=()) -> SolverResult

Run the variable-inertial forward-reflected-anchored-backward method
(Izuchukwu2023 Algorithm 4.5) on `prob` from `x0`, using `x0` as the anchor x̂
and as both initial iterates w_0 = w_1. Returns a `SolverResult`;
per-iteration history is captured iff a `HistoryCallback` is in `observers`.

Validates δ_0 > 0, δ_1 > 0, 0 < r̄ < 1/2, and 0 ≤ ϑ̄ < (1/2 − r̄)/2 (the
β̄-independent part of Izuchukwu2023 Thm 4.4; an admissible β̄ ∈ (0,1/4) with
ϑ̄ < β̄/2 and r̄ ∈ (β̄,(1−2β̄)/2) exists for the manuscript params, e.g. β̄≈0.1).
Throws `ArgumentError` on violation.
"""
function solve(alg::IFRAB, prob::TestProblem, x0::Vector{Float64};
               stopping::Tuple,
               observers::Tuple = ())

    # ── Parameter validation (Izuchukwu2023 Alg 4.5 + Thm 4.4) ────────────
    alg.delta_0 > 0 ||
        throw(ArgumentError("IFRAB: δ_0 must be > 0, got $(alg.delta_0)"))
    alg.delta_1 > 0 ||
        throw(ArgumentError("IFRAB: δ_1 must be > 0, got $(alg.delta_1)"))
    (0 < alg.rbar < 0.5) ||
        throw(ArgumentError("IFRAB: require 0 < r̄ < 1/2 (r̄ ∈ (β̄,(1−2β̄)/2) ⊂ (0,1/2)), got r̄ = $(alg.rbar)"))
    bound_vartheta = (0.5 - alg.rbar) / 2
    (0 <= alg.vartheta_bar < bound_vartheta) ||
        throw(ArgumentError("IFRAB: require 0 ≤ ϑ̄ < (1/2 − r̄)/2 = $bound_vartheta (Izuchukwu2023 Thm 4.4), " *
                            "got ϑ̄ = $(alg.vartheta_bar)"))

    # ── State setup ───────────────────────────────────────────────────────
    state = SolverState(:IFRAB, x0)
    t0    = time()

    # Anchor x̂ and w_0; given one point, x̂ = w_0 = w_1 = x0.
    w_anchor = copy(x0)               # x̂ (constant)
    w_older  = copy(x0)               # w_{n−1}, starts as w_0

    # Cached B-values: Bw_n preloaded as B(w_1) = B(x0); Bw_{n−1} := Bw_0 = Bw_1.
    Bw_curr       = prob.B(state.x)
    Bw_prev       = Bw_curr           # alias safe (never mutated in place)
    state.f_evals = 1

    # Steps: δ_n starts at δ_1; δ_{n−1} starts at δ_0.
    δ_curr          = alg.delta_1
    δ_prev          = alg.delta_0
    state.step_size = δ_curr

    # Fire :init observers (k=0, state.x = w_1, residuals = NaN, step = δ_1).
    for cb in observers
        on_event!(cb, state, :init)
    end

    # ── Main loop ─────────────────────────────────────────────────────────
    while true
        # w_n; its array is never mutated in place, so aliasing is safe.
        w_n = state.x

        # Contract step 1: snapshot w_n as the previous principal iterate.
        state.x_prev = copy(w_n)

        # 1-based index (Izuchukwu Alg 4.5 n ≥ 1; σ_n, c_n, ϑ_n indexed by n).
        n   = state.k + 1
        σ_n = _ifrab_sigma(alg.sigma_rule, n)
        c_n = _ifrab_c(alg.c_rule, n)
        ϑ_n = _ifrab_vartheta(alg.vartheta_rule, alg.vartheta_bar, n)

        # ── Anchored + variable-inertial forward-reflected resolvent step ─
        # arg = σ_n x̂ + (1−σ_n)(w_n + ϑ_n(w_n − w_{n−1}))
        #       − δ_n B w_n − δ_{n−1}(1−σ_n)(B w_n − B w_{n−1})
        inertial = w_n .+ ϑ_n .* (w_n .- w_older)
        arg      = σ_n .* w_anchor .+ (1 - σ_n) .* inertial .-
                   δ_curr .* Bw_curr .-
                   (δ_prev * (1 - σ_n)) .* (Bw_curr .- Bw_prev)
        w_next   = prob.resolvent_A(arg, δ_curr)

        # ── One fresh B-eval: B(w_{n+1}) (stepsize + residual + next cache) ─
        Bw_next       = prob.B(w_next)
        state.f_evals += 1

        # ── Adaptive (non-monotone) stepsize update (Izuchukwu eq 3.2) ───
        Δw      = w_n .- w_next
        ΔB      = Bw_curr .- Bw_next
        norm_ΔB = norm(ΔB)
        cap     = δ_curr + c_n     # growth cap δ_n + c_n
        δ_next  = norm_ΔB > 0 ? min(alg.rbar * norm(Δw) / norm_ΔB, cap) : cap

        # ── Update SolverState per the "Solver responsibility" contract ──
        state.x         = w_next
        state.elapsed   = time() - t0
        state.step_size = δ_next

        # Universal residual at w_{n+1} with ρ = δ_{n+1}, reusing Bw_next.
        rho           = δ_next
        prox_val      = prob.resolvent_A(w_next .- rho .* Bw_next, rho)
        Rn            = norm(w_next .- prox_val)
        state.residual        = Rn
        state.scaled_residual = rho > 0 ? Rn / rho : NaN

        # Contract step 6: increment k AFTER the residual block.
        state.k += 1

        # Contract step 7: fire :iter observers.
        for cb in observers
            on_event!(cb, state, :iter)
        end

        # Contract step 8: check stopping; first-halt wins.
        halted = false
        for cb in stopping
            should_stop, reason = check_stop(cb, state)
            if should_stop
                state.flag = reason
                halted     = true
                break
            end
        end
        halted && break

        # ── Rotate iteration state for the next loop ─────────────────────
        w_older = w_n          # w_{n−1} ← w_n (array never mutated; alias safe)
        Bw_prev = Bw_curr      # B w_{n−1} ← B w_n
        Bw_curr = Bw_next      # B w_n ← B w_{n+1} (cached; no re-eval)
        δ_prev  = δ_curr       # δ_{n−1} ← δ_n
        δ_curr  = δ_next       # δ_n ← δ_{n+1}
    end

    # Fire :terminate observers.
    for cb in observers
        on_event!(cb, state, :terminate)
    end

    # ── Build SolverResult ───────────────────────────────────────────────
    history = IterRecord[]
    for cb in observers
        if cb isa HistoryCallback
            history = cb.history
            break
        end
    end

    return make_result(
        converged       = state.flag === :converged,
        iterations      = state.k,
        f_evals         = state.f_evals,
        cpu_time        = state.elapsed,
        x               = state.x,
        flag            = state.flag,
        history         = history,
        residual        = state.residual,
        scaled_residual = state.scaled_residual,
    )
end
