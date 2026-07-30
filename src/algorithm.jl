# algorithm.jl — `solve` method bodies for each AbstractAlgorithm subtype.
#
# Contains the solver implementations for the proposed methods and the baseline
# competitors.
#
# Every body honours the 9-step "Solver responsibility" contract documented
# in `callbacks.jl`'s `SolverState` docstring. In particular:
#   - state.x_prev is updated BEFORE state.x is overwritten.
#   - state.residual and state.scaled_residual are updated BEFORE invoking
#     the stopping tuple (otherwise NanStopping short-circuits or
#     ResidualStopping reads stale values).

"""
    algorithm_B!(state, B, x)

Evaluate the forward operator `B(x)` and register one F-evaluation.  Use this
only when the value is required by the algorithmic update, line search,
stepsize rule, or cache for a later algorithmic update.  Calls made solely for
residual diagnostics or stopping checks must use `B(x)` directly and are not
included in `state.f_evals`.
"""
@inline function algorithm_B!(state::SolverState, B, x)
    state.f_evals += 1
    return B(x)
end

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
- `:cs_inv_k_plus_100_sq` → 1/(k+100)^2, the compressed-sensing ξ_k rule;
  `exp` is ignored.
The `exp > 1` admissibility is validated once at `solve` entry, not here.
"""
@inline function _epcm_delta(rule::Symbol, exp::Float64, k::Int)
    rule === :power && return 1.0 / (k + 1.0)^exp
    rule === :zero  && return 0.0
    rule === :cs_inv_k_plus_100_sq && return 1.0 / (k + 100.0)^2
    throw(ArgumentError("Unknown EPCM delta_rule :$rule; known: :power, :zero, :cs_inv_k_plus_100_sq"))
end

"""
    _epcm_sigma(rule::Symbol, scale::Float64, exp::Float64, k::Int) -> Float64

EPCM's σ_k Halpern-anchoring weight (parameter restriction: σ_k∈(0,1), σ_k→0,
∑ σ_k = ∞; Step 11-2 parametrization). Rules:
- `:power` → σ_k = `scale` / (k+2)^`exp`. Admissible iff `exp ∈ (0,1]` (so
  ∑ diverges) and σ_0 = `scale`/2^`exp` < 1. Default `scale=1, exp=1`
  reproduces the manuscript's 1/(k+2) (Lieder-optimal Halpern weight).
- `:cs_inv_1000k_plus_1` → σ_k = 1/(1000k+1), for the compressed-sensing
  experiment. `scale` and `exp` are ignored.
- `:log_power` → σ_k = `scale`/(1 + log(k+1))^`exp`; this normalized
  logarithmic family is defined at k=0.
The admissibility is validated once at `solve` entry, not here.
"""
@inline function _epcm_sigma(rule::Symbol, scale::Float64, exp::Float64, k::Int)
    rule === :power && return scale / (k + 2.0)^exp
    rule === :cs_inv_1000k_plus_1 && return 1.0 / (1000.0 * k + 1.0)
    rule === :log_power && return scale / (1.0 + log(k + 1.0))^exp
    throw(ArgumentError("Unknown EPCM sigma_rule :$rule; known: :power, :cs_inv_1000k_plus_1, :log_power"))
end

"""
    _maefbfp_alpha(scale::Float64, exp::Float64, k::Int) -> Float64

MAEFBFP's nonnegative decay sequence
`α_k = scale / (k+1)^exp`, with `scale ≥ 0`, `exp > 0`, so `α_k → 0`.
"""
@inline _maefbfp_alpha(scale::Float64, exp::Float64, k::Int) = scale / (k + 1.0)^exp

"""
    _maefbfp_beta(scale::Float64, exp::Float64, k::Int) -> Float64

MAEFBFP's lower-bounded decay sequence
`β_k = 1 + scale / (k+1)^exp`, with `scale ≥ 0`, `exp > 0`, so `β_k ≥ 1`
and `β_k → 1`.
"""
@inline _maefbfp_beta(scale::Float64, exp::Float64, k::Int) = 1.0 + scale / (k + 1.0)^exp

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

"""
    _vafbs_alpha(rule::Symbol, scale::Float64, n::Int) -> Float64

VAFBS's viscosity weight sequence α_n. Rules:
- `:c_over_n` → α_n = scale / n

Admissibility is validated once at `solve` entry: `scale ∈ (0,1)` ensures
`α_n ∈ (0,1)` for all `n ≥ 1`, while still satisfying `α_n → 0` and
`∑ α_n = ∞`.
"""
@inline function _vafbs_alpha(rule::Symbol, scale::Float64, n::Int)
    rule === :c_over_n && return scale / n
    throw(ArgumentError("Unknown VAFBS alpha_rule :$rule; known: :c_over_n"))
end

"""
    _mditsm_alpha(rule::Symbol, scale::Float64, base::Float64, n::Int) -> Float64

MDITSM's first inertial sequence α_n. Index `n` is 1-based (`n = state.k + 1`).
Rules:
- `:one_minus_scale_over_base_pow_n` → α_n = 1 - scale / base^n

The paper's Example 5.1 choice is recovered by `scale=1`, `base=10`, giving
α_n = 1 - 10^{-n}.
"""
@inline function _mditsm_alpha(rule::Symbol, scale::Float64, base::Float64, n::Int)
    rule === :one_minus_scale_over_base_pow_n && return 1.0 - scale / (base^n)
    throw(ArgumentError("Unknown MDITSM alpha_rule :$rule; known: :one_minus_scale_over_base_pow_n"))
end

"""
    _mditsm_beta(rule::Symbol, cap::Float64, shift::Float64, n::Int) -> Float64

MDITSM's second inertial sequence β_n. Index `n` is 1-based. Rules:
- `:cap_minus_inv_n_plus_shift` → β_n = cap - 1/(shift + n)

The paper's Example 5.1 choice is recovered by `cap=0.1`, `shift=1000`.
"""
@inline function _mditsm_beta(rule::Symbol, cap::Float64, shift::Float64, n::Int)
    rule === :cap_minus_inv_n_plus_shift && return cap - 1.0 / (shift + n)
    throw(ArgumentError("Unknown MDITSM beta_rule :$rule; known: :cap_minus_inv_n_plus_shift"))
end

"""
    _mditsm_theta(rule::Symbol, cap::Float64, shift::Float64, n::Int) -> Float64

MDITSM's relaxation sequence θ_n. Index `n` is 1-based. Rules:
- `:cap_minus_inv_n_plus_shift` → θ_n = cap - 1/(shift + n)

The paper's Example 5.1 choice is recovered by `cap=0.45`, `shift=1000`.
"""
@inline function _mditsm_theta(rule::Symbol, cap::Float64, shift::Float64, n::Int)
    rule === :cap_minus_inv_n_plus_shift && return cap - 1.0 / (shift + n)
    throw(ArgumentError("Unknown MDITSM theta_rule :$rule; known: :cap_minus_inv_n_plus_shift"))
end

"""
    _mditsm_mu_aux(rule::Symbol, scale::Float64, exp::Float64, n::Int) -> Float64

MDITSM's relaxation sequence μ_n in the adaptive stepsize update. Index `n` is
1-based. Rules:
- `:scale_over_n_pow` → μ_n = scale / n^exp
- `:zero`             → μ_n ≡ 0
"""
@inline function _mditsm_mu_aux(rule::Symbol, scale::Float64, exp::Float64, n::Int)
    rule === :scale_over_n_pow && return scale / (n^exp)
    rule === :zero             && return 0.0
    throw(ArgumentError("Unknown MDITSM mu_rule :$rule; known: :scale_over_n_pow, :zero"))
end

"""
    _mditsm_p(rule::Symbol, scale::Float64, exp::Float64, n::Int) -> Float64

MDITSM's additive stepsize-growth budget p_n. Index `n` is 1-based. Rules:
- `:scale_over_n_pow` → p_n = scale / n^exp
- `:zero`             → p_n ≡ 0
"""
@inline function _mditsm_p(rule::Symbol, scale::Float64, exp::Float64, n::Int)
    rule === :scale_over_n_pow && return scale / (n^exp)
    rule === :zero             && return 0.0
    throw(ArgumentError("Unknown MDITSM p_rule :$rule; known: :scale_over_n_pow, :zero"))
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
# Per-iter ALGORITHM cost: **1 new B-eval for z_k** (B(z_{k-1}) is cached) —
# single-call. The universal residual at x_{k+1} also evaluates B(x_{k+1}), but
# that is monitoring only (not algorithm work) and is NOT counted in f_evals.
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
    bound_theta0 = sqrt(alg.gamma / (2 * (3 * alg.gamma + 1)))
    alg.vartheta_0 < bound_theta0 ||
        throw(ArgumentError("EPCM: strict admissibility — ϑ_0 < √(γ/(2(3γ+1))) = $bound_theta0, " *
                            "got ϑ_0 = $(alg.vartheta_0). Strictness leaves room for an admissible μ ∈ (1, $bound_theta0/$(alg.vartheta_0)) in the convergence proof."))
    (0 < alg.zeta < 4 / (3 + alg.gamma)) ||
        throw(ArgumentError("EPCM: ζ must be in (0, 4/(3+γ)) = (0, $(4/(3+alg.gamma))), got $(alg.zeta)"))
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
        τ_next = if norm_ΔBz > alg.vartheta_0 / τ_curr * norm_Δz
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
        # ρ = τ_{k+1}. The B(x_{k+1}) here is for the universal residual R_n ONLY
        # (monitoring); it is NOT an algorithm eval — the EPCM iteration is
        # single-call (one new B(z_k)/iter) — so it does NOT increment f_evals.
        Bxnext        = prob.B(x_next)
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
# EFBFP — Extrapolated Forward–Backward–Forward (manuscript Algorithm 2)
# ============================================================================
#
# EFBFP = EPCM without the projection–contraction step. Steps 1–3 are identical
# to EPCM; Step 3's forward correction v_k IS the next iterate x_{k+1} (no u_k,
# θ_k, ζ). Stepsize = EPCM's ϑ-rule. Single-call: 1 new B(z_k)/iter (B(z_{k-1})
# cached); the residual B(x_{k+1}) is monitoring only and is NOT counted.

"""
    solve(alg::EFBFP, prob::TestProblem, x0::Vector{Float64};
          stopping::Tuple, observers::Tuple = ()) -> SolverResult

Run the EFBFP iteration (manuscript Algorithm 2) on `prob` from `x0`. EPCM minus
the projection–contraction step: x_{k+1} = v_k. Validates the input constraints
(0 < ϑ_1 < ϑ_0, τ_0 > 0, σ/δ admissibility) before iterating.
"""
function solve(alg::EFBFP, prob::TestProblem, x0::Vector{Float64};
               stopping::Tuple,
               observers::Tuple = ())

    # ── Parameter validation (manuscript Algorithm 2 input constraints) ──
    alg.vartheta_0 > 0 ||
        throw(ArgumentError("EFBFP: ϑ_0 must be > 0, got $(alg.vartheta_0)"))
    (0 < alg.vartheta_1 < alg.vartheta_0) ||
        throw(ArgumentError("EFBFP: require 0 < ϑ_1 < ϑ_0, got ϑ_1=$(alg.vartheta_1), ϑ_0=$(alg.vartheta_0)"))
    alg.vartheta_0 < 1 / sqrt(6) ||
        throw(ArgumentError("EFBFP: strict admissibility — ϑ_0 < 1/√6 ≈ 0.4082 (Theorem thm:efbfp), got ϑ_0 = $(alg.vartheta_0)."))
    alg.tau_0 > 0 ||
        throw(ArgumentError("EFBFP: τ_0 must be > 0, got $(alg.tau_0)"))
    if alg.sigma_rule === :power
        (0 < alg.sigma_exp <= 1) ||
            throw(ArgumentError("EFBFP: sigma_exp must be in (0, 1] for Σσ_k = ∞, got $(alg.sigma_exp)"))
        alg.sigma_scale > 0 ||
            throw(ArgumentError("EFBFP: sigma_scale must be > 0, got $(alg.sigma_scale)"))
        σ0 = alg.sigma_scale / 2.0^alg.sigma_exp
        σ0 < 1 ||
            throw(ArgumentError("EFBFP: require σ_0 = sigma_scale/2^sigma_exp < 1, got σ_0 = $σ0"))
    end
    if alg.delta_rule === :power
        alg.delta_exp > 1 ||
            throw(ArgumentError("EFBFP: delta_exp must be > 1 for Σδ_k < ∞, got $(alg.delta_exp)"))
    end

    state   = SolverState(:EFBFP, x0)
    t0      = time()
    x_0     = copy(x0)
    z_prev  = copy(x0)              # z_{-1} = x_0
    state.f_evals = 0
    Bz_prev = algorithm_B!(state, prob.B, z_prev) # B(z_{-1}) — first f_eval
    τ_curr  = alg.tau_0
    state.step_size = τ_curr
    for cb in observers
        on_event!(cb, state, :init)
    end

    while true
        state.x_prev = copy(state.x)

        # Step 1: Extrapolation
        σ_k = _epcm_sigma(alg.sigma_rule, alg.sigma_scale, alg.sigma_exp, state.k)
        w_k = σ_k .* x_0 .+ (1 - σ_k) .* state.x

        # Step 2: Resolvent (uses cached Bz_prev = B(z_{k-1}))
        z_k = prob.resolvent_A(w_k .- τ_curr .* Bz_prev, τ_curr)
        Bz_k = algorithm_B!(state, prob.B, z_k)

        # Step 3: Forward correction = iterate update (NO projection–contraction)
        x_next = z_k .+ τ_curr .* (Bz_prev .- Bz_k)

        # Step 4: Stepsize update — mirrors EPCM's solve() verbatim. NB: the code
        # condition is ϑ_0·τ_k (kept identical to EPCM so the only EFBFP↔EPCM
        # difference is the PC step); the manuscript box uses ϑ_0/τ_k. This
        # code↔paper discrepancy is shared with EPCM and tracked separately.
        Δz       = z_prev .- z_k
        ΔBz      = Bz_prev .- Bz_k
        norm_Δz  = norm(Δz)
        norm_ΔBz = norm(ΔBz)
        τ_next = if norm_ΔBz > alg.vartheta_0 / τ_curr * norm_Δz
            alg.vartheta_1 * norm_Δz / norm_ΔBz
        else
            δ_k = _epcm_delta(alg.delta_rule, alg.delta_exp, state.k)
            (1.0 + δ_k) * τ_curr
        end

        state.x         = x_next
        state.elapsed   = time() - t0
        state.step_size = τ_next

        # Residual monitoring (single-call: this B(x_next) is NOT an algorithm eval)
        Bxnext        = prob.B(x_next)
        rho           = τ_next
        prox_val      = prob.resolvent_A(x_next .- rho .* Bxnext, rho)
        Rn            = norm(x_next .- prox_val)
        state.residual        = Rn
        state.scaled_residual = rho > 0 ? Rn / rho : NaN

        state.k += 1
        for cb in observers
            on_event!(cb, state, :iter)
        end
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

        z_prev  = z_k
        Bz_prev = Bz_k
        τ_curr  = τ_next
    end

    for cb in observers
        on_event!(cb, state, :terminate)
    end
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
# AEFBFP — Adaptive Extrapolated Forward–Backward–Forward (manuscript Alg 3)
# ============================================================================
#
# AEFBFP = EFBFP with the min-rule self-adaptive stepsize. Steps 1–3 identical
# to EFBFP; Step 4 is the Malitsky–Tam / IFRAB form
#   τ_{k+1} = min{ μ‖z_k−z_{k-1}‖/‖B(z_k)−B(z_{k-1})‖, τ_k+ξ_k }   (B differs)
#         or  τ_k + ξ_k                                            (otherwise).
# Single-call accounting identical to EFBFP.

"""
    solve(alg::AEFBFP, prob::TestProblem, x0::Vector{Float64};
          stopping::Tuple, observers::Tuple = ()) -> SolverResult

Run the AEFBFP iteration (manuscript Algorithm 3) on `prob` from `x0`. EFBFP with
the min-rule self-adaptive stepsize. Validates the input constraints
(μ ∈ (0,1), τ_0 > 0, xi_exp > 1, σ admissibility) before iterating.
"""
function solve(alg::AEFBFP, prob::TestProblem, x0::Vector{Float64};
               stopping::Tuple,
               observers::Tuple = ())

    # ── Parameter validation (manuscript Algorithm 3 input constraints) ──
    (0 < alg.mu < 0.5) ||
        throw(ArgumentError("AEFBFP: strict admissibility — μ ∈ (0, 1/2 ≈ 0.5) (Theorem thm:aefbfp), got μ = $(alg.mu)."))
    alg.tau_0 > 0 ||
        throw(ArgumentError("AEFBFP: τ_0 must be > 0, got $(alg.tau_0)"))
    if alg.sigma_rule === :power
        (0 < alg.sigma_exp <= 1) ||
            throw(ArgumentError("AEFBFP: sigma_exp must be in (0, 1] for Σσ_k = ∞, got $(alg.sigma_exp)"))
        alg.sigma_scale > 0 ||
            throw(ArgumentError("AEFBFP: sigma_scale must be > 0, got $(alg.sigma_scale)"))
        σ0 = alg.sigma_scale / 2.0^alg.sigma_exp
        σ0 < 1 ||
            throw(ArgumentError("AEFBFP: require σ_0 = sigma_scale/2^sigma_exp < 1, got σ_0 = $σ0"))
    elseif alg.sigma_rule === :log_power
        alg.sigma_exp > 0 ||
            throw(ArgumentError("AEFBFP: sigma_exp must be > 0 for the :log_power rule, got $(alg.sigma_exp)"))
        (0 < alg.sigma_scale < 1) ||
            throw(ArgumentError("AEFBFP: require 0 < sigma_scale < 1 for the :log_power rule, got $(alg.sigma_scale)"))
    end
    if alg.xi_rule === :power
        alg.xi_exp > 1 ||
            throw(ArgumentError("AEFBFP: xi_exp must be > 1 for Σξ_k < ∞, got $(alg.xi_exp)"))
    end

    state   = SolverState(:AEFBFP, x0)
    t0      = time()
    x_0     = copy(x0)
    z_prev  = copy(x0)              # z_{-1} = x_0
    state.f_evals = 0
    Bz_prev = algorithm_B!(state, prob.B, z_prev) # B(z_{-1}) — first f_eval
    τ_curr  = alg.tau_0
    state.step_size = τ_curr
    for cb in observers
        on_event!(cb, state, :init)
    end

    while true
        state.x_prev = copy(state.x)

        # Step 1: Extrapolation
        σ_k = _epcm_sigma(alg.sigma_rule, alg.sigma_scale, alg.sigma_exp, state.k)
        w_k = σ_k .* x_0 .+ (1 - σ_k) .* state.x

        # Step 2: Resolvent (uses cached Bz_prev = B(z_{k-1}))
        z_k = prob.resolvent_A(w_k .- τ_curr .* Bz_prev, τ_curr)
        Bz_k = algorithm_B!(state, prob.B, z_k)

        # Step 3: Forward correction = iterate update
        x_next = z_k .+ τ_curr .* (Bz_prev .- Bz_k)

        # Step 4: Min-rule stepsize (ξ_k = 1/(k+1)^xi_exp, summable, additive)
        Δz       = z_prev .- z_k
        ΔBz      = Bz_prev .- Bz_k
        norm_Δz  = norm(Δz)
        norm_ΔBz = norm(ΔBz)
        ξ_k      = _epcm_delta(alg.xi_rule, alg.xi_exp, state.k)
        τ_next = norm_ΔBz > 0 ?
            min(alg.mu * norm_Δz / norm_ΔBz, τ_curr + ξ_k) :
            τ_curr + ξ_k

        state.x         = x_next
        state.elapsed   = time() - t0
        state.step_size = τ_next

        # Residual monitoring (single-call: this B(x_next) is NOT an algorithm eval)
        Bxnext        = prob.B(x_next)
        rho           = τ_next
        prox_val      = prob.resolvent_A(x_next .- rho .* Bxnext, rho)
        Rn            = norm(x_next .- prox_val)
        state.residual        = Rn
        state.scaled_residual = rho > 0 ? Rn / rho : NaN

        state.k += 1
        for cb in observers
            on_event!(cb, state, :iter)
        end
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

        z_prev  = z_k
        Bz_prev = Bz_k
        τ_curr  = τ_next
    end

    for cb in observers
        on_event!(cb, state, :terminate)
    end
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
# MAEFBFP — Modified Adaptive Extrapolated Forward–Backward–Forward
# ============================================================================
#
# MAEFBFP modifies AEFBFP's min-rule factor μ to the iteration-dependent
# coefficient
#   α_k + β_k μ,
# where α_k = alpha_scale/(k+1)^alpha_exp → 0 and
# β_k = 1 + beta_scale/(k+1)^beta_exp → 1 with β_k ≥ 1. The stepsize becomes
#   τ_{k+1} = min{ (α_k + β_k μ)‖z_k-z_{k-1}‖ / ‖B(z_k)-B(z_{k-1})‖, τ_k+ξ_k }
#         or  τ_k + ξ_k
# in the zero-denominator case. Setting alpha_scale = beta_scale = 0 recovers
# AEFBFP exactly.

"""
    solve(alg::MAEFBFP, prob::TestProblem, x0::Vector{Float64};
          stopping::Tuple, observers::Tuple = ()) -> SolverResult

Run the modified AEFBFP iteration on `prob` from `x0`, replacing AEFBFP's
constant min-rule factor `μ` by `α_k + β_k μ`, with `α_k → 0`, `β_k → 1`,
`α_k ≥ 0`, `β_k ≥ 1`. The implementation enforces the sufficient uniform bound
`alpha_scale + (1 + beta_scale) * mu < 1/sqrt(6)` so the effective factor
stays below AEFBFP's admissibility ceiling at every iteration.
"""
function solve(alg::MAEFBFP, prob::TestProblem, x0::Vector{Float64};
               stopping::Tuple,
               observers::Tuple = ())

    bound_eff = 1 / sqrt(6)
    (0 < alg.mu < bound_eff) ||
        throw(ArgumentError("MAEFBFP: base μ must be in (0, 1/√6 ≈ 0.4082), got μ = $(alg.mu)."))
    alg.tau_0 > 0 ||
        throw(ArgumentError("MAEFBFP: τ_0 must be > 0, got $(alg.tau_0)"))
    alg.alpha_scale >= 0 ||
        throw(ArgumentError("MAEFBFP: alpha_scale must be ≥ 0, got $(alg.alpha_scale)"))
    alg.beta_scale >= 0 ||
        throw(ArgumentError("MAEFBFP: beta_scale must be ≥ 0, got $(alg.beta_scale)"))
    alg.alpha_exp > 0 ||
        throw(ArgumentError("MAEFBFP: alpha_exp must be > 0 so α_k → 0, got $(alg.alpha_exp)"))
    alg.beta_exp > 0 ||
        throw(ArgumentError("MAEFBFP: beta_exp must be > 0 so β_k → 1, got $(alg.beta_exp)"))
    eff0 = alg.alpha_scale + (1 + alg.beta_scale) * alg.mu
    eff0 < bound_eff ||
        throw(ArgumentError("MAEFBFP: require α_0 + β_0 μ = alpha_scale + (1+beta_scale)μ < 1/√6 ≈ $bound_eff, got $eff0"))
    if alg.sigma_rule === :power
        (0 < alg.sigma_exp <= 1) ||
            throw(ArgumentError("MAEFBFP: sigma_exp must be in (0, 1] for Σσ_k = ∞, got $(alg.sigma_exp)"))
        alg.sigma_scale > 0 ||
            throw(ArgumentError("MAEFBFP: sigma_scale must be > 0, got $(alg.sigma_scale)"))
        σ0 = alg.sigma_scale / 2.0^alg.sigma_exp
        σ0 < 1 ||
            throw(ArgumentError("MAEFBFP: require σ_0 = sigma_scale/2^sigma_exp < 1, got σ_0 = $σ0"))
    end
    if alg.xi_rule === :power
        alg.xi_exp > 1 ||
            throw(ArgumentError("MAEFBFP: xi_exp must be > 1 for Σξ_k < ∞, got $(alg.xi_exp)"))
    end

    state   = SolverState(:MAEFBFP, x0)
    t0      = time()
    x_0     = copy(x0)
    z_prev  = copy(x0)
    state.f_evals = 0
    Bz_prev = algorithm_B!(state, prob.B, z_prev)
    τ_curr  = alg.tau_0
    state.step_size = τ_curr
    for cb in observers
        on_event!(cb, state, :init)
    end

    while true
        state.x_prev = copy(state.x)

        σ_k = _epcm_sigma(alg.sigma_rule, alg.sigma_scale, alg.sigma_exp, state.k)
        w_k = σ_k .* x_0 .+ (1 - σ_k) .* state.x

        z_k = prob.resolvent_A(w_k .- τ_curr .* Bz_prev, τ_curr)
        Bz_k = algorithm_B!(state, prob.B, z_k)

        x_next = z_k .+ τ_curr .* (Bz_prev .- Bz_k)

        Δz       = z_prev .- z_k
        ΔBz      = Bz_prev .- Bz_k
        norm_Δz  = norm(Δz)
        norm_ΔBz = norm(ΔBz)
        ξ_k      = _epcm_delta(alg.xi_rule, alg.xi_exp, state.k)
        α_k      = _maefbfp_alpha(alg.alpha_scale, alg.alpha_exp, state.k)
        β_k      = _maefbfp_beta(alg.beta_scale, alg.beta_exp, state.k)
        eff_k    = α_k + β_k * alg.mu
        τ_next = norm_ΔBz > 0 ?
            min(eff_k * norm_Δz / norm_ΔBz, τ_curr + ξ_k) :
            τ_curr + ξ_k

        state.x         = x_next
        state.elapsed   = time() - t0
        state.step_size = τ_next

        Bxnext        = prob.B(x_next)
        rho           = τ_next
        prox_val      = prob.resolvent_A(x_next .- rho .* Bxnext, rho)
        Rn            = norm(x_next .- prox_val)
        state.residual        = Rn
        state.scaled_residual = rho > 0 ? Rn / rho : NaN

        state.k += 1
        for cb in observers
            on_event!(cb, state, :iter)
        end
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

        z_prev  = z_k
        Bz_prev = Bz_k
        τ_curr  = τ_next
    end

    for cb in observers
        on_event!(cb, state, :terminate)
    end
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
# VAFBS — Viscosity-Approximation Forward–Backward Splitting (Thong2019 Alg 3.1)
# ============================================================================
#
# Paper notation: A is the single-valued monotone Lipschitz operator, B is the
# maximal monotone operator. In this repo that maps to:
#   paper A ↔ prob.B
#   paper B ↔ prob.resolvent_A
#
# Iteration (n = 1, 2, ... in the paper; `n = state.k + 1` here):
#
#   y_n = J^B_{λ_n}(x_n - λ_n A x_n),
#         where λ_n is the largest element of {δ, δℓ, δℓ², ...} satisfying
#         λ_n⟨A x_n - A y_n, x_n - y_n⟩ ≤ μ‖x_n - y_n‖²
#   d_n = x_n - y_n - λ_n(A x_n - A y_n)
#   η_n = (1-μ)‖x_n - y_n‖² / ‖d_n‖²
#   z_n = x_n - γ η_n d_n
#   x_{n+1} = α_n f(x_n) + (1-α_n) z_n
#
# For this repo, the contraction is encoded as `f(x) = c x`, with `c` stored as
# `alg.f_scale`. This keeps the algorithm fully hashable and reproducible.
#
# B-eval accounting: cache A x_n = prob.B(x_n) across iterations; each rejected
# or accepted line-search trial evaluates A y_n once, and the universal residual
# at x_{n+1} evaluates A x_{n+1} once more (that value is cached for the next
# iteration). So the steady-state cost is `(# line-search trials) + 1` fresh
# `prob.B` evaluations per iteration, plus 1 preload `prob.B(x0)`.

"""
    solve(alg::VAFBS, prob::TestProblem, x0::Vector{Float64};
          stopping::Tuple, observers::Tuple = ()) -> SolverResult

Run the viscosity-approximation forward-backward splitting method
(Thong–Cholamjiak 2019, Algorithm 3.1) on `prob` from `x0`.
"""
function solve(alg::VAFBS, prob::TestProblem, x0::Vector{Float64};
               stopping::Tuple,
               observers::Tuple = ())

    alg.delta > 0 ||
        throw(ArgumentError("VAFBS: delta must be > 0, got $(alg.delta)"))
    (0 < alg.ell < 1) ||
        throw(ArgumentError("VAFBS: require 0 < ell < 1, got ell = $(alg.ell)"))
    (0 < alg.mu < 1) ||
        throw(ArgumentError("VAFBS: require 0 < mu < 1, got mu = $(alg.mu)"))
    (0 < alg.gamma < 2) ||
        throw(ArgumentError("VAFBS: require 0 < gamma < 2, got gamma = $(alg.gamma)"))
    if alg.alpha_rule === :c_over_n
        (0 < alg.alpha_scale < 1) ||
            throw(ArgumentError("VAFBS: alpha_scale must be in (0,1) for alpha_rule=:c_over_n, got $(alg.alpha_scale)"))
    end
    (0 <= alg.f_scale < 1) ||
        throw(ArgumentError("VAFBS: f_scale must be in [0,1), got $(alg.f_scale)"))

    state = SolverState(:VAFBS, x0)
    t0    = time()

    state.f_evals = 0
    Ax_curr = algorithm_B!(state, prob.B, state.x) # paper A(x_n); cached across iterations
    state.step_size = alg.delta

    for cb in observers
        on_event!(cb, state, :init)
    end

    while true
        x_n = state.x
        state.x_prev = copy(x_n)

        λ_n    = alg.delta
        y_n    = similar(x_n)
        Ay_n   = similar(x_n)
        lsiter = 0
        while true
            y_trial = prob.resolvent_A(x_n .- λ_n .* Ax_curr, λ_n)
            Ay_trial = algorithm_B!(state, prob.B, y_trial)

            diff_xy = x_n .- y_trial
            lhs = λ_n * dot(Ax_curr .- Ay_trial, diff_xy)
            rhs = alg.mu * sum(abs2, diff_xy)
            if lhs <= rhs + 1.0e-12 * max(1.0, rhs)
                y_n  = y_trial
                Ay_n = Ay_trial
                break
            end

            λ_n *= alg.ell
            lsiter += 1
            lsiter <= 200 ||
                error("VAFBS: line search did not accept within 200 reductions at iteration $(state.k + 1)")
        end

        diff_xy    = x_n .- y_n
        norm_xy_sq = sum(abs2, diff_xy)
        if norm_xy_sq == 0.0
            state.x               = copy(y_n)
            state.elapsed         = time() - t0
            state.step_size       = λ_n
            state.residual        = 0.0
            state.scaled_residual = 0.0
            state.k += 1
            for cb in observers
                on_event!(cb, state, :iter)
            end
            state.flag = :converged
            break
        end

        d_n       = diff_xy .- λ_n .* (Ax_curr .- Ay_n)
        norm_d_sq = sum(abs2, d_n)
        if norm_d_sq == 0.0
            state.x               = copy(x_n)
            state.elapsed         = time() - t0
            state.step_size       = λ_n
            state.residual        = 0.0
            state.scaled_residual = 0.0
            state.k += 1
            for cb in observers
                on_event!(cb, state, :iter)
            end
            state.flag = :converged
            break
        end

        η_n = (1 - alg.mu) * norm_xy_sq / norm_d_sq
        z_n = x_n .- (alg.gamma * η_n) .* d_n

        n   = state.k + 1
        α_n = _vafbs_alpha(alg.alpha_rule, alg.alpha_scale, n)
        x_next = (α_n * alg.f_scale) .* x_n .+ (1 - α_n) .* z_n

        state.x         = x_next
        state.elapsed   = time() - t0
        state.step_size = λ_n

        # This value is cached as A(x_n) for the next iteration, so it counts.
        Bxnext        = algorithm_B!(state, prob.B, x_next)
        rho           = λ_n
        prox_val      = prob.resolvent_A(x_next .- rho .* Bxnext, rho)
        Rn            = norm(x_next .- prox_val)
        state.residual        = Rn
        state.scaled_residual = rho > 0 ? Rn / rho : NaN

        state.k += 1
        for cb in observers
            on_event!(cb, state, :iter)
        end

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

        Ax_curr = Bxnext
    end

    for cb in observers
        on_event!(cb, state, :terminate)
    end

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
# MDITSM — Modified Tseng Splitting Method with Double Inertial Steps
#           (Wang2023 Algorithm 3.1)
# ============================================================================
#
# Paper notation: A is the single-valued monotone Lipschitz operator, B is the
# maximal monotone operator. In this repo that maps to:
#   paper A ↔ prob.B
#   paper B ↔ prob.resolvent_A
#
# Iteration (n = 1, 2, …; iterates x_{n−1}, x_n; current step λ_n):
#
#   w_n = x_n + α_n (x_n − x_{n−1})
#   z_n = x_n + β_n (x_n − x_{n−1})
#   y_n = J^B_{λ_n}(w_n − λ_n A w_n)
#   λ_{n+1} = min{ (μ_n+μ)‖w_n−y_n‖ / ‖A w_n−A y_n‖, λ_n + p_n }
#             if A w_n ≠ A y_n, else λ_n + p_n
#   x_{n+1} = (1−θ_n)z_n + θ_n(y_n − λ_n(A y_n − A w_n))
#
# Wang seeds x_0, x_1 arbitrarily; given one initial point, we set x_0 = x_1 = x0,
# so the first inertial displacement x_1 − x_0 vanishes. This matches the repo's
# existing inertial-method convention (cf. IMTTM, IFRAB) and avoids introducing a
# second caller-side initialization API.
#
# B-eval accounting: w_n changes every iteration, so B(w_n) cannot be cached
# across iterations. Per-iter ALGORITHM cost is therefore 2 fresh B-evals:
# B(w_n) and B(y_n). The extra B(x_{n+1}) used for the universal residual is
# monitoring only and is NOT counted in f_evals. No preload.

"""
    solve(alg::MDITSM, prob::TestProblem, x0::Vector{Float64};
          stopping::Tuple, observers::Tuple=()) -> SolverResult

Run Wang et al.'s modified Tseng splitting method with double inertial steps
(Algorithm 3.1) on `prob` from `x0`, using the repo convention x_0 = x_1 = x0.
Returns a `SolverResult`; per-iteration history is captured iff a
`HistoryCallback` is in `observers`.
"""
function solve(alg::MDITSM, prob::TestProblem, x0::Vector{Float64};
               stopping::Tuple,
               observers::Tuple = ())

    alg.lambda_1 > 0 ||
        throw(ArgumentError("MDITSM: lambda_1 must be > 0, got $(alg.lambda_1)"))
    (0 < alg.mu < 1) ||
        throw(ArgumentError("MDITSM: mu must be in (0, 1), got $(alg.mu)"))
    alg.alpha_base > 1 ||
        throw(ArgumentError("MDITSM: alpha_base must be > 1, got $(alg.alpha_base)"))
    (0 < alg.alpha_scale <= alg.alpha_base) ||
        throw(ArgumentError("MDITSM: require 0 < alpha_scale <= alpha_base; got alpha_scale=$(alg.alpha_scale), alpha_base=$(alg.alpha_base)"))
    (0 <= alg.beta_cap < 1) ||
        throw(ArgumentError("MDITSM: require 0 <= beta_cap < 1, got $(alg.beta_cap)"))
    alg.beta_shift > -1 ||
        throw(ArgumentError("MDITSM: beta_shift must satisfy shift > -1, got $(alg.beta_shift)"))
    (0 < alg.theta_cap <= 1) ||
        throw(ArgumentError("MDITSM: require 0 < theta_cap <= 1, got $(alg.theta_cap)"))
    alg.theta_shift > -1 ||
        throw(ArgumentError("MDITSM: theta_shift must satisfy shift > -1, got $(alg.theta_shift)"))
    if alg.mu_rule === :scale_over_n_pow
        alg.mu_scale >= 0 ||
            throw(ArgumentError("MDITSM: mu_scale must be >= 0, got $(alg.mu_scale)"))
        alg.mu_exp > 0 ||
            throw(ArgumentError("MDITSM: mu_exp must be > 0 for mu_rule=:scale_over_n_pow, got $(alg.mu_exp)"))
    end
    if alg.p_rule === :scale_over_n_pow
        alg.p_scale >= 0 ||
            throw(ArgumentError("MDITSM: p_scale must be >= 0, got $(alg.p_scale)"))
        alg.p_exp > 1 ||
            throw(ArgumentError("MDITSM: p_exp must be > 1 for p_rule=:scale_over_n_pow, got $(alg.p_exp)"))
    end
    α1 = _mditsm_alpha(alg.alpha_rule, alg.alpha_scale, alg.alpha_base, 1)
    β1 = _mditsm_beta(alg.beta_rule, alg.beta_cap, alg.beta_shift, 1)
    θ1 = _mditsm_theta(alg.theta_rule, alg.theta_cap, alg.theta_shift, 1)
    (0 <= α1 <= 1) ||
        throw(ArgumentError("MDITSM: alpha_1 must lie in [0, 1], got $(α1)"))
    (0 <= β1 < 1) ||
        throw(ArgumentError("MDITSM: beta_1 must lie in [0, 1), got $(β1)"))
    (0 < θ1 <= 1) ||
        throw(ArgumentError("MDITSM: theta_1 must lie in (0, 1], got $(θ1)"))

    state = SolverState(:MDITSM, x0)
    state.f_evals = 0
    t0    = time()

    x_older = copy(x0)       # x_0; state.x starts at x_1 = x0
    λ_curr  = alg.lambda_1
    state.step_size = λ_curr

    for cb in observers
        on_event!(cb, state, :init)
    end

    while true
        x_curr = state.x
        state.x_prev = copy(x_curr)

        n   = state.k + 1
        α_n = _mditsm_alpha(alg.alpha_rule, alg.alpha_scale, alg.alpha_base, n)
        β_n = _mditsm_beta(alg.beta_rule, alg.beta_cap, alg.beta_shift, n)
        θ_n = _mditsm_theta(alg.theta_rule, alg.theta_cap, alg.theta_shift, n)
        μ_n = _mditsm_mu_aux(alg.mu_rule, alg.mu_scale, alg.mu_exp, n)
        p_n = _mditsm_p(alg.p_rule, alg.p_scale, alg.p_exp, n)

        inertial = x_curr .- x_older
        w_n      = x_curr .+ α_n .* inertial
        z_n      = x_curr .+ β_n .* inertial

        Bw_n = algorithm_B!(state, prob.B, w_n)
        y_n  = prob.resolvent_A(w_n .- λ_curr .* Bw_n, λ_curr)
        By_n = algorithm_B!(state, prob.B, y_n)

        Δw      = w_n .- y_n
        norm_Δw = norm(Δw)
        if norm_Δw == 0.0
            state.x               = copy(y_n)
            state.elapsed         = time() - t0
            state.step_size       = λ_curr
            state.residual        = 0.0
            state.scaled_residual = 0.0
            state.k += 1
            for cb in observers
                on_event!(cb, state, :iter)
            end
            state.flag = :converged
            break
        end

        ΔB      = Bw_n .- By_n
        norm_ΔB = norm(ΔB)
        λ_next  = norm_ΔB > 0 ? min((alg.mu + μ_n) * norm_Δw / norm_ΔB, λ_curr + p_n) :
                                (λ_curr + p_n)

        correction = y_n .- λ_curr .* (By_n .- Bw_n)
        x_next     = (1 - θ_n) .* z_n .+ θ_n .* correction

        state.x         = x_next
        state.elapsed   = time() - t0
        state.step_size = λ_next

        # Diagnostic only: this is intentionally NOT part of F-evals.
        Bxnext        = prob.B(x_next)
        rho           = λ_next
        prox_val      = prob.resolvent_A(x_next .- rho .* Bxnext, rho)
        Rn            = norm(x_next .- prox_val)
        state.residual        = Rn
        state.scaled_residual = rho > 0 ? Rn / rho : NaN

        state.k += 1
        for cb in observers
            on_event!(cb, state, :iter)
        end

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

        x_older = x_curr
        λ_curr  = λ_next
    end

    for cb in observers
        on_event!(cb, state, :terminate)
    end

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
# MFRBSM - Modified Forward-Backward Splitting Method
#          (Hieu-Anh-Muu 2021 Algorithm 3.1)
# ============================================================================
#
# Iteration:
#   x_{n+1} = J^A_{\lambda_n}(x_n - \lambda_n B(x_n)
#                            - \lambda_{n-1}(B(x_n) - B(x_{n-1})))
#   \lambda_{n+1} = min{ \lambda_n,
#                        \mu ||x_{n+1} - x_n|| / ||B(x_{n+1}) - B(x_n)|| }
#
# The paper assumes x_{-1}, x_0 are given. With the repo's one-point solver API
# we use the standard inertial-method convention x_{-1} = x_0 = x0, so
# B(x_{-1}) = B(x_0) at initialization.
#
# B-eval accounting: cache B(x_n) and B(x_{n-1}) across iterations. After the
# initial preload B(x_0), the steady-state algorithm cost is 1 fresh B-eval/iter:
# B(x_{n+1}), which is reused both for the adaptive stepsize update and as the
# cached B(x_n) value of the next iteration.

function solve(alg::MFRBSM, prob::TestProblem, x0::Vector{Float64};
               stopping::Tuple,
               observers::Tuple = ())

    alg.lambda_minus1 > 0 ||
        throw(ArgumentError("MFRBSM: lambda_minus1 must be > 0, got $(alg.lambda_minus1)"))
    alg.lambda_0 > 0 ||
        throw(ArgumentError("MFRBSM: lambda_0 must be > 0, got $(alg.lambda_0)"))
    (0 < alg.mu < 0.5) ||
        throw(ArgumentError("MFRBSM: mu must be in (0, 0.5), got $(alg.mu)"))

    state = SolverState(:MFRBSM, x0)
    t0    = time()

    Bx_curr       = prob.B(state.x)
    Bx_prev       = copy(Bx_curr)   # x_{-1} = x_0 = x0
    state.f_evals = 1
    lambda_prev   = alg.lambda_minus1
    lambda_curr   = alg.lambda_0
    state.step_size = lambda_curr

    for cb in observers
        on_event!(cb, state, :init)
    end

    while true
        x_curr = state.x
        state.x_prev = copy(x_curr)

        x_next = prob.resolvent_A(
            x_curr .- lambda_curr .* Bx_curr .- lambda_prev .* (Bx_curr .- Bx_prev),
            lambda_curr,
        )

        Bx_next = prob.B(x_next)
        state.f_evals += 1

        delta_B      = Bx_next .- Bx_curr
        norm_delta_B = norm(delta_B)
        norm_step    = norm(x_next .- x_curr)
        lambda_next  = norm_delta_B > 0 ?
                       min(lambda_curr, alg.mu * norm_step / norm_delta_B) :
                       lambda_curr

        state.x         = x_next
        state.elapsed   = time() - t0
        state.step_size = lambda_next

        rho           = lambda_next
        prox_val      = prob.resolvent_A(x_next .- rho .* Bx_next, rho)
        Rn            = norm(x_next .- prox_val)
        state.residual        = Rn
        state.scaled_residual = rho > 0 ? Rn / rho : NaN

        state.k += 1
        for cb in observers
            on_event!(cb, state, :iter)
        end

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

        Bx_prev     = Bx_curr
        Bx_curr     = Bx_next
        lambda_prev = lambda_curr
        lambda_curr = lambda_next
    end

    for cb in observers
        on_event!(cb, state, :terminate)
    end

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
# ============================================================================
# RFBSM — Relaxed Forward–Backward Splitting Method (Cholamjiak2021 Algorithm 1)
# ============================================================================
#
# Paper notation matches the repo convention directly: A is the set-valued
# maximal monotone operator and B is the single-valued monotone Lipschitz
# operator.
#
# Iteration:
#   y_n = J^A_{λ_n}(x_n − λ_n B(x_n))
#   x_{n+1} = (1−θ)x_n + θ y_n + θ λ_n (B(x_n) − B(y_n))
#   λ_{n+1} = min{ λ_n, μ ||x_n−y_n|| / ||B(x_n)−B(y_n)|| }
#
# B-eval accounting: cache B(x_n) across iterations, so the steady-state cost
# is 2 B-evals/iter: B(y_n) plus B(x_{n+1}) for the residual block (reused next
# iteration). This mirrors MTTM's accounting.

function solve(alg::RFBSM, prob::TestProblem, x0::Vector{Float64};
               stopping::Tuple,
               observers::Tuple = ())

    alg.lambda_0 > 0 ||
        throw(ArgumentError("RFBSM: lambda_0 must be > 0, got $(alg.lambda_0)"))
    (0 < alg.theta <= 1) ||
        throw(ArgumentError("RFBSM: theta must be in (0, 1], got $(alg.theta)"))
    (0 < alg.mu < 1) ||
        throw(ArgumentError("RFBSM: mu must be in (0, 1), got $(alg.mu)"))

    state = SolverState(:RFBSM, x0)
    t0    = time()

    state.f_evals = 0
    Bx_curr       = algorithm_B!(state, prob.B, state.x)
    λ_curr          = alg.lambda_0
    state.step_size = λ_curr

    for cb in observers
        on_event!(cb, state, :init)
    end

    while true
        x_curr = state.x
        state.x_prev = copy(x_curr)

        y_n  = prob.resolvent_A(x_curr .- λ_curr .* Bx_curr, λ_curr)
        By_n = algorithm_B!(state, prob.B, y_n)

        x_next = (1 - alg.theta) .* x_curr .+ alg.theta .* y_n .+
                 (alg.theta * λ_curr) .* (Bx_curr .- By_n)

        ΔB      = Bx_curr .- By_n
        norm_ΔB = norm(ΔB)
        λ_next  = norm_ΔB > 0 ? min(λ_curr, alg.mu * norm(x_curr .- y_n) / norm_ΔB) : λ_curr

        state.x         = x_next
        state.elapsed   = time() - t0
        state.step_size = λ_next

        Bxnext        = algorithm_B!(state, prob.B, x_next)
        rho           = λ_next
        prox_val      = prob.resolvent_A(x_next .- rho .* Bxnext, rho)
        Rn            = norm(x_next .- prox_val)
        state.residual        = Rn
        state.scaled_residual = rho > 0 ? Rn / rho : NaN

        state.k += 1
        for cb in observers
            on_event!(cb, state, :iter)
        end

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

        Bx_curr = Bxnext
        λ_curr  = λ_next
    end

    for cb in observers
        on_event!(cb, state, :terminate)
    end

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
# IRFBSM — Inertial Relaxed Forward–Backward Splitting Method
#           (Cholamjiak2021 Algorithm 2)
# ============================================================================
#
# Iteration:
#   w_n = x_n + α(x_n − x_{n−1})
#   y_n = J^A_{λ_n}(w_n − λ_n B(w_n))
#   x_{n+1} = (1−θ)w_n + θ y_n + θ λ_n (B(w_n) − B(y_n))
#   λ_{n+1} = min{ λ_n, μ ||w_n−y_n|| / ||B(w_n)−B(y_n)|| }
#
# The paper assumes x_{-1}, x_0 are given. With the repo's one-point solver API
# we set x_{-1} = x_0 = x0, exactly as we do for the other inertial baselines.
#
# B-eval accounting: w_n changes every iteration, so B(w_n) cannot be cached.
# Per-iter ALGORITHM cost is 2 fresh B-evals: B(w_n) and B(y_n). B(x_{n+1}) in
# the residual block is monitoring only and is not counted in f_evals.

function solve(alg::IRFBSM, prob::TestProblem, x0::Vector{Float64};
               stopping::Tuple,
               observers::Tuple = ())

    alg.lambda_0 > 0 ||
        throw(ArgumentError("IRFBSM: lambda_0 must be > 0, got $(alg.lambda_0)"))
    (0 < alg.theta <= 1) ||
        throw(ArgumentError("IRFBSM: theta must be in (0, 1], got $(alg.theta)"))
    (0 < alg.mu < 1) ||
        throw(ArgumentError("IRFBSM: mu must be in (0, 1), got $(alg.mu)"))
    (0 <= alg.alpha < 1) ||
        throw(ArgumentError("IRFBSM: alpha must be in [0, 1), got $(alg.alpha)"))
    K = alg.theta * (1 - alg.mu^2) / (2 - alg.theta + alg.mu * alg.theta)^2 + (1 - alg.theta) / alg.theta
    lhs_alpha = alg.alpha * (1 + alg.alpha) / (1 - alg.alpha)^2
    lhs_alpha < K ||
        throw(ArgumentError("IRFBSM: require alpha(1+alpha)/(1-alpha)^2 < K = $K, got lhs = $(lhs_alpha)"))

    state = SolverState(:IRFBSM, x0)
    state.f_evals = 0
    t0    = time()

    x_older = copy(x0)    # x_{-1} = x_0 = x0
    λ_curr  = alg.lambda_0
    state.step_size = λ_curr

    for cb in observers
        on_event!(cb, state, :init)
    end

    while true
        x_curr = state.x
        state.x_prev = copy(x_curr)

        w_n  = x_curr .+ alg.alpha .* (x_curr .- x_older)
        Bw_n = algorithm_B!(state, prob.B, w_n)
        y_n  = prob.resolvent_A(w_n .- λ_curr .* Bw_n, λ_curr)
        By_n = algorithm_B!(state, prob.B, y_n)

        x_next = (1 - alg.theta) .* w_n .+ alg.theta .* y_n .+
                 (alg.theta * λ_curr) .* (Bw_n .- By_n)

        ΔB      = Bw_n .- By_n
        norm_ΔB = norm(ΔB)
        λ_next  = norm_ΔB > 0 ? min(λ_curr, alg.mu * norm(w_n .- y_n) / norm_ΔB) : λ_curr

        state.x         = x_next
        state.elapsed   = time() - t0
        state.step_size = λ_next

        # Diagnostic only: this is intentionally NOT part of F-evals.
        Bxnext        = prob.B(x_next)
        rho           = λ_next
        prox_val      = prob.resolvent_A(x_next .- rho .* Bxnext, rho)
        Rn            = norm(x_next .- prox_val)
        state.residual        = Rn
        state.scaled_residual = rho > 0 ? Rn / rho : NaN

        state.k += 1
        for cb in observers
            on_event!(cb, state, :iter)
        end

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

        x_older = x_curr
        λ_curr  = λ_next
    end

    for cb in observers
        on_event!(cb, state, :terminate)
    end

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

# MTTM â€” Mann Tseng-Type Method (Gibali2018 Algorithm 1)

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
# be cached across iters (unlike MTTM's B(x_n)). Per-iter ALGORITHM cost:
# B(w_n) + B(y_n) = 2 B-evals/iter (two-call). The extra B(x_{n+1}) for the
# universal residual is monitoring only and is NOT counted in f_evals. No preload.

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

        # Universal residual at x_{n+1} with ρ = λ_{n+1}. B(x_{n+1}) is for R_n
        # monitoring only — not algorithm work — so it does NOT increment f_evals.
        Bxnext        = prob.B(x_next)
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
    state.f_evals = 0
    Bw_curr       = algorithm_B!(state, prob.B, state.x)
    Bw_prev       = Bw_curr           # alias safe (never mutated in place)

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
        Bw_next       = algorithm_B!(state, prob.B, w_next)

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
