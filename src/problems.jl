# problems.jl — Test-problem definitions for the EPCMMIP benchmark.
#
# Defines the `InitialPoint` and parametric `TestProblem{B,R,Res,Meta}` structs,
# the five problem builders (P1-P5), the public dispatcher `get_problem`,
# and the deterministic seed combinator used to generate initial points.
#
# Design references:
#  - script_port_plan.md §4 Q3, Q5, Q6a, §5.1, §5.6, §5.6a
#  - plan_review_findings.md (B3 P3-rank-1 verification, B5 deterministic seed)
#
# Notation: A is set-valued (resolvent via `resolvent_A(x, ρ)`); B is
# single-valued and stored in `prob.B`. The inclusion is 0 ∈ A(x) + B(x).
# `resolvent_A`'s `ρ` argument is honoured by P2 and P4 (soft-thresholding);
# P1 (Volterra SFP half-space proj) and P3 (box) use normal-cone resolvents that
# are `ρ`-independent. Suite (revised 2026-05-28): P1 = Tan Ex5.2 Volterra SFP,
# P2 = Yao Ex4.2 ℓ1+quad, P3 = Izuchukwu Ex6.2 control, P4 = Tan Ex5.3 LASSO,
# P5 = ℓ₂ monotone inclusion with A = 2I and B = x₊.
#
# `problems.jl` is included AFTER `resolvents.jl` (`clipping_box`,
# `soft_thresholding`) and BEFORE `algorithm_types.jl`. It does NOT know
# about `AbstractAlgorithm` — problems are pure problem definitions.

# ============================================================================
# InitialPoint
# ============================================================================

"""
    InitialPoint(label::String, seed_idx::Int, x0::Vector{Float64})

A labelled initial iterate for a benchmark run. The benchmark loop writes:
- `label`     → `init_point` column in the DB.
- `seed_idx`  → `seed_idx` column in the DB.
- `x0`        → passed to the solver (copy before mutation).

`seed_idx` is the integer that fed the deterministic seed combinator that
generated `x0`. It is the canonical key; `label` is a human-readable view
(typically `"seed\$seed_idx"`).
"""
struct InitialPoint
    label::String
    seed_idx::Int
    x0::Vector{Float64}
end

# ============================================================================
# TestProblem
# ============================================================================

"""
    TestProblem{B,R,Res,Meta}

A test problem for the monotone-inclusion benchmark `0 ∈ A(x) + B(x)`.
Parametric on the concrete types of the functor fields so solver methods
specialise per problem and dispatch is fully concrete in the hot loop.

Fields:
- `id::Int`           — problem ID (matches `PROBLEM_IDS`).
- `name::String`      — descriptive, e.g. `"Tan2022a_Ex5.1"`.
- `dim::Int`          — problem dimension.
- `B::B`              — closwhat ure `x → Bx` (single-valued operator).
- `resolvent_A::R`    — closure `(x, ρ) → J^A_ρ(x)`. `ρ` argument is ignored
                         for P1 and P3 (normal-cone resolvents are `ρ`-indep).
- `native_residual::Res` — closure `(x, x_prev) → Float64` (each problem's
                         source-paper criterion; D1 decision 2026-05-28):
                         * P1 (Volterra SFP): `‖(I−P_C)x‖² + ‖B(x)‖²` (Tan E_n)
                         * P2: `‖x − x_prev‖` (Inf if `x_prev` empty)
                         * P3: `0.5 ‖z − clip_{[-1,1]}(z − B(z))‖²`
                         * P4 (LASSO): `‖x − x_prev‖` (Inf if `x_prev` empty)
                         * P5 (ℓ₂ example): `‖x − J^A_1(x − B(x))‖`
- `exact_x::Union{Vector{Float64},Nothing}` — known optimum if available
                         (P1: zeros; P3: bang-bang; P4: planted signal; P2: nothing).
- `initial_points::Vector{InitialPoint}` — pre-generated initial iterates.
- `metadata::Meta`    — NamedTuple of problem-specific extras
                         (P1: `(L, N, h, t)`; P2: `(b, L)`; P3: `(c, L, K, h, M_factor)`;
                          P4: `(C, y, gamma, L, M, k, x_star)`;
                          P5: `(L, N)`).

**Invariant**: `prob.metadata.L::Float64` is present on all builders
and is the canonical Lipschitz constant of `B`. Algorithm types in Step 4.5
that need an `L` estimate for default step-size rules read it here.
"""
struct TestProblem{B,R,Res,Meta}
    id::Int
    name::String
    dim::Int
    B::B
    resolvent_A::R
    native_residual::Res
    exact_x::Union{Vector{Float64},Nothing}
    initial_points::Vector{InitialPoint}
    metadata::Meta
end

"List of supported problem IDs. Matches the manuscript Numerical-Experiments section.
Suite revised 2026-05-28: P1=Volterra SFP (Tan Ex5.2), P2=ℓ1+quad (Yao Ex4.2),
P3=optimal control (Izuchukwu Ex6.2), P4=LASSO (Tan Ex5.3). The old box-VIP
(Tan Ex5.1) was dropped — EPCM's ill-conditioned worst case (see plan_review_findings D2)."
const PROBLEM_IDS = [1, 2, 3, 4, 5]

# ============================================================================
# Deterministic seed combinator
# ============================================================================

"""
    _make_rng(pid::Int, dim::Int, seed_idx::Int) -> Xoshiro

Deterministic per-(problem, dim, seed_idx) RNG. **Stable across Julia
versions** — does not use `Base.hash` (which is not stability-guaranteed).
Reserve `seed_idx = 0` for problem-data generation (P1's random `G`,
P2's random `b`) so it does not collide with initial-point seeds `1, 2, …`.

**Slot-width assumption**: the formula `pid·10⁶ + dim·10³ + seed_idx`
requires `dim < 1000` and `seed_idx < 1000` to avoid collisions across
slots. Both are comfortably inside the current envelope (max `dim = 900`,
max `seed_idx = 20`). If you ever sweep `dim ≥ 1000` or run more than
~999 seeds per cell, bump the multipliers (e.g. `10⁹` and `10⁶`); still
well inside UInt64.
"""
function _make_rng(pid::Int, dim::Int, seed_idx::Int)
    seed = UInt64(pid) * UInt64(1_000_000) +
           UInt64(dim) * UInt64(1_000) +
           UInt64(seed_idx)
    return Xoshiro(seed)
end

# ============================================================================
# Problem 1: Tan2022a Example 5.2lekin jab hum  — Volterra split-feasibility in L²([0,1])
# ============================================================================
#
# Find x ∈ C with Tx ∈ Q, where
#   C = {x : ∫₀¹ x ≤ 1}                       (half-space)
#   Q = {x : ∫₀¹|x − sin t|² ≤ 16}            (ball, centre sin, radius 4)
#   T = Volterra integration (Tx)(t)=∫₀ᵗ x,  ‖T‖ = 2/π,  adjoint (T*x)(t)=∫ₜ¹ x.
# x* = 0 is a solution.
#
# Solved as the monotone inclusion 0 ∈ A(x) + B(x):
#   A = N_C            (resolvent J^A = P_C, ρ-INDEPENDENT)
#   B(x) = T*(I − P_Q) T x   (∇ of ½‖(I−P_Q)Tx‖²; ONLY-monotone, cocoercive,
#                             Lipschitz L = ‖T‖² = (2/π)² ≈ 0.405)
#
# This problem showcases EPCM's design: it is infinite-dimensional (so *strong*
# convergence matters) and ‖T‖ is not readily available (so the L-free adaptive
# step is the right tool), yet it is only-monotone and well-conditioned.
#
# DISCRETIZATION (quadrature-weighted; decision 2026-05-28): N-point midpoint
# grid t_i=(i−½)/N, h=1/N. Work in ξ_i = √h·x(t_i) so the solvers' EUCLIDEAN
# inner product ⟨ξ,η⟩=Σξη ≈ ∫xy = ⟨x,y⟩_{L²}. Then everything is clean Euclidean
# ℝ^N:  T̃ξ = h·cumsum(ξ),  T̃ᵀv = h·revcumsum(v),  ‖T̃‖₂ → 2/π;  P_Q = Euclidean
# ball proj about s=√h·sin(t); P_C = half-space proj with normal √h·𝟙 (‖·‖²=hN=1).
# Native (Tan E_n): ‖(I−P_C)ξ‖² + ‖B(ξ)‖² < 1e-5.

"""
    _build_problem1(dim::Int; n_seeds::Int=10) -> TestProblem

Build P1 (Tan2022a Ex 5.2, Volterra split-feasibility) discretized to a grid of
`N = dim` points in quadrature-weighted coordinates ξ = √h·x. `n_seeds` seeded
random initial functions. x* = 0.
"""
function _build_problem1(dim::Int; n_seeds::Int=10)
    dim >= 2 || throw(ArgumentError("_build_problem1 (Volterra SFP): dim (grid N) must be >= 2, got $dim"))
    n_seeds >= 1 || throw(ArgumentError("_build_problem1: n_seeds must be >= 1, got $n_seeds"))

    N   = dim
    h   = 1.0 / N
    sqh = sqrt(h)
    t   = [(i - 0.5) / N for i in 1:N]          # midpoint grid
    s   = sqh .* sin.(t)                         # √h·sin(t): centre of Q in ξ-coords

    # Volterra T̃ and its Euclidean adjoint, as cumulative sums × h.
    T_apply  = v -> h .* cumsum(v)                              # T̃ v
    Tt_apply = v -> h .* reverse(cumsum(reverse(v)))           # T̃ᵀ v = h·revcumsum(v)

    # P_Q: projection onto the Euclidean ball of radius 4 centred at s.
    P_Q = let s = s
        w -> begin
            d  = w .- s
            nd = norm(d)
            nd > 4.0 ? (s .+ (4.0 / nd) .* d) : copy(w)
        end
    end

    # B(ξ) = T̃ᵀ (I − P_Q)(T̃ ξ)
    B_fn = let T_apply = T_apply, Tt_apply = Tt_apply, P_Q = P_Q
        ξ -> begin
            Tξ = T_apply(ξ)
            return Tt_apply(Tξ .- P_Q(Tξ))
        end
    end

    # P_C: projection onto {ξ : √h·Σξ ≤ 1} (normal √h·𝟙, ‖normal‖² = hN = 1).
    # ρ-independent (normal-cone resolvent).
    resolvent_A_fn = let sqh = sqh
        (ξ, ρ) -> begin
            a = sqh * sum(ξ)                     # = ∫₀¹ x
            return a > 1.0 ? (ξ .- (a - 1.0) * sqh) : copy(ξ)
        end
    end

    # Native E_n (Tan Ex 5.2): ‖(I − P_C)ξ‖² + ‖B(ξ)‖².
    native_res_fn = let resolvent_A_fn = resolvent_A_fn, B_fn = B_fn
        (ξ, ξ_prev) -> sum(abs2, ξ .- resolvent_A_fn(ξ, 1.0)) + sum(abs2, B_fn(ξ))
    end

    L = (2.0 / π)^2                              # continuous ‖T‖² (adaptive methods don't use it)

    # Initial points: seeded random SMOOTH functions at Tan's scale (‖x0‖_{L²} ≈
    # 300), x0(t) = c·sin(ω t + φ). Smoothness matters: the Volterra integral
    # shrinks white noise (‖T̃·noise‖ ~ N^{-1/2}) so small/noisy starts make the
    # SFP trivially feasible (T̃x0 ∈ Q, E_n≈0 in one step); smooth large starts
    # keep ‖T̃x0‖ ≈ c/ω ≫ 4 (outside Q) at every grid size N — genuinely needing
    # iterations, matching Tan's `x_1 = 600 sin(t)`-style initials.
    initial_points = InitialPoint[]
    for n in 1:n_seeds
        rng = _make_rng(1, dim, n)
        c   = 300.0 + 300.0 * rand(rng)          # amplitude (‖x0‖ ≈ c/√2 ≈ 210–420)
        ω   = 1.0 + 9.0 * rand(rng)              # frequency
        φ   = 2π * rand(rng)                      # phase
        ξ0  = sqh .* (c .* sin.(ω .* t .+ φ))    # quadrature-weighted smooth initial
        push!(initial_points, InitialPoint("seed$n", n, ξ0))
    end

    metadata = (L = L, N = N, h = h, t = t)

    return TestProblem(
        1, "Tan2022a_Ex5.2_VolterraSFP", N,
        B_fn, resolvent_A_fn, native_res_fn,
        zeros(N),                                # x* = 0 (ξ* = 0)
        initial_points,
        metadata,
    )
end

# ============================================================================
# Problem 2: Yao2024 Example 4.2 — ℓ₁ + quadratic minimisation
# ============================================================================
#
# Minimise φ(x) = ‖x‖_2² + b'x + 3 + ‖x‖_1 over ℝ^m.
# Inclusion: 0 ∈ ∂‖x‖_1 + (2x + b)
#   A = ∂‖·‖_1                    (resolvent = soft-thresholding by ρ)
#   B(x) = 2x + b                 (Lipschitz with L = 2)
#
# `b` is a fresh Gaussian draw per (problem, dim). No closed-form x*.

"""
    _build_problem2(dim::Int; n_seeds::Int=10) -> TestProblem

Build P2 at dimension `m = dim`.

Generates `b ~ N(0, I)` from the per-(2, dim, 0) RNG reproducibly, and
generates `n_seeds` initial points sampled from `N(0, I_m)`.

`n_seeds` defaults to 10.
"""
function _build_problem2(dim::Int; n_seeds::Int=10)

    # --- Basic input checks ---
    dim >= 1 || throw(ArgumentError("_build_problem2: dim must be >= 1, got $dim"))
    n_seeds >= 1 || throw(ArgumentError("_build_problem2: n_seeds must be >= 1, got $n_seeds"))

    # --- Data: b ~ N(0, I) ---
    data_rng = _make_rng(2, dim, 0)
    b = randn(data_rng, dim)

    # Lipschitz constant of B(x) = 2x + b
    L = 2.0

    # --- Operators ---
    B_fn = let b = b
        x -> 2.0 .* x .+ b
    end

    resolvent_A_fn = (x, ρ) -> soft_thresholding(x, ρ)

    # Native residual e_n = ‖x_{n+1} - x_n‖.
    # At initialization, x_prev may be empty, so return Inf as a safe sentinel.
    native_res_fn = (x, x_prev) -> isempty(x_prev) ? Inf : norm(x .- x_prev)

    # --- Initial points: n_seeds Gaussian draws ---
    initial_points = InitialPoint[]

    for n in 1:n_seeds
        rng = _make_rng(2, dim, n)
        x0 = randn(rng, dim)
        push!(initial_points, InitialPoint("seed$n", n, x0))
    end

    # --- Metadata ---
    metadata = (b = b, L = L)

    return TestProblem(
        2,
        "Yao2024_Ex4.2",
        dim,
        B_fn,
        resolvent_A_fn,
        native_res_fn,
        nothing,          # no closed-form x*
        initial_points,
        metadata,
    )
end

# ============================================================================
# Problem 3: Izuchukwu2023 Example 6.2 — optimal-control problem
# ============================================================================
#
# Control-only iterate z ∈ ℝ^K, where K = number of Euler nodes on t ∈ [0,2]
# (the problem DIMENSION; default 100), h = 2/K. Refining the mesh (larger K)
# discretises the SAME continuous control problem at higher resolution.
# Inclusion form 0 ∈ A(z) + B(z) with:
#   - A = N_{[-1,1]^K}            (resolvent = clip to [-1, 1], ρ-indep)
#   - B(z) = M z + c              (affine, per script_port_plan.md §5.6)
#
# M has all entries 2h (rank-1, ‖M‖_2 = 2hK = 4 for ALL K — Lipschitz is
# K-INVARIANT, so step sizes stay comparable across dimensions).
# c_j = -h(K - j) for j = 0, …, K-1 (0-indexed; using the manuscript's
# continuous-co-state convention — see plan note for the alternative
# direct-chain-rule form; both give the same bang-bang exact solution).
#
# Implementation: M is NOT stored as a dense K×K matrix. The closure
# computes `Mz` as `(2h · sum(z)) .* 1`, a rank-1 product, then adds c.
# Memory: stores only the K-vector c (plus the scalar 2h).
#
# Exact bang-bang: z*_j = +1 for j < round(0.6K), −1 after (switch at
# continuous-time t = 1.2 ⟹ node index 0.6K; = 60 at K=100).
#
# NOTE on the native residual Tol_n = 0.5‖z−clip(z−Bz)‖²: it is an UN-normalised
# squared sum over K nodes, so at larger K the fixed 1e-6 bar is effectively
# stricter per-node ⟹ iteration counts rise with K. This keeps the K=100 cell
# identical to the source; divide by K if a K-invariant bar is wanted instead.

"""
    _build_problem3(dim::Int = 0; n_seeds::Int=20) -> TestProblem

Build P3 (optimal control) at the `K = dim` Euler discretisation (`dim ≤ 0`
defaults to the source `K = 100`; `K ≥ 10` required). Affine `B(z) = Mz + c` is
implemented via a rank-1 closure (no dense matrix); `n_seeds` initial points
sampled uniformly on `[-1, 1]^K`. `K` IS the problem dimension — refining the
time mesh raises it (same continuous control problem at higher resolution).
"""
function _build_problem3(dim::Int = 0; n_seeds::Int=20)
    n_seeds >= 1 || throw(ArgumentError("_build_problem3: n_seeds must be >= 1, got $n_seeds"))

    K = dim <= 0 ? 100 : dim                                   # K = Euler nodes (dim); default 100
    K >= 10 || throw(ArgumentError("_build_problem3: K (dim) must be >= 10, got $K"))
    h = 2.0 / K                                                # mesh = 0.02 at K=100
    dim = K

    # --- Affine B(z) = M z + c, rank-1 M, c per manuscript convention ---
    c = Float64[-h * (K - j) for j in 0:K-1]                   # length K
    L = 2.0 * h * K                                            # ‖M‖_2 = 2hK = 4
    M_factor = 2.0 * h                                         # M = M_factor · 1·1ᵀ

    # B closure: efficient rank-1 + offset (no dense M)
    B_fn = let M_factor = M_factor, c = c
        z -> begin
            coeff = M_factor * sum(z)
            return coeff .+ c
        end
    end

    resolvent_A_fn = (z, ρ) -> clipping_box(z, -1.0, 1.0)

    # Native residual: Tol_n = 0.5 ‖z − clip_{[-1,1]}(z − B(z))‖²  (Izuchukwu, ρ = 1)
    native_res_fn = let B_fn = B_fn
        (z, z_prev) -> begin
            Bz = B_fn(z)
            return 0.5 * sum(abs2, z .- clipping_box(z .- Bz, -1.0, 1.0))
        end
    end

    # Exact bang-bang: switch at t=1.2 ⟹ node index 0.6K (= 60 at K=100)
    switch = round(Int, 0.6 * K)
    exact_z = vcat(fill(1.0, switch), fill(-1.0, K - switch))

    # --- Initial points: n_seeds uniform draws on [-1, 1]^K (Q5 post-review) ---
    initial_points = InitialPoint[]
    for n in 1:n_seeds
        rng = _make_rng(3, dim, n)
        z0 = -1.0 .+ 2.0 .* rand(rng, dim)
        push!(initial_points, InitialPoint("seed$n", n, z0))
    end

    metadata = (c = c, L = L, K = K, h = h, M_factor = M_factor)

    return TestProblem(
        3, "Izuchukwu2023_Ex6.2", dim,
        B_fn, resolvent_A_fn, native_res_fn,
        exact_z,
        initial_points,
        metadata,
    )
end

# ============================================================================
# Problem 4: Tan2022a Example 5.3 — LASSO / compressed sensing
# ============================================================================
#
# Recover a k-sparse signal x* (±1 spikes) from y = C x* + ε, C ∈ ℝ^{M×N} with
# M < N (UNDERDETERMINED), rows orthonormalized ⟹ ‖C‖₂ = 1; noise var 1e-4.
#
# Unconstrained LASSO  min_x ½‖Cx − y‖² + γ‖x‖₁  as 0 ∈ A(x) + B(x):
#   A = ∂(γ‖·‖₁)         (resolvent = soft-threshold by ρ·γ; ρ-DEPENDENT, set-valued)
#   B(x) = Cᵀ(C x − y)   (forward; ONLY-monotone — CᵀC singular since M<N —
#                         cocoercive, Lipschitz L = ‖C‖² = 1)
#
# Controlled contrast with P2 (same soft-threshold resolvent, but B there is
# 2x+b → strongly monotone; here B is only monotone). Native stopping:
# ‖x_{k+1}−x_k‖ < 1e-6 (Cauchy; additive noise floors out ‖x−x*‖). The planted
# x* is stored in metadata for MSE reporting only.

"""
    _build_problem4(dim::Int; n_seeds::Int=10, gamma::Float64=1.0e-3) -> TestProblem

Build P4 (Tan2022a Ex 5.3, LASSO) at signal length `N = dim`, with `M = N÷2`
measurements (underdetermined) and `k ≈ N/20` sparsity. `C` has orthonormalized
rows (‖C‖=1); planted ±1 signal x*; `y = Cx* + ε`, noise var 1e-4. The
regularization `gamma` is a problem constant (default 1e-3 — light: the
soft-threshold is by ρ·γ ≈ γ since L=1 ⟹ ρ=O(1), so γ≈1e-3 denoises the
ε≈1e-2 noise without erasing the ±1 spikes; Tan's Cor 4.4 quotes γ=1 for a
general statement, not this signal, and would threshold the spikes away).
"""
function _build_problem4(dim::Int; n_seeds::Int=10, gamma::Float64=1.0e-3)
    dim >= 4 || throw(ArgumentError("_build_problem4 (LASSO): dim (signal length N) must be >= 4, got $dim"))
    n_seeds >= 1 || throw(ArgumentError("_build_problem4: n_seeds must be >= 1, got $n_seeds"))
    gamma > 0 || throw(ArgumentError("_build_problem4: gamma must be > 0, got $gamma"))

    N = dim
    M = max(2, N ÷ 2)                            # underdetermined: M < N
    k = max(1, round(Int, N / 20))               # ~5% sparse

    # --- Data (reproducible from the per-(4, dim, 0) RNG) ---
    data_rng = _make_rng(4, dim, 0)
    C0 = randn(data_rng, M, N)
    # Orthonormalize ROWS of C: take the thin Q of C0' (N×M) ⟹ C = Qᵀ has C Cᵀ = I_M, ‖C‖₂ = 1.
    Qthin = Matrix(qr(C0').Q)[:, 1:M]            # N×M, orthonormal columns
    C = collect(Qthin')                          # M×N, orthonormal rows
    # planted k-sparse ±1 signal
    x_star = zeros(N)
    support = randperm(data_rng, N)[1:k]
    x_star[support] .= rand(data_rng, (-1.0, 1.0), k)
    ε  = sqrt(1.0e-4) .* randn(data_rng, M)      # noise std = 1e-2
    y  = C * x_star .+ ε

    L = 1.0                                      # ‖C‖² = 1 (orthonormal rows)

    # --- Operators ---
    B_fn           = let C = C, y = y;   x -> C' * (C * x .- y);   end
    resolvent_A_fn = let γ = gamma;      (x, ρ) -> soft_thresholding(x, ρ * γ);   end
    # Native = ‖x_{k+1} − x_k‖ (Cauchy); Inf sentinel at :init (x_prev empty).
    native_res_fn  = (x, x_prev) -> isempty(x_prev) ? Inf : norm(x .- x_prev)

    # --- Initial points: small seeded random starts ---
    initial_points = InitialPoint[]
    for n in 1:n_seeds
        rng = _make_rng(4, dim, n)
        x0 = 0.1 .* randn(rng, N)
        push!(initial_points, InitialPoint("seed$n", n, x0))
    end

    metadata = (C = C, y = y, gamma = gamma, L = L, M = M, k = k, x_star = x_star)

    return TestProblem(
        4, "Tan2022a_Ex5.3_LASSO", N,
        B_fn, resolvent_A_fn, native_res_fn,
        x_star,                                  # planted signal (noise floor on ‖x−x*‖)
        initial_points,
        metadata,
    )
end



# ============================================================================
# Problem 5: ℓ₂ monotone inclusion with A = 2I and B = x₊
# ============================================================================
#
# Hilbert-space example on ℓ₂(ℝ):
#   A(x) = 2x
#   B(x) = ((x_i + |x_i|)/2)_i = x₊
#
# The infinite-dimensional example is benchmarked through its finite-dimensional
# truncation to ℝᴺ, where `dim = N` is the number of retained coordinates. The
# inclusion 0 ∈ A(x) + B(x) has the unique solution x* = 0.
#
# Resolvent:
#   J^A_ρ(u) = (I + ρA)⁻¹ u = u / (1 + 2ρ)
#
# Since the source example does not prescribe a separate native stopping
# quantity, we use the fixed-point residual at ρ = 1:
#   ‖x - J^A_1(x - B(x))‖.

"""
    _build_problem5(dim::Int; n_seeds::Int=10) -> TestProblem

Build the finite-dimensional truncation of the ℓ₂ monotone-inclusion example
with `A = 2I` and `B = x₊`. Here `dim = N` is the number of coordinates kept
from the ambient ℓ₂ space. The exact solution is `x* = 0`.
"""
function _build_problem5(dim::Int; n_seeds::Int=10)
    dim >= 1 || throw(ArgumentError("_build_problem5: dim must be >= 1, got $dim"))
    n_seeds >= 1 || throw(ArgumentError("_build_problem5: n_seeds must be >= 1, got $n_seeds"))

    N = dim
    L = 1.0

    B_fn = x -> 0.5 .* (x .+ abs.(x))
    resolvent_A_fn = (x, ρ) -> x ./ (1.0 + 2.0 * ρ)
    native_res_fn = let B_fn = B_fn, resolvent_A_fn = resolvent_A_fn
        (x, x_prev) -> norm(x .- resolvent_A_fn(x .- B_fn(x), 1.0))
    end

    initial_points = InitialPoint[]
    for n in 1:n_seeds
        rng = _make_rng(5, dim, n)
        x0 = randn(rng, N)
        push!(initial_points, InitialPoint("seed$n", n, x0))
    end

    metadata = (L = L, N = N)

    return TestProblem(
        5, "Example4.2_l2_positivepart", N,
        B_fn, resolvent_A_fn, native_res_fn,
        zeros(N),
        initial_points,
        metadata,
    )
end

# ============================================================================
# Public dispatcher + accessors
# ============================================================================

"""
    get_problem(id::Int; dim::Int=0, n_seeds::Int=(id == 3 ? 20 : 10)) -> TestProblem

Build and return the test problem identified by `id ∈ PROBLEM_IDS = [1, 2, 3, 4, 5]`
at the given dimension.

- **P1** (Volterra SFP): `dim = N` is the grid resolution (`dim ≥ 2`).
- **P2** (ℓ1+quad): `dim = m ≥ 1`.
- **P3** (optimal control): `dim = K` Euler nodes (`dim ≥ 10`; `0` ⇒ `K = 100`).
- **P4** (LASSO): `dim = N` is the signal length (`dim ≥ 4`); `M = N÷2`.
- **P5** (ℓ₂ example): `dim = N ≥ 1` is the truncation length.

`n_seeds` controls the number of generated initial points. The default
depends on `id`: **20** for P3 (single dim, more seeds → less noisy profile)
and **10** for P1/P2/P4/P5 (multiple dims each). Override for `--quick` runs.

Re-generates data and initial points from the deterministic seed combinator
on every call, so calls with the same `(id, dim, n_seeds)` return
bitwise-identical problems across Julia sessions (Julia version permitting —
`Xoshiro` itself is documented version-stable).
"""
function get_problem(id::Int; dim::Int = 0, n_seeds::Int = (id == 3 ? 20 : 10))
    id == 1 && return _build_problem1(dim; n_seeds=n_seeds)
    id == 2 && return _build_problem2(dim; n_seeds=n_seeds)
    id == 3 && return _build_problem3(dim; n_seeds=n_seeds)
    id == 4 && return _build_problem4(dim; n_seeds=n_seeds)
    id == 5 && return _build_problem5(dim; n_seeds=n_seeds)
    throw(ArgumentError("Unknown problem id: $id. Known: $PROBLEM_IDS"))
end





using Random
using LinearAlgebra


# ============================================================================
# BUILD_SPARSE_LOGISTIC_TRUE
#
# Build a synthetic sparse logistic-regression problem:
#
#     minimize  Φ(x) = f(x) + rho ||x||₁,
#
# where
#
#     f(x) = (1/N) sum_i log(1 + exp(-b_i a_i'x)).
#
# Its monotone-inclusion formulation is
#
#     0 ∈ A(x) + B(x),
#
# where
#
#     A = ∂(rho ||·||₁),
#     B = ∇f.
#
# The synthetic data are generated using a hidden sparse vector x_true:
#
#     1. Generate X ∈ R^(N×m).
#     2. Generate `sparsity` nonzero coefficients.
#     3. Append zeros and randomly permute the complete vector.
#     4. Form scores = X*x_true.
#     5. Set b_i = +1 if score_i ≥ 0 and -1 otherwise.
#
# The solver receives B, the resolvent of A, the native residual and the
# initial points. The hidden vector x_true is stored only in metadata for
# evaluating coefficient and support recovery.
#
# Inputs:
#   dim          - Int; number of features and dimension of x
#   n_samples    - Int; number of data samples
#   sparsity     - Int; number of nonzero entries in x_true
#   rho          - Float64; ℓ₁-regularization parameter
#   n_seeds      - Int; number of initial points used in the benchmark
#   seed         - Int; data-generation seed index passed to _make_rng
#   problem_id   - Int; unique problem identifier used by TestProblem
#
# Outputs:
#   TestProblem containing:
#     B_fn            - logistic-gradient operator
#     resolvent_A_fn  - soft-thresholding resolvent
#     native_res_fn   - fixed-point residual
#     initial_points  - seeded initial points
#     metadata        - generated data and diagnostic information
#
# Metadata:
#   X                - data matrix of size N × m
#   b                - labels in {-1,+1}
#   x_true           - hidden sparse coefficient vector
#   x_unpermuted     - sparse vector before permutation
#   nonzero_values   - generated nonzero coefficients
#   permutation      - permutation used to construct x_true
#   support          - nonzero positions of x_true
#   nnz_true         - actual number of nonzero entries
#   rho              - regularization parameter
#   L                - Lipschitz constant of B
#   N                - number of samples
#   m                - number of features
#   sparsity         - requested sparsity
#   loss_fn          - smooth logistic-loss function
#   objective_fn     - complete objective function
# ============================================================================


"""
    _softplus_stable(t)

Numerically stable evaluation of

    log(1 + exp(t)).
"""
@inline function _softplus_stable(t::Float64)
    return max(t, 0.0) + log1p(exp(-abs(t)))
end


"""
    _inverse_logistic_stable(t)

Numerically stable evaluation of

    1 / (1 + exp(t)).

This quantity appears in the gradient of

    log(1 + exp(-bᵢ aᵢ'x)).
"""
@inline function _inverse_logistic_stable(t::Float64)

    if t >= 0.0
        et = exp(-t)
        return et / (1.0 + et)
    else
        et = exp(t)
        return 1.0 / (1.0 + et)
    end
end


function _build_sparse_logistic_true(
    dim::Int;
    n_samples::Int = 5 * dim,
    sparsity::Int = max(1, dim ÷ 10),
    rho::Float64 = 1.0e-3,
    n_seeds::Int = 10,
    seed::Int = 0,
    problem_id::Int = 61,
)

    # ------------------------------------------------------------------------
    # 1. Validate inputs
    # ------------------------------------------------------------------------

    dim >= 1 ||
        throw(ArgumentError(
            "_build_sparse_logistic_true: dim must be positive, got $dim."
        ))

    n_samples >= 1 ||
        throw(ArgumentError(
            "_build_sparse_logistic_true: n_samples must be positive, " *
            "got $n_samples."
        ))

    1 <= sparsity <= dim ||
        throw(ArgumentError(
            "_build_sparse_logistic_true: sparsity must satisfy " *
            "1 ≤ sparsity ≤ dim; got sparsity=$sparsity and dim=$dim."
        ))

    rho >= 0.0 ||
        throw(ArgumentError(
            "_build_sparse_logistic_true: rho must be nonnegative, got $rho."
        ))

    n_seeds >= 1 ||
        throw(ArgumentError(
            "_build_sparse_logistic_true: n_seeds must be at least 1, " *
            "got $n_seeds."
        ))


    # ------------------------------------------------------------------------
    # 2. Set dimensions
    # ------------------------------------------------------------------------

    m = dim
    N = n_samples

    data_rng = _make_rng(problem_id, dim, seed)


    # ------------------------------------------------------------------------
    # 3. Generate the data matrix
    #
    #     X ∈ R^(N×m).
    #
    # Rows represent samples and columns represent features.
    # ------------------------------------------------------------------------

    X = randn(data_rng, N, m)


    # ------------------------------------------------------------------------
    # 4. Generate the hidden sparse vector through permutation
    # ------------------------------------------------------------------------

    # Generate exactly `sparsity` coefficient values.
    nonzero_values = randn(data_rng, sparsity)

    # Initially place the nonzero values first and append zeros:
    #
    #     x_unpermuted
    #       = [c₁, ..., c_sparsity, 0, ..., 0]'.
    #
    x_unpermuted = vcat(
        nonzero_values,
        zeros(m - sparsity),
    )

    # Random permutation of 1,2,...,m.
    permutation = randperm(data_rng, m)

    # Rearrange the complete vector.
    x_true = x_unpermuted[permutation]

    # Nonzero positions of the final sparse vector.
    support = findall(!iszero, x_true)

    # Actual number of nonzero entries.
    nnz_true = count(!iszero, x_true)


    # ------------------------------------------------------------------------
    # 5. Generate class labels from x_true
    #
    #     score_i = a_i'x_true,
    #
    #     b_i = +1,  if score_i ≥ 0,
    #           -1,  otherwise.
    # ------------------------------------------------------------------------

    scores = X * x_true

    b = ifelse.(scores .>= 0.0, 1.0, -1.0)


    # ------------------------------------------------------------------------
    # 6. Define the smooth logistic-loss function
    #
    #     f(x)
    #       = (1/N) sum_i log(1 + exp(-b_i a_i'x)).
    # ------------------------------------------------------------------------

    loss_fn = let X = X, b = b, N = N, m = m

        x -> begin

            length(x) == m ||
                throw(DimensionMismatch(
                    "loss_fn: x must have length $m, " *
                    "got length $(length(x))."
                ))

            margins = b .* (X * x)

            total_loss = 0.0

            @inbounds for i in eachindex(margins)
                total_loss += _softplus_stable(-margins[i])
            end

            return total_loss / N
        end
    end


    # ------------------------------------------------------------------------
    # 7. Define B(x) = ∇f(x)
    #
    #     B(x)
    #       = -(1/N) X'[
    #           b ./ (1 + exp(b .* (X*x)))
    #         ].
    # ------------------------------------------------------------------------

    B_fn = let X = X, b = b, N = N, m = m

        x -> begin

            length(x) == m ||
                throw(DimensionMismatch(
                    "B_fn: x must have length $m, " *
                    "got length $(length(x))."
                ))

            margins = b .* (X * x)

            weights = similar(margins)

            @inbounds for i in eachindex(margins)
                weights[i] =
                    _inverse_logistic_stable(margins[i])
            end

            return -(X' * (b .* weights)) / N
        end
    end


    # ------------------------------------------------------------------------
    # 8. Define the resolvent of A = ∂(rho ||·||₁)
    #
    # For an algorithmic stepsize λ > 0,
    #
    #     J^A_λ(v)
    #       = prox_{λ rho ||·||₁}(v)
    #       = soft_thresholding(v, λ*rho).
    # ------------------------------------------------------------------------

    resolvent_A_fn = let rho = rho, m = m

        (v, λ) -> begin

            length(v) == m ||
                throw(DimensionMismatch(
                    "resolvent_A_fn: v must have length $m, " *
                    "got length $(length(v))."
                ))

            λ > 0.0 ||
                throw(ArgumentError(
                    "resolvent_A_fn: λ must be positive, got $λ."
                ))

            return soft_thresholding(v, λ * rho)
        end
    end


    # ------------------------------------------------------------------------
    # 9. Define the complete objective
    #
    #     Φ(x) = f(x) + rho ||x||₁.
    # ------------------------------------------------------------------------

    objective_fn = let loss_fn = loss_fn, rho = rho

        x -> loss_fn(x) + rho * norm(x, 1)
    end


    # ------------------------------------------------------------------------
    # 10. Calculate a Lipschitz constant of B
    #
    # Since the loss is averaged over N samples,
    #
    #     L = ||X||₂²/(4N).
    # ------------------------------------------------------------------------

    X_opnorm = opnorm(X, 2)

    L = 0.25 * X_opnorm^2 / N


    # ------------------------------------------------------------------------
    # 11. Define the native fixed-point residual
    #
    # At stepsize λ = 1,
    #
    #     R(x)
    #       = ||x - J^A_1(x - B(x))||.
    #
    # The second argument x_prev is required by the TestProblem interface,
    # but is not needed for this residual.
    # ------------------------------------------------------------------------

    native_res_fn =
        let B_fn = B_fn,
            resolvent_A_fn = resolvent_A_fn

            (x, x_prev) -> begin

                proximal_point =
                    resolvent_A_fn(
                        x .- B_fn(x),
                        1.0,
                    )

                return norm(x .- proximal_point)
            end
        end


    # ------------------------------------------------------------------------
    # 12. Generate benchmark initial points
    #
    # Each initial point is sampled uniformly from [-1,1]^m.
    # ------------------------------------------------------------------------

    initial_points = InitialPoint[]

    for initial_seed in 1:n_seeds

        rng = _make_rng(
            problem_id,
            dim,
            initial_seed,
        )

        x0 =
            -1.0 .+
            2.0 .* rand(rng, m)

        push!(
            initial_points,
            InitialPoint(
                "seed$initial_seed",
                initial_seed,
                x0,
            ),
        )
    end


    # ------------------------------------------------------------------------
    # 13. Store generated data and diagnostic quantities
    # ------------------------------------------------------------------------

    metadata = (
        X = X,
        b = b,

        x_true = x_true,
        x_unpermuted = x_unpermuted,
        nonzero_values = nonzero_values,
        permutation = permutation,
        support = support,
        nnz_true = nnz_true,

        rho = rho,
        L = L,

        N = N,
        m = m,
        sparsity = sparsity,

        seed = seed,
        problem_id = problem_id,

        loss_fn = loss_fn,
        objective_fn = objective_fn,
    )


    # ------------------------------------------------------------------------
    # 14. Return the repository-compatible TestProblem
    # ------------------------------------------------------------------------

    return TestProblem(
        problem_id,
        "SparseLogisticTrue_L1",
        m,
        B_fn,
        resolvent_A_fn,
        native_res_fn,
        nothing,
        initial_points,
        metadata,
    )
end
"""
    get_sparse_logistic_problem(; dim::Int=1024, n_seeds::Int=10, rho_scale::Float64=5.0e-3) -> TestProblem

Named accessor for the sparse ℓ1-regularized logistic-regression problem.
Provided separately from `get_problem(id, ...)` so the benchmark's numbered
suite remains unchanged.
"""
function get_sparse_logistic_problem(; dim::Int=1024, n_seeds::Int=10, rho_scale::Float64=5.0e-3)
    return _build_sparse_logistic(dim; n_seeds=n_seeds, rho_scale=rho_scale)
end

"""
    get_initial_points(prob::TestProblem) -> Vector{InitialPoint}

Return the pre-generated initial-point list. Sugar over field access; provided
for API symmetry with the toolkit's pattern.
"""
get_initial_points(prob::TestProblem) = prob.initial_points