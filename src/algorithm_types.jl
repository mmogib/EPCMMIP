# algorithm_types.jl — Algorithm type hierarchy + per-problem presets.
#
# Defines the AbstractAlgorithm hierarchy, the five concrete algorithm
# structs (EPCM, MTTM, IMTTM, IFRAB, SFRBM), per-algorithm preset registries,
# and the project's trait methods (`name`, `version`). Method bodies for
# `solve(::T, prob, x0; stopping, observers)` live in `algorithm.jl`
# (Steps 7 and 10).
#
# Design references:
#  - script_port_plan.md §4 Q3 (post-review revision), §5.1 algorithm_types.jl row.
#  - paper/main.tex §Parameter choices (lines 1593-1626) — source-paper defaults
#    and per-problem `tau_0` (EPCM) / `lambda_0` (IMTTM) adjustments.
#  - plan_review_findings.md B2 (IFRAB ϑ_n = ϑ̄·n/(n+1) parameter-free) /
#    B4 (per-(method, problem) tuning).
#
# Design principles:
#  - Algorithm INSTANCES carry their parameters as concrete-typed fields.
#  - Rule-selectors (e.g. α_n = 1/(n+1)) are encoded as `Symbol` tags the
#    solver dispatches on. **No `Function`-typed fields** — that would
#    make `make_config_hash` non-reproducible (gensym counter leaks into
#    repr(Function)).
#  - Each algorithm declares a const `<ALG>_PRESETS::Dict{Symbol,NamedTuple}`.
#    Keys: `:P1, :P2, :P3` (per-problem tuned defaults — overwritten by
#    `s25_promote_best_params.jl` after LHS) plus `:paper` (source-paper
#    defaults used by `s28_validate_baselines.jl`).
#  - Outer constructor `T(preset::Symbol; kwargs...)` merges the preset
#    NamedTuple with explicit kwarg overrides.
#  - Trait methods `name(::Type{T})` and `version(::Type{T})` feed
#    `make_config_hash(alg, prob_id, eps, maxiter)`.

# ============================================================================
# Abstract type + interface declaration
# ============================================================================

"""
    AbstractAlgorithm

Root abstract type for all monotone-inclusion solvers in this project.

## Interface (each concrete subtype `T` must implement):

  - `name(::Type{T})::String`             — short tag, e.g. `"EPCM"`.
  - `version(::Type{T})::VersionNumber`   — semver; bumps produce fresh
                                            DB hashes via `make_config_hash`.
  - `solve(alg::T, prob::TestProblem, x0::Vector{Float64}; stopping::Tuple, observers::Tuple) -> SolverResult`
                                            — body in `algorithm.jl`.

## Conventions

  - Const registry `<ALG>_PRESETS::Dict{Symbol,NamedTuple}` with keys
    `:P1, :P2, :P3, :paper`.
  - Outer constructor `T(preset::Symbol; kwargs...) = T(; merge(preset, kwargs)...)`.

## Field-typing rules

  - All param fields must be concretely typed (`Float64`, `Symbol`, ...).
  - **No `Function`-typed fields.** Encode rule-selectors as `Symbol`
    tags the `solve` method dispatches on. This keeps `make_config_hash`
    reproducible across Julia sessions (gensym counter would otherwise
    leak into `repr(Function)`).

## Versioning discipline

  - PATCH bump: bug fix in the solver body (same math, different output bits).
  - MINOR bump: logic change (different mathematical output).
  - MAJOR bump: signature break.
  - Bumping the version produces a different `config_hash`; old DB rows
    preserve historical state; new runs land in fresh rows.
"""
abstract type AbstractAlgorithm end

"""
    name(::Type{T}) where {T <: AbstractAlgorithm} -> String

Short tag for `T` (e.g. `"EPCM"`). Used as the `method` column in the DB
and in performance-profile legends. Each concrete subtype implements this.

Convenience forwarder: `name(alg::AbstractAlgorithm) = name(typeof(alg))`.
"""
function name end
name(alg::AbstractAlgorithm) = name(typeof(alg))

"""
    version(::Type{T}) where {T <: AbstractAlgorithm} -> VersionNumber

Semver for `T`. Bumping it produces a fresh `config_hash` for every run.
Use `VersionNumber` (not `String`) so results can be filtered with
`v"1.0.0" ≤ ver < v"2.0.0"` semantics. Each concrete subtype implements this.

Convenience forwarder: `version(alg::AbstractAlgorithm) = version(typeof(alg))`.
"""
function version end
version(alg::AbstractAlgorithm) = version(typeof(alg))

"""
    solve(alg::AbstractAlgorithm, prob::TestProblem, x0::Vector{Float64};
          stopping::Tuple, observers::Tuple=()) -> SolverResult

Solve the monotone inclusion `0 ∈ A(x) + B(x)` defined by `prob`, starting
from `x0`, using algorithm `alg`. Method bodies live in `algorithm.jl`
(Step 7 for EPCM, Step 10 for the four baselines). Each method honours
the 9-step "Solver responsibility" contract in `callbacks.jl`'s
`SolverState` docstring.
"""
function solve end

# ============================================================================
# EPCM — Extrapolated Projection–Contraction Method (proposed)
# ============================================================================
#
# Source: this paper, `\EPCM` macro. Manuscript §Parameter choices lines 1598-1605:
#   γ=1, ϑ_0=0.2, ϑ_1=0.1, ζ=0.5; δ_k = 1/(k+1)², σ_k = 1/(k+2).
#   τ_0: 0.01 (P1), 1e-4 (P2), 0.1 (P3).
#   (μ is a proof-only constant post-refactor — not a field; see struct docstring.)
#
# σ_k, δ_k are parametrized (Step 11-2) as a sensitivity-study family staying
# inside the Algorithm 1 admissibility envelope (σ_k∈(0,1),→0,Σ=∞; δ_k≥0,Σ<∞):
#   σ_k = sigma_scale / (k+2)^sigma_exp   (sigma_exp∈(0,1]; σ_0 = sigma_scale/2^sigma_exp < 1)
#   δ_k = 1 / (k+1)^delta_exp             (delta_exp > 1; or delta_rule=:zero ⟹ δ_k≡0)
# Defaults sigma_scale=1, sigma_exp=1, delta_exp=2 reproduce σ_k=1/(k+2), δ_k=1/(k+1)².

"""
    EPCM(; gamma, vartheta_0, vartheta_1, zeta, tau_0,
           delta_rule, sigma_rule, delta_exp, sigma_exp, sigma_scale)
    EPCM(preset::Symbol; kwargs...)

Extrapolated projection–contraction method (proposed). Parameters per
manuscript §Parameter choices. All kwargs are required in the explicit
constructor; the preset constructor fills from `EPCM_PRESETS`.

## Parametrized sequences (Step 11-2 sensitivity-study family)

  - `sigma_rule = :power` ↦ σ_k = `sigma_scale` / (k+2)^`sigma_exp`.
    Admissibility (Algorithm 1: σ_k∈(0,1), σ_k→0, Σσ_k=∞) requires
    `sigma_exp ∈ (0,1]` and σ_0 = `sigma_scale`/2^`sigma_exp` < 1.
    Default `sigma_exp=1, sigma_scale=1` ⟹ σ_k = 1/(k+2) (Lieder-optimal Halpern).
    `sigma_exp<1` strengthens anchoring; `sigma_scale<1` weakens it.
  - `delta_rule = :power` ↦ δ_k = 1 / (k+1)^`delta_exp`.
    Admissibility (Σδ_k<∞) requires `delta_exp > 1`. Default `delta_exp=2`
    ⟹ δ_k = 1/(k+1)² (total step-growth headroom Π(1+δ_k) ≈ 3.68×).
    Smaller `delta_exp` ⟹ more headroom (helps τ_0 robustness); larger ⟹ less.
  - `delta_rule = :zero` ↦ δ_k ≡ 0 (monotone-shrink baseline; `delta_exp` ignored).

Validated at `solve` entry; out-of-envelope values throw `ArgumentError`.

Note: the proof-side constant `μ ∈ (1, 2)` (with `μ·ϑ_0 < √(γ/(2(2γ+1)))`)
is NOT a field of this struct. Its existence is guaranteed by the strict
admissibility constraint `ϑ_0 < √(γ/(2(2γ+1)))` (validated at `solve`
entry); the iteration does not depend on which admissible μ is chosen.
"""
struct EPCM <: AbstractAlgorithm
    gamma::Float64
    vartheta_0::Float64
    vartheta_1::Float64
    zeta::Float64
    tau_0::Float64
    delta_rule::Symbol
    sigma_rule::Symbol
    delta_exp::Float64
    sigma_exp::Float64
    sigma_scale::Float64

    function EPCM(; gamma::Float64, vartheta_0::Float64, vartheta_1::Float64,
                  zeta::Float64, tau_0::Float64,
                  delta_rule::Symbol, sigma_rule::Symbol,
                  delta_exp::Float64, sigma_exp::Float64, sigma_scale::Float64)
        return new(gamma, vartheta_0, vartheta_1, zeta, tau_0,
                   delta_rule, sigma_rule, delta_exp, sigma_exp, sigma_scale)
    end
end

"""
EPCM presets keyed by `Symbol`. Per-problem entries (`:P1, :P2, :P3`)
are initially seeded with the manuscript defaults and the per-problem
`tau_0`; `s25_promote_best_params.jl` overwrites them after LHS tuning.
The `delta_exp=2, sigma_exp=1, sigma_scale=1` defaults reproduce the
manuscript's `δ_k=1/(k+1)²`, `σ_k=1/(k+2)`.

There is no `:paper` preset for EPCM — EPCM IS this paper. Use a per-
problem preset and override `tau_0` if you need a custom scaling.
"""
const EPCM_PRESETS = Dict{Symbol,NamedTuple}(
    # τ_0 scaled per problem (OAT centre, tuned later). P1=Volterra (L≈0.4),
    # P2=ℓ1+quad (L=2), P3=control (L=4), P4=LASSO (L=1).
    :P1 => (gamma=1.0, vartheta_0=0.2, vartheta_1=0.1, zeta=0.5, tau_0=1.0,
            delta_rule=:power, sigma_rule=:power,
            delta_exp=2.0, sigma_exp=1.0, sigma_scale=1.0),
    :P2 => (gamma=1.0, vartheta_0=0.2, vartheta_1=0.1, zeta=0.5, tau_0=1.0e-4,
            delta_rule=:power, sigma_rule=:power,
            delta_exp=2.0, sigma_exp=1.0, sigma_scale=1.0),
    :P3 => (gamma=1.0, vartheta_0=0.2, vartheta_1=0.1, zeta=0.5, tau_0=0.1,
            delta_rule=:power, sigma_rule=:power,
            delta_exp=2.0, sigma_exp=1.0, sigma_scale=1.0),
    :P4 => (gamma=1.0, vartheta_0=0.2, vartheta_1=0.1, zeta=0.5, tau_0=0.5,
            delta_rule=:power, sigma_rule=:power,
            delta_exp=2.0, sigma_exp=1.0, sigma_scale=1.0),
)

function EPCM(preset::Symbol; kwargs...)
    haskey(EPCM_PRESETS, preset) ||
        throw(ArgumentError("Unknown EPCM preset :$preset. Known: $(join(sort(collect(keys(EPCM_PRESETS))), ", "))"))
    return EPCM(; merge(EPCM_PRESETS[preset], NamedTuple(kwargs))...)
end

name(::Type{EPCM})    = "EPCM"
version(::Type{EPCM}) = v"1.2.0"   # 1.1.0→1.2.0: θ_k clamped at 0 via max{0,·} (PC descent needs θ_k≥0; notes §24.3). NB: clamp is in solve(), not a struct field ⟹ config_hash unchanged — wipe/re-run if comparing to pre-clamp results.

# ============================================================================
# MTTM — Mann Tseng-Type Method (Gibali2018 Alg 1)
# ============================================================================
#
# Source: Gibali--Thong 2018, Algorithm 1. Manuscript line 1607 (first sentence):
#   λ_0 = 7.55, μ = 0.85, α_n = 1/n, β_n = (n-1)/(2n)
# Source examples are SCFP (Ex 3) and sparse signal recovery (Ex 4), neither
# of which is in our problem set — so :paper = :P1 = :P2 = :P3 initially.

"""
    MTTM(; lambda_0, mu, alpha_rule, beta_rule)
    MTTM(preset::Symbol; kwargs...)

Mann Tseng-type method (Gibali2018 Algorithm 1). Manuscript §Parameter choices
line 1607.

`alpha_rule = :inv_n` ↦ α_n = 1/n.
`beta_rule = :n_minus_1_over_2n` ↦ β_n = (n-1)/(2n).
"""
struct MTTM <: AbstractAlgorithm
    lambda_0::Float64
    mu::Float64
    alpha_rule::Symbol
    beta_rule::Symbol

    function MTTM(; lambda_0::Float64, mu::Float64,
                  alpha_rule::Symbol, beta_rule::Symbol)
        return new(lambda_0, mu, alpha_rule, beta_rule)
    end
end

"""
MTTM presets. `:paper` is the Gibali2018 Ex 3 / Ex 4 verbatim values
(used by `s28_validate_baselines.jl`). Per-problem presets are seeded
identically; `s25` rewrites them after LHS.
"""
const MTTM_PRESETS = Dict{Symbol,NamedTuple}(
    :paper => (lambda_0=7.55, mu=0.85, alpha_rule=:inv_n, beta_rule=:n_minus_1_over_2n),
    :P1    => (lambda_0=7.55, mu=0.85, alpha_rule=:inv_n, beta_rule=:n_minus_1_over_2n),
    :P2    => (lambda_0=7.55, mu=0.85, alpha_rule=:inv_n, beta_rule=:n_minus_1_over_2n),
    :P3    => (lambda_0=7.55, mu=0.85, alpha_rule=:inv_n, beta_rule=:n_minus_1_over_2n),
    :P4    => (lambda_0=7.55, mu=0.85, alpha_rule=:inv_n, beta_rule=:n_minus_1_over_2n),
)

function MTTM(preset::Symbol; kwargs...)
    haskey(MTTM_PRESETS, preset) ||
        throw(ArgumentError("Unknown MTTM preset :$preset. Known: $(join(sort(collect(keys(MTTM_PRESETS))), ", "))"))
    return MTTM(; merge(MTTM_PRESETS[preset], NamedTuple(kwargs))...)
end

name(::Type{MTTM})    = "MTTM"
version(::Type{MTTM}) = v"1.0.0"

# ============================================================================
# IMTTM — Inertial Mann Tseng-Type Method (Tan2022a Alg 3.3)
# ============================================================================
#
# Source: Tan--Cho 2022, Algorithm 3.3. Manuscript line 1607 (later sentences):
#   μ=0.5, α_n=1/(n+1), β_n=0.5(1−α_n), ε_n=100/(n+1)², θ=0.5.
#   λ_0=1 (default); λ_0=0.01 for P1 (Tan's Ex 5.1 adjustment, manuscript-cited).

"""
    IMTTM(; lambda_0, mu, theta, alpha_rule, beta_rule, epsilon_rule)
    IMTTM(preset::Symbol; kwargs...)

Inertial Mann Tseng-type method (Tan2022a Algorithm 3.3).

`alpha_rule = :inv_n_plus_1`    ↦ α_n = 1/(n+1).
`beta_rule = :tan`              ↦ β_n = 0.5·(1 − α_n).
`epsilon_rule = :hundred_inv_sq`↦ ε_n = 100/(n+1)².
"""
struct IMTTM <: AbstractAlgorithm
    lambda_0::Float64
    mu::Float64
    theta::Float64
    alpha_rule::Symbol
    beta_rule::Symbol
    epsilon_rule::Symbol

    function IMTTM(; lambda_0::Float64, mu::Float64, theta::Float64,
                   alpha_rule::Symbol, beta_rule::Symbol, epsilon_rule::Symbol)
        return new(lambda_0, mu, theta, alpha_rule, beta_rule, epsilon_rule)
    end
end

"""
IMTTM presets. `:paper` = Tan2022a Ex 5.1 home-cell values (= our P1
home cell with λ_0 = 0.01). `:P1` matches `:paper`. `:P2, :P3` use
Tan's default `λ_0 = 1`. `s25` may retune any of these post-LHS.
"""
# `:paper` keeps Tan's Ex 5.1 home value (λ_0=0.01, for s28 home-cell validation
# against the box-VIP, which is no longer in the benchmark suite). The benchmark
# presets use Tan's default λ_0=1 (Ex 5.2/5.3 are well-conditioned, no 1/L≪0.01
# adjustment needed); P1 is now the Volterra SFP, not the box-VIP.
const IMTTM_PRESETS = Dict{Symbol,NamedTuple}(
    :paper => (lambda_0=0.01, mu=0.5, theta=0.5,
               alpha_rule=:inv_n_plus_1, beta_rule=:tan, epsilon_rule=:hundred_inv_sq),
    :P1    => (lambda_0=1.0,  mu=0.5, theta=0.5,
               alpha_rule=:inv_n_plus_1, beta_rule=:tan, epsilon_rule=:hundred_inv_sq),
    :P2    => (lambda_0=1.0,  mu=0.5, theta=0.5,
               alpha_rule=:inv_n_plus_1, beta_rule=:tan, epsilon_rule=:hundred_inv_sq),
    :P3    => (lambda_0=1.0,  mu=0.5, theta=0.5,
               alpha_rule=:inv_n_plus_1, beta_rule=:tan, epsilon_rule=:hundred_inv_sq),
    :P4    => (lambda_0=1.0,  mu=0.5, theta=0.5,
               alpha_rule=:inv_n_plus_1, beta_rule=:tan, epsilon_rule=:hundred_inv_sq),
)

function IMTTM(preset::Symbol; kwargs...)
    haskey(IMTTM_PRESETS, preset) ||
        throw(ArgumentError("Unknown IMTTM preset :$preset. Known: $(join(sort(collect(keys(IMTTM_PRESETS))), ", "))"))
    return IMTTM(; merge(IMTTM_PRESETS[preset], NamedTuple(kwargs))...)
end

name(::Type{IMTTM})    = "IMTTM"
version(::Type{IMTTM}) = v"1.0.0"

# ============================================================================
# IFRAB — Variable-Inertial Forward-Reflected-Anchored-Backward (Izuchukwu2023 Alg 4.5)
# ============================================================================
#
# Source: Izuchukwu--Reich--Shehu--Taiwo 2023, Algorithm 4.5 + stepsize eq 3.2.
# Manuscript lines 1609-1622: δ_0=0.1, δ_1=0.3, r̄=0.3, σ_n=0.005/(3n+25000),
# c_n=1/(n²+1), and ϑ_n = ϑ̄·n/(n+1) with ϑ̄=0.04 (per B2 post-review).

"""
    IFRAB(; delta_0, delta_1, rbar, vartheta_bar, sigma_rule, c_rule, vartheta_rule)
    IFRAB(preset::Symbol; kwargs...)

Variable-inertial forward-reflected-anchored-backward method (Izuchukwu2023
Algorithm 4.5).

`sigma_rule = :izuchukwu_sigma`  ↦ σ_n = 0.005/(3n + 25000).
`c_rule = :inv_n_sq_plus_1`      ↦ c_n = 1/(n² + 1).
`vartheta_rule = :n_over_n_plus_1` ↦ ϑ_n = ϑ̄ · n/(n+1).
"""
struct IFRAB <: AbstractAlgorithm
    delta_0::Float64
    delta_1::Float64
    rbar::Float64
    vartheta_bar::Float64
    sigma_rule::Symbol
    c_rule::Symbol
    vartheta_rule::Symbol

    function IFRAB(; delta_0::Float64, delta_1::Float64, rbar::Float64,
                   vartheta_bar::Float64, sigma_rule::Symbol, c_rule::Symbol,
                   vartheta_rule::Symbol)
        return new(delta_0, delta_1, rbar, vartheta_bar,
                   sigma_rule, c_rule, vartheta_rule)
    end
end

"""
IFRAB presets. The Izuchukwu2023 Ex 6.2 home-cell values (= our P3 home cell)
seed all four presets identically; `s25` rewrites :P1, :P2 post-LHS.
"""
const IFRAB_PRESETS = Dict{Symbol,NamedTuple}(
    :paper => (delta_0=0.1, delta_1=0.3, rbar=0.3, vartheta_bar=0.04,
               sigma_rule=:izuchukwu_sigma, c_rule=:inv_n_sq_plus_1,
               vartheta_rule=:n_over_n_plus_1),
    :P1    => (delta_0=0.1, delta_1=0.3, rbar=0.3, vartheta_bar=0.04,
               sigma_rule=:izuchukwu_sigma, c_rule=:inv_n_sq_plus_1,
               vartheta_rule=:n_over_n_plus_1),
    :P2    => (delta_0=0.1, delta_1=0.3, rbar=0.3, vartheta_bar=0.04,
               sigma_rule=:izuchukwu_sigma, c_rule=:inv_n_sq_plus_1,
               vartheta_rule=:n_over_n_plus_1),
    :P3    => (delta_0=0.1, delta_1=0.3, rbar=0.3, vartheta_bar=0.04,
               sigma_rule=:izuchukwu_sigma, c_rule=:inv_n_sq_plus_1,
               vartheta_rule=:n_over_n_plus_1),
    :P4    => (delta_0=0.1, delta_1=0.3, rbar=0.3, vartheta_bar=0.04,
               sigma_rule=:izuchukwu_sigma, c_rule=:inv_n_sq_plus_1,
               vartheta_rule=:n_over_n_plus_1),
)

function IFRAB(preset::Symbol; kwargs...)
    haskey(IFRAB_PRESETS, preset) ||
        throw(ArgumentError("Unknown IFRAB preset :$preset. Known: $(join(sort(collect(keys(IFRAB_PRESETS))), ", "))"))
    return IFRAB(; merge(IFRAB_PRESETS[preset], NamedTuple(kwargs))...)
end

name(::Type{IFRAB})    = "IFRAB"
version(::Type{IFRAB}) = v"1.0.0"

# ============================================================================
# SFRBM — Strongly-Convergent Forward-Reflected-Backward with Momentum (Yao2024 Alg 2)
# ============================================================================
#
# Source: Yao--Adamu--Shehu 2024, Algorithm 2. Manuscript lines 1623-1627:
#   λ_{-1} = λ_0 = μ = 1e-4, θ = 10, β_k = 1/(5000(k+1)).

"""
    SFRBM(; lambda_minus1, lambda_0, mu, theta, beta_rule)
    SFRBM(preset::Symbol; kwargs...)

Strongly-convergent forward-reflected-backward with momentum (Yao2024
Algorithm 2).

`beta_rule = :yao` ↦ β_k = 1/(5000(k+1)).
"""
struct SFRBM <: AbstractAlgorithm
    lambda_minus1::Float64
    lambda_0::Float64
    mu::Float64
    theta::Float64
    beta_rule::Symbol

    function SFRBM(; lambda_minus1::Float64, lambda_0::Float64,
                   mu::Float64, theta::Float64, beta_rule::Symbol)
        return new(lambda_minus1, lambda_0, mu, theta, beta_rule)
    end
end

"""
SFRBM presets. Yao2024 Examples 4.1–4.3 all use the same parameters; we
seed all presets identically. (P2 home cell = Yao Ex 4.2; `s25` may
retune :P1 and :P3 post-LHS.)
"""
const SFRBM_PRESETS = Dict{Symbol,NamedTuple}(
    :paper => (lambda_minus1=1.0e-4, lambda_0=1.0e-4, mu=1.0e-4,
               theta=10.0, beta_rule=:yao),
    :P1    => (lambda_minus1=1.0e-4, lambda_0=1.0e-4, mu=1.0e-4,
               theta=10.0, beta_rule=:yao),
    :P2    => (lambda_minus1=1.0e-4, lambda_0=1.0e-4, mu=1.0e-4,
               theta=10.0, beta_rule=:yao),
    :P3    => (lambda_minus1=1.0e-4, lambda_0=1.0e-4, mu=1.0e-4,
               theta=10.0, beta_rule=:yao),
    :P4    => (lambda_minus1=1.0e-4, lambda_0=1.0e-4, mu=1.0e-4,
               theta=10.0, beta_rule=:yao),
)

function SFRBM(preset::Symbol; kwargs...)
    haskey(SFRBM_PRESETS, preset) ||
        throw(ArgumentError("Unknown SFRBM preset :$preset. Known: $(join(sort(collect(keys(SFRBM_PRESETS))), ", "))"))
    return SFRBM(; merge(SFRBM_PRESETS[preset], NamedTuple(kwargs))...)
end

name(::Type{SFRBM})    = "SFRBM"
version(::Type{SFRBM}) = v"1.0.0"
