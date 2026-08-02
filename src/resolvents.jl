# resolvents.jl — Closed-form resolvent helpers for the benchmark problems.
#
# Each problem's set-valued operator A has a known closed-form resolvent
# J^A_ρ(x) = (I + ρ A)^{-1}(x). The helpers below implement those:
#
#   - `clipping_box`        — for normal-cone box problems such as P3
#   - `soft_thresholding`   — for ℓ₁-subdifferential problems such as P2/P4
#
# These are pure, allocating functions. They are called by problem builders in
# `problems.jl` to construct `prob.resolvent_A` closures with the
# problem-specific (lo, hi) or (∂‖·‖_1, ρ) baked in. Solvers do not call
# them directly — solvers call `prob.resolvent_A(x, ρ)`.

"""
    clipping_box(x::AbstractVector{Float64}, lo::Real, hi::Real) -> Vector{Float64}

Componentwise projection of `x` onto the box `[lo, hi]^n`. Equivalent to the
resolvent `J^A_ρ(x)` for `A = N_{[lo,hi]^n}` — note that the normal-cone
resolvent is **independent of the step size `ρ`**, so the problem builder
discards `ρ` when wiring this through `prob.resolvent_A`.

Allocating; returns a fresh `Vector{Float64}`.

Used for:
- **P3** (Izuchukwu2023 Ex 6.2): `clipping_box(z, -1.0, 1.0)`.
  (P1 was the box-VIP Ex 5.1 before the 2026-05-28 suite revision; it is now the
  Volterra SFP Ex 5.2, whose half-space resolvent is built inline in `problems.jl`.)
"""
function clipping_box(x::AbstractVector{Float64}, lo::Real, hi::Real)
    lo <= hi || throw(ArgumentError("clipping_box requires lo <= hi, got lo=$lo, hi=$hi"))
    return clamp.(x, convert(Float64, lo), convert(Float64, hi))
end

"""
    soft_thresholding(x::AbstractVector{Float64}, threshold::Real) -> Vector{Float64}

Componentwise soft-thresholding: `sign(x_i) * max(|x_i| - threshold, 0)`.

This is the resolvent `J^A_ρ(x)` for `A = ∂‖·‖_1` with `threshold = ρ`. The
problem builder for P2 wires it as `prob.resolvent_A = (x, ρ) -> soft_thresholding(x, ρ)`,
so the step size flows through as the threshold.

Allocating; returns a fresh `Vector{Float64}`. Componentwise output is `0.0`
exactly when `|x_i| <= threshold`.

Used for:
- **P2** (Yao2024 Ex 4.2): `soft_thresholding(x, ρ)`.
"""
function soft_thresholding(x::AbstractVector{Float64}, threshold::Real)
    threshold >= 0 || throw(ArgumentError("soft_thresholding requires threshold >= 0, got $threshold"))
    t = convert(Float64, threshold)
    return sign.(x) .* max.(abs.(x) .- t, 0.0)
end

"""
    project_simplex(v) -> Vector{Float64}

Euclidean projection onto `{x >= 0 : sum(x) = 1}` using the sort-based
threshold algorithm of complexity `O(n log n)`.
"""
function project_simplex(v::AbstractVector{<:Real})
    isempty(v) && throw(ArgumentError("project_simplex requires a nonempty vector"))
    values = Float64.(v)
    sorted = sort(values; rev = true)
    shifted_cumsum = cumsum(sorted) .- 1.0
    rho = findlast(i -> sorted[i] - shifted_cumsum[i] / i > 0.0, eachindex(sorted))
    rho === nothing && error("simplex projection threshold was not found")
    theta = shifted_cumsum[rho] / rho
    return max.(values .- theta, 0.0)
end

"Project the first `x_dim` and remaining coordinates onto separate simplices."
function project_product_simplex(u::AbstractVector{<:Real}, x_dim::Integer)
    1 <= x_dim < length(u) ||
        throw(ArgumentError("x_dim must satisfy 1 <= x_dim < length(u)"))
    return vcat(project_simplex(@view(u[1:x_dim])),
                project_simplex(@view(u[(x_dim + 1):end])))
end
