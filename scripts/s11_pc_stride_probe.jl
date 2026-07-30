# ============================================================================
# s11: PC-stride probe — is the projection–contraction step a real accelerator?
# ============================================================================
#
# DECISION INSTRUMENT for the PC angle (see notes/pc_stride_viability.md).
#
# The PC step is EPCM's only structural distinction from IFRAB/SFRBM:
#     x_{k+1} = v_k − ζ θ_k u_k,   effective stride (1 + ζ θ_k).
# Rigorous analysis (note §2–§4) reduced "is the PC a real accelerator?" to a
# single per-iteration quantity:
#     θ_k = ρ^He_k − 1 = ⟨r_k,u_k⟩/‖u_k‖² − 1   (typically ≈ 0),
#     θ_k ≤ μϑ_0 (τ_k/τ_{k+1}) · ‖Δz‖/‖u_k‖,    μϑ_0 < ½  (admissibility cap),
# maximized when the B-reflection d_k = B(z_{k−1})−B(z_k) ALIGNS with the
# resolvent gap r_k = w_k − z_k. Structural hypothesis: low-rank / structured
# operator variation forces that alignment ⟹ larger θ_k. Prediction:
#   P3 (rank-1 M = 2h·𝟙𝟙ᵀ, d_k always ∥ 𝟙) → LARGE θ_k  (explains P3 being best),
#   P2 (M = 2I) / P4 (M = CᵀC, full-rank)   → small  θ_k.
#
# This script runs EPCM (its per-problem preset) and logs, per iteration:
#   θ_k (clamped), raw θ_k (pre-clamp, to see how often <0), stride 1+ζθ_k,
#   ‖u_k‖, ‖Δz‖, ‖d_k‖, ‖g_k‖=‖u_k‖/τ_k, and the two diagnostic ratios
#   ‖Δz‖/‖u_k‖ and ‖d_k‖/‖g_k‖, plus τ_k and the native residual.
# Then prints per-problem summary stats + a VERDICT comparison, and writes a
# per-problem trajectory CSV under results/pc_probe/.
#
# READ-ONLY w.r.t. the project: it does NOT touch the DB and does NOT modify
# any solver. The iteration below is a FAITHFUL inline mirror of
# `solve(::EPCM, …)` Steps 1–7 in src/algorithm.jl (EPCM v1.2.0, incl. the
# θ_k = max{0,·} clamp), using the real prob.B / prob.resolvent_A /
# _epcm_sigma / _epcm_delta. If algorithm.jl's EPCM loop changes, re-sync here.
#
# Usage:
#   julia --project=jcode jcode/scripts/s11_pc_stride_probe.jl
#   julia --project=jcode jcode/scripts/s11_pc_stride_probe.jl --maxiter=20000 --problems=P2,P3,P4,P5
#   julia --project=jcode jcode/scripts/s11_pc_stride_probe.jl --vartheta0=0.40 --vartheta1=0.39   # bigger-step variant
# Output: console + results/logs/log_pc_stride_probe_<ts>.txt + results/pc_probe/traj_<P>.csv

include(joinpath(@__DIR__, "..", "src", "includes.jl"))
using Statistics

# ── tiny ARG parser ─────────────────────────────────────────────────────────
function getarg(flag::AbstractString, default::AbstractString)
    for a in ARGS
        startswith(a, flag * "=") && return split(a, "="; limit = 2)[2]
    end
    return default
end

# Admissibility ceiling for ϑ_0 (Algorithm 1 Input): ϑ_0 < √(γ/(2(2γ+1))).
vartheta0_ceiling(γ) = sqrt(γ / (2 * (2γ + 1)))

# ── Faithful inline mirror of solve(::EPCM) Steps 1–7, with diagnostics ─────
# Returns (records::Vector{NamedTuple}, converged::Bool, iters::Int, final_native::Float64).
function run_epcm_probe(alg::EPCM, prob, x0::Vector{Float64};
                        maxiter::Int, native_tol_val::Float64, consec::Int = 2)
    x_0     = copy(x0)          # anchor x_0 (constant)
    x_curr  = copy(x0)          # x_k
    z_prev  = copy(x0)          # z_{k-1}; z_{-1} := x_0 by convention
    Bz_prev = prob.B(z_prev)    # B(z_{k-1}); preloaded
    τ_curr  = alg.tau_0         # τ_k

    records      = NamedTuple[]
    below_count  = 0
    converged    = false
    final_native = NaN
    k = 0
    while k < maxiter
        # Step 1: extrapolation
        σ_k = _epcm_sigma(alg.sigma_rule, alg.sigma_scale, alg.sigma_exp, k)
        w_k = σ_k .* x_0 .+ (1 - σ_k) .* x_curr
        # Step 2: resolvent (uses cached B(z_{k-1}))
        z_k = prob.resolvent_A(w_k .- τ_curr .* Bz_prev, τ_curr)
        # one new B-eval: B(z_k)
        Bz_k = prob.B(z_k)
        # Step 3: forward correction
        v_k = z_k .+ τ_curr .* (Bz_prev .- Bz_k)
        # Step 4: direction u_k = w_k − v_k
        u_k = w_k .- v_k
        # Step 5: PC coefficient θ_k (CLAMPED at 0, v1.2.0) — log raw + clamped
        norm_u_sq = sum(abs2, u_k)
        raw_theta = norm_u_sq > 0 ? dot(v_k .- z_k, u_k) / norm_u_sq : 0.0
        θ_k       = max(0.0, raw_theta)
        # Step 6: x_{k+1} = v_k − ζ θ_k u_k
        x_next = v_k .- (alg.zeta * θ_k) .* u_k
        # Step 7: stepsize update
        Δz       = z_prev .- z_k
        ΔBz      = Bz_prev .- Bz_k
        norm_Δz  = norm(Δz)
        norm_ΔBz = norm(ΔBz)               # = ‖d_k‖
        τ_next = if norm_ΔBz > alg.vartheta_0 * τ_curr * norm_Δz
            alg.vartheta_1 * norm_Δz / norm_ΔBz
        else
            (1.0 + _epcm_delta(alg.delta_rule, alg.delta_exp, k)) * τ_curr
        end

        # ── diagnostics ──────────────────────────────────────────────────
        norm_u     = sqrt(norm_u_sq)
        norm_g     = norm_u / τ_curr                       # ‖g_k‖ = ‖u_k‖/τ_k
        stride     = 1.0 + alg.zeta * θ_k                  # effective stride multiplier
        ratio_dz_u = norm_u > 0 ? norm_Δz / norm_u : NaN    # ‖Δz‖/‖u_k‖  (drives the cap)
        ratio_d_g  = norm_g > 0 ? norm_ΔBz / norm_g : NaN   # ‖d_k‖/‖g_k‖ (= reflection vs residual)
        native     = prob.native_residual(x_next, x_curr)

        push!(records, (
            k = k + 1, tau = τ_curr, raw_theta = raw_theta, theta = θ_k,
            stride = stride, norm_u = norm_u, norm_dz = norm_Δz,
            norm_d = norm_ΔBz, norm_g = norm_g,
            ratio_dz_u = ratio_dz_u, ratio_d_g = ratio_d_g, native = native,
        ))
        final_native = native

        # advance
        x_curr  = x_next
        z_prev  = z_k
        Bz_prev = Bz_k
        τ_curr  = τ_next
        k += 1

        # stopping: native source-paper criterion, de-bounced (matches the suite)
        if isfinite(native) && native < native_tol_val
            below_count += 1
            if below_count >= consec
                converged = true
                break
            end
        else
            below_count = 0
        end
    end
    return records, converged, k, final_native
end

# ── summary over a trajectory (finite-ratio iters only) ─────────────────────
function summarize(records::Vector{NamedTuple})
    # filter non-finite (a diverging config would otherwise make quantile error)
    θ   = Float64[r.theta      for r in records if isfinite(r.theta)]
    raw = Float64[r.raw_theta  for r in records if isfinite(r.raw_theta)]
    st  = Float64[r.stride     for r in records if isfinite(r.stride)]
    rdu = Float64[r.ratio_dz_u for r in records if isfinite(r.ratio_dz_u)]
    rdg = Float64[r.ratio_d_g  for r in records if isfinite(r.ratio_d_g)]
    q(v, p) = isempty(v) ? NaN : quantile(v, p)
    m(v)    = isempty(v) ? NaN : mean(v)
    md(v)   = isempty(v) ? NaN : median(v)
    return (
        n           = length(records),
        theta_mean  = m(θ),  theta_med = md(θ), theta_q90 = q(θ, 0.90), theta_max = isempty(θ) ? NaN : maximum(θ),
        stride_mean = m(st), stride_med = md(st), stride_q90 = q(st, 0.90), stride_max = isempty(st) ? NaN : maximum(st),
        rdu_med     = md(rdu), rdu_q90 = q(rdu, 0.90),
        rdg_med     = md(rdg), rdg_q90 = q(rdg, 0.90),
        frac_pc_active = isempty(θ)   ? NaN : mean(θ   .> 0.0),   # fraction of iters with θ_k>0
        frac_clamped   = isempty(raw) ? NaN : mean(raw .< 0.0),   # fraction where the clamp engaged (raw θ<0)
    )
end

# ── write a per-problem trajectory CSV (manual, no CSV.jl dependency) ───────
function write_traj_csv(path::AbstractString, records::Vector{NamedTuple})
    open(path, "w") do io
        println(io, "k,tau,raw_theta,theta,stride,norm_u,norm_dz,norm_d,norm_g,ratio_dz_u,ratio_d_g,native")
        for r in records
            @printf(io, "%d,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e\n",
                    r.k, r.tau, r.raw_theta, r.theta, r.stride, r.norm_u, r.norm_dz,
                    r.norm_d, r.norm_g, r.ratio_dz_u, r.ratio_d_g, r.native)
        end
    end
    return path
end

function main()
    logpath, tee, _ = setup_logging("pc_stride_probe")

    maxiter = parse(Int, getarg("--maxiter", "10000"))
    v0_ovr  = getarg("--vartheta0", "")
    v1_ovr  = getarg("--vartheta1", "")
    probsel = getarg("--problems", "P1,P2,P3,P4,P5")
    wanted  = Set(String.(strip.(split(probsel, ","))))
    wlist   = join(sort(collect(wanted)), ",")
    allp    = Set(["P1", "P2", "P3", "P4", "P5"])
    all(p -> p in allp, wanted) || error("--problems entries must be among P1|P2|P3|P4|P5")

    println(tee, "=" ^ 78)
    println(tee, "  s11 PC-stride probe — does the projection–contraction step accelerate?  $(Dates.now())")
    println(tee, "=" ^ 78)
    println(tee, "  maxiter=$maxiter  problems=$wlist" *
                 (isempty(v0_ovr) ? "" : "  vartheta0=$v0_ovr") *
                 (isempty(v1_ovr) ? "" : "  vartheta1=$v1_ovr"))
    println(tee, "  Hypothesis: P3 (rank-1 operator variation) shows the LARGEST θ_k / stride; P2/P4/P5 should hug 1.")
    println(tee, "  Logging the REAL EPCM (v1.2.0) iteration; CSV trajectories under results/pc_probe/.\n")

    # problem build specs (one representative instance per problem)
    specs = [
        (id = 1, label = "P1", build = () -> get_problem(1; dim = 128, n_seeds = 1)),
        (id = 2, label = "P2", build = () -> get_problem(2; dim = 300, n_seeds = 1)),
        (id = 3, label = "P3", build = () -> get_problem(3;            n_seeds = 1)),
        (id = 4, label = "P4", build = () -> get_problem(4; dim = 256, n_seeds = 1)),
        (id = 5, label = "P5", build = () -> get_problem(5; dim = 256, n_seeds = 1)),
    ]
    specs = filter(s -> s.label in wanted, specs)

    outdir = joinpath(JCODE_ROOT, "results", "pc_probe")
    isdir(outdir) || mkpath(outdir)

    summaries = NamedTuple[]
    for s in specs
        prob = s.build()
        # EPCM as configured for this problem, with optional ϑ overrides
        kw = NamedTuple()
        isempty(v0_ovr) || (kw = merge(kw, (vartheta_0 = parse(Float64, v0_ovr),)))
        isempty(v1_ovr) || (kw = merge(kw, (vartheta_1 = parse(Float64, v1_ovr),)))
        alg = EPCM(Symbol(s.label); kw...)

        ceil0 = vartheta0_ceiling(alg.gamma)
        admissible = (0 < alg.vartheta_1 < alg.vartheta_0 < ceil0) && (0 < alg.zeta < 2 / (2 + alg.gamma))
        admissible || println(tee, "  ⚠ $(s.label): ϑ params outside admissibility (ϑ_0=$(alg.vartheta_0) vs ceiling $(round(ceil0,digits=4))); diagnostic only.")

        init = first(get_initial_points(prob))
        records, conv, iters, fnat =
            run_epcm_probe(alg, prob, copy(init.x0);
                           maxiter = maxiter, native_tol_val = native_tol(s.label))

        csvpath = write_traj_csv(joinpath(outdir, "traj_$(s.label).csv"), records)
        st = summarize(records)
        push!(summaries, merge((label = s.label, dim = prob.dim, L = prob.metadata.L,
                                zeta = alg.zeta, vartheta_0 = alg.vartheta_0,
                                converged = conv, iters = iters, final_native = fnat), st))

        @printf(tee, "  %-3s dim=%-4d L=%.3f  ϑ_0=%.3f ζ=%.2f  iters=%-6d conv=%-5s  → %s\n",
                s.label, prob.dim, prob.metadata.L, alg.vartheta_0, alg.zeta,
                iters, string(conv), basename(csvpath))
    end

    # ── summary table ───────────────────────────────────────────────────────
    println(tee, "\n--- θ_k and stride statistics over each full trajectory ---")
    @printf(tee, "  %-3s %8s %8s %8s %8s | %8s %8s %8s | %8s %8s | %8s %8s\n",
            "P", "θ_mean", "θ_med", "θ_q90", "θ_max",
            "str_med", "str_q90", "str_max", "Δz/u_med", "Δz/u_q90",
            "PCactive", "clamp%")
    println(tee, "  " * "-" ^ 104)
    for r in summaries
        @printf(tee, "  %-3s %8.3f %8.3f %8.3f %8.3f | %8.3f %8.3f %8.3f | %8.3f %8.3f | %7.1f%% %7.1f%%\n",
                r.label, r.theta_mean, r.theta_med, r.theta_q90, r.theta_max,
                r.stride_med, r.stride_q90, r.stride_max, r.rdu_med, r.rdu_q90,
                100 * r.frac_pc_active, 100 * r.frac_clamped)
    end
    println(tee, "\n  θ = PC coefficient (clamped);  stride = 1+ζθ_k (effective step multiplier vs base FRB);")
    println(tee, "  Δz/u = ‖z_{k-1}−z_k‖/‖u_k‖ (drives the admissibility cap θ_k ≤ μϑ_0·Δz/u, μϑ_0<½);")
    println(tee, "  PCactive = % iters with θ_k>0;  clamp% = % iters with raw θ_k<0 (clamp engaged).")

    # ── verdict heuristic ─────────────────────────────────────────────────────
    println(tee, "\n--- VERDICT (heuristic; inspect CSVs + medians for the real call) ---")
    bylabel = Dict(r.label => r for r in summaries)
    if haskey(bylabel, "P3")
        p3 = bylabel["P3"]
        others = [bylabel[l] for l in ("P2", "P4") if haskey(bylabel, l)]
        othermax = isempty(others) ? NaN : maximum(o.stride_med for o in others)
        if p3.stride_med >= 1.5 && (isnan(othermax) || p3.stride_med >= 1.3 * othermax)
            println(tee, "  → SUPPORTS the PC-niche hypothesis: P3 median stride = $(round(p3.stride_med,digits=3))",
                         isnan(othermax) ? "" : " vs P2/P4 ≤ $(round(othermax,digits=3))",
                         ". The PC step does real work on structured/low-rank operator variation.")
        elseif p3.stride_med <= 1.05
            println(tee, "  → KILLS the PC angle: even P3 median stride ≈ $(round(p3.stride_med,digits=3)) (≈1).",
                         " The PC step is inert; EPCM has no per-iteration edge. Stop pursuing PC.")
        else
            println(tee, "  → INCONCLUSIVE: P3 median stride = $(round(p3.stride_med,digits=3)).",
                         " Modest PC effect; inspect trajectory CSVs (does θ_k persist or only spike near convergence?).")
        end
    else
        println(tee, "  (P3 not in the run; re-run including P3 for the decisive comparison.)")
    end

    teardown_logging(tee, logpath)
    return summaries
end

main()
