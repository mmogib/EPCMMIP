# ============================================================================
# s20: Latin-Hypercube JOINT parameter search — one (method, problem)
# ============================================================================
#
# Tunes a solver's parameters JOINTLY (unlike s10's one-at-a-time screen) by
# sampling its admissible parameter envelope with a Latin Hypercube design,
# running each sample on `--seeds` initial points, and ranking configs by
# median-iterations-to-converge. Results land in the DB tagged `script="s20"`;
# `s25_promote_best_params.jl` reads the best s20 row per cell into the presets.
#
# WHY joint (not OAT): for the proposed EPCM the competitive regime is a JOINT
# effect — weak anchoring (small sigma_scale) + a large step (vartheta_0 near
# its √(γ/(2(2γ+1))) limit) + a large contraction ζ together. No single-param
# move off the conservative manuscript centre reaches it, so s10 OAT reports
# 0/2 for EPCM/Volterra while a joint sample reaches tolerance in ~hundreds of
# iters (see notes/plan_review_findings.md "EPCM competitiveness"). LHS spreads
# samples across all dims at once, so it finds that corner.
#
# ADMISSIBILITY BY CONSTRUCTION: every coupled constraint is honoured by
# mapping a unit coordinate to a FRACTION of the (possibly other-param-
# dependent) bound, so no sample is wasted on a flag=:error reject:
#   EPCM   ϑ_0 = f·√(γ/(2(2γ+1))),  ϑ_1 = f·ϑ_0,  ζ = f·2/(2+γ),
#          σ_0 = sigma_scale/2^sigma_exp < 1 (sigma_scale ≤ 1 < 2^sigma_exp),
#          sigma_exp ∈ (0,1],  delta_exp > 1.
#   SFRBM  μ = f·1/(2(1+θ))  ⟹ 2μ < 1/(1+θ).
#   IFRAB  ϑ̄ = f·(1/2−r̄)/2,  r̄ ∈ (0,1/2).
# (A straggler that still violates a bound is caught in `solve` and recorded
#  flag=:error, exactly as in s10.)
#
# Policy (shared with s10/s28/s30):
#   - Convergence rule: each problem's NATIVE source-paper criterion < native_tol
#     (D1; supersedes B1's universal R_n). R_n/R̃_n still recorded.
#   - Budget: maxiter = universal_maxiter(problem, eps) (tolerance-aware;
#     3000/3000/1000/3000 for P1/P2/P3/P4 at eps=1e-6).
#   - Non-converged runs (flag ≠ :converged) are excluded from the median.
#   - Default screening dim is the MID dim per problem (configurable). History
#     is NOT collected (tuning runs are lean).
#
# REPRODUCIBILITY: the LHS design is deterministic given (LHS_SEED, n_samples,
# dims), so re-running skips already-stored configs (`--force` to redo).
# Changing `--samples` regenerates the whole design (different bins ⟹ different
# config hashes); pick a value and stick with it for a given cell.
#
# CLI:
#   julia --project=jcode jcode/scripts/s20_lhs_parameter_search.jl --method=EPCM --problem=P1
#   flags: --dim=D   --seeds=N   --samples=M
#          --quick (seeds=2, small dim, fewer samples)
#          --force (re-run rows already in the DB)   --export (CSV)   --verbose
#
# Output: console + jcode/results/logs/log_s20_lhs_<method>_<problem>_<ts>.txt

include(joinpath(@__DIR__, "..", "src", "includes.jl"))

const METHOD_TYPES = Dict("EPCM" => EPCM, "MTTM" => MTTM, "IMTTM" => IMTTM,
                          "SFRBM" => SFRBM, "IFRAB" => IFRAB)
# Suite (2026-05-28): P1=Volterra SFP (dim=grid N), P2=ℓ1+quad (m), P3=control
# (K=100 fixed), P4=LASSO (signal N). Same dims as s10 so the OAT screen and
# the LHS search optimize at the same conditioning.
const MID_DIM   = Dict("P1" => 128, "P2" => 300, "P3" => 100, "P4" => 512)
const QUICK_DIM = Dict("P1" => 32,  "P2" => 100, "P3" => 100, "P4" => 64)

"Fixed RNG seed for the LHS design (reproducible configs ⟹ DB skip works)."
const LHS_SEED = 20260528

# ============================================================================
# Helpers
# ============================================================================

"Parse `--key=value` opts and bare `--flag`s from ARGS."
function parse_cli(args)
    opts  = Dict{String,String}()
    flags = Set{String}()
    for a in args
        startswith(a, "--") || continue
        body = a[3:end]
        if occursin('=', body)
            k, v = split(body, '=', limit = 2)
            opts[k] = v
        else
            push!(flags, body)
        end
    end
    return opts, flags
end

"Median of a real vector (∞ for empty)."
function _median(v::AbstractVector{<:Real})
    isempty(v) && return Inf
    s = sort(collect(float.(v)))
    n = length(s)
    return isodd(n) ? s[(n + 1) ÷ 2] : (s[n ÷ 2] + s[n ÷ 2 + 1]) / 2
end

"Map unit u∈[0,1] to a LINEAR range [a,b]."
@inline _lin(u, a, b) = a + u * (b - a)

"Map unit u∈[0,1] to a LOG range [a,b] (a,b > 0)."
@inline _log(u, a, b) = a * (b / a)^u

"""
    lhs_unit(rng, n, d) -> n×d Matrix

Latin Hypercube design on [0,1]^d: each column stratifies [0,1] into `n` equal
bins, randomly permutes the bin order, and jitters within each bin. Every
1-D projection is therefore perfectly stratified (one point per bin).
"""
function lhs_unit(rng, n::Int, d::Int)
    H = Matrix{Float64}(undef, n, d)
    for j in 1:d
        perm = randperm(rng, n)
        @inbounds for i in 1:n
            H[i, j] = (perm[i] - 1 + rand(rng)) / n
        end
    end
    return H
end

"""
    with_overrides(alg::T, ov::NamedTuple) -> T

Rebuild `alg` with the fields in `ov` overridden and all others copied. Uses
the keyword constructor, so `ov`'s keys must be fields of `T`. (Non-sampled
rule fields — delta_rule/sigma_rule/alpha_rule/… — come from the preset.)
"""
function with_overrides(alg::T, ov::NamedTuple) where {T<:AbstractAlgorithm}
    base = NamedTuple{fieldnames(T)}(map(f -> getfield(alg, f), fieldnames(T)))
    return T(; merge(base, ov)...)
end

# ── Per-method LHS spec: (dims, build(u)->overrides::NamedTuple, labels) ─────
# `build` maps a length-`dims` unit vector to the override NamedTuple, mapping
# coupled constraints to fractions of their bounds (admissible by construction).

function lhs_spec(::EPCM, problem_str::AbstractString)
    tau_c  = EPCM_PRESETS[Symbol(problem_str)].tau_0    # per-problem τ_0 centre
    # τ_0 has an UPPER limit: too large ⟹ the forward term τ·B overshoots and
    # the iterate blows up to NaN before the adaptive rule corrects. On P1
    # (Volterra; the T*T operator amplifies) the ceiling is low (~1.5×centre,
    # confirmed by --quick: every NaN sample had τ_0 ≳ 1.5). The other problems
    # are not blow-up-prone, so keep the wide range there. NaN samples are
    # discarded (flag=:nan), but probing deep into the NaN zone just wastes the
    # budget, so we cap the upper multiplier on P1.
    tau_lo, tau_hi = problem_str == "P1" ? (0.05, 1.5) : (0.1, 10.0)
    # ANCHORING (sigma_exp, sigma_scale) is restricted to the WEAK regime where
    # EPCM converges: strong anchoring (small sigma_exp, large sigma_scale)
    # decelerates so much that nothing finishes within maxiter (the manuscript
    # centre, sigma_scale=1, is exactly this stall). We search for the FASTEST
    # config AMONG converging ones, so the auxiliary knobs stay in-regime while
    # the SPEED knobs (γ, ϑ_0, ϑ_1, ζ, delta_exp) keep their full envelope.
    labels = ["gamma", "vartheta_0", "vartheta_1", "zeta", "tau_0",
              "sigma_exp", "sigma_scale", "delta_exp"]
    build = function (u)
        gamma       = _lin(u[1], 0.5, 3.0)
        bound_θ0    = sqrt(gamma / (2 * (2 * gamma + 1)))    # ϑ_0 ceiling
        vartheta_0  = _lin(u[2], 0.30, 0.97) * bound_θ0
        vartheta_1  = _lin(u[3], 0.10, 0.97) * vartheta_0    # < ϑ_0
        zeta        = _lin(u[4], 0.30, 0.97) * (2 / (2 + gamma))
        tau_0       = _log(u[5], tau_c * tau_lo, tau_c * tau_hi)
        sigma_exp   = _lin(u[6], 0.70, 1.00)                 # weak anchoring
        sigma_scale = _log(u[7], 0.01, 0.30)                 # weak anchoring
        delta_exp   = _lin(u[8], 1.10, 3.00)
        return (gamma = gamma, vartheta_0 = vartheta_0, vartheta_1 = vartheta_1,
                zeta = zeta, tau_0 = tau_0, sigma_exp = sigma_exp,
                sigma_scale = sigma_scale, delta_exp = delta_exp)
    end
    return (length(labels), build, labels)
end

function lhs_spec(::MTTM, problem_str::AbstractString)
    lam_c  = MTTM_PRESETS[Symbol(problem_str)].lambda_0      # 7.55
    labels = ["lambda_0", "mu"]
    build = u -> (lambda_0 = _log(u[1], lam_c * 0.1, lam_c * 10.0),
                  mu       = _lin(u[2], 0.10, 0.95))
    return (length(labels), build, labels)
end

function lhs_spec(::IMTTM, problem_str::AbstractString)
    lam_c  = IMTTM_PRESETS[Symbol(problem_str)].lambda_0     # 1.0
    labels = ["lambda_0", "mu", "theta"]
    build = u -> (lambda_0 = _log(u[1], lam_c * 0.1, lam_c * 10.0),
                  mu       = _lin(u[2], 0.10, 0.95),
                  theta    = _lin(u[3], 0.05, 0.95))
    return (length(labels), build, labels)
end

function lhs_spec(::SFRBM, problem_str::AbstractString)
    lam_c  = SFRBM_PRESETS[Symbol(problem_str)].lambda_0     # 1e-4
    labels = ["lambda", "theta", "mu"]                       # λ_0 = λ_{-1}
    build = function (u)
        lam      = _log(u[1], lam_c * 0.1, lam_c * 100.0)    # 1e-5 .. 1e-2
        theta    = _log(u[2], 0.5, 20.0)
        bound_mu = 1.0 / (2 * (1 + theta))                   # 2μ < 1/(1+θ)
        mu       = _lin(u[3], 0.05, 0.95) * bound_mu
        return (lambda_minus1 = lam, lambda_0 = lam, theta = theta, mu = mu)
    end
    return (length(labels), build, labels)
end

function lhs_spec(::IFRAB, problem_str::AbstractString)
    labels = ["delta_0", "delta_1", "rbar", "vartheta_bar"]
    build = function (u)
        delta_0      = _lin(u[1], 0.05, 0.50)
        delta_1      = _lin(u[2], 0.10, 0.70)
        rbar         = _lin(u[3], 0.05, 0.49)                # ∈ (0,1/2)
        bound_v      = (0.5 - rbar) / 2                       # ϑ̄ ceiling
        vartheta_bar = _lin(u[4], 0.0, 0.95) * bound_v
        return (delta_0 = delta_0, delta_1 = delta_1, rbar = rbar,
                vartheta_bar = vartheta_bar)
    end
    return (length(labels), build, labels)
end

# ── Compact tuned-field formatter per method (for the ranking table) ─────────
_tuned_str(a::EPCM)  = @sprintf("γ=%.2f ϑ0=%.3f ϑ1=%.3f ζ=%.3f τ0=%.3g sx=%.2f ss=%.3g dx=%.2f",
                                a.gamma, a.vartheta_0, a.vartheta_1, a.zeta,
                                a.tau_0, a.sigma_exp, a.sigma_scale, a.delta_exp)
_tuned_str(a::MTTM)  = @sprintf("λ0=%.3g μ=%.3f", a.lambda_0, a.mu)
_tuned_str(a::IMTTM) = @sprintf("λ0=%.3g μ=%.3f θ=%.3f", a.lambda_0, a.mu, a.theta)
_tuned_str(a::SFRBM) = @sprintf("λ=%.3g θ=%.3f μ=%.4g", a.lambda_0, a.theta, a.mu)
_tuned_str(a::IFRAB) = @sprintf("δ0=%.3f δ1=%.3f r̄=%.3f ϑ̄=%.4f",
                                a.delta_0, a.delta_1, a.rbar, a.vartheta_bar)

"""
    default_samples(dims; quick) -> Int

LHS size scaled to the parameter-space dimension: ≈20·dims (clamped to
[80,200]) for the full sweep, ≈3·dims (≥16) for `--quick`. Higher-dim methods
(EPCM, 8) get more samples than low-dim ones (MTTM, 2).
"""
default_samples(dims::Int; quick::Bool) =
    quick ? max(3 * dims, 16) : clamp(20 * dims, 80, 200)

"""
    run_one(alg, prob, ip, eps, maxiter) -> (SolverResult, native::Float64)

Solve a single (config, initial-point) run with no history, stopping on the
NATIVE criterion (D1). Catches the admissibility `ArgumentError` thrown by
`solve` for any out-of-envelope straggler, returning a flag=:error result so
the sweep continues.
"""
function run_one(alg, prob, ip, eps, maxiter)
    stopping = (NativeResStopping(prob.native_residual, eps), MaxIterStopping(maxiter), NanStopping())
    nrec = NativeResRecorder(prob.native_residual)
    try
        result = solve(alg, prob, copy(ip.x0); stopping = stopping, observers = (nrec,))
        return (result, nrec.value)
    catch e
        e isa ArgumentError || rethrow(e)
        return (make_result(converged = false, iterations = 0, f_evals = 0,
                            cpu_time = 0.0, x = copy(ip.x0), flag = :error), NaN)
    end
end

"Aggregate a config's s20 result rows: (n_total, n_conv, med_iters, med_cpu)."
function summarize_config(db, hash, problem, dim)
    df = DBInterface.execute(db, """
        SELECT converged, iterations, cpu_time FROM results
        WHERE script='s20' AND config_hash=? AND problem=? AND dimension=?
    """, (hash, problem, dim)) |> DataFrame
    conv_iters = Float64[]
    conv_cpu   = Float64[]
    for r in eachrow(df)
        if r.converged == 1
            push!(conv_iters, float(r.iterations))
            push!(conv_cpu, float(r.cpu_time))
        end
    end
    return (n_total = nrow(df), n_conv = length(conv_iters),
            med_iters = _median(conv_iters), med_cpu = _median(conv_cpu))
end

# ============================================================================
# Main
# ============================================================================

function main()
    opts, flags = parse_cli(ARGS)

    method_str    = get(opts, "method", "")
    problem_str   = get(opts, "problem", "")
    known_methods = join(sort(collect(keys(METHOD_TYPES))), ", ")
    haskey(METHOD_TYPES, method_str) ||
        error("--method must be one of {$known_methods}; got '$method_str'")
    problem_str in ("P1", "P2", "P3", "P4") ||
        error("--problem must be P1|P2|P3|P4; got '$problem_str'")

    quick     = "quick" in flags
    force     = "force" in flags
    do_export = "export" in flags
    verbose   = "verbose" in flags

    n_seeds = haskey(opts, "seeds") ? parse(Int, opts["seeds"]) : (quick ? 2 : 5)
    dim = if problem_str == "P3"
        100                                   # P3 is fixed K=100
    elseif haskey(opts, "dim")
        parse(Int, opts["dim"])
    else
        quick ? QUICK_DIM[problem_str] : MID_DIM[problem_str]
    end

    logpath, tee, _ = setup_logging("s20_lhs_$(method_str)_$(problem_str)")
    # eps defaults to the native source-paper tolerance (D1); --eps overrides it. eps drives
    # BOTH the stopping threshold (NativeResStopping) AND the tolerance-aware budget
    # (universal_maxiter) — preserving the "same budget across all methods for a given
    # tolerance" fairness discipline. eps enters make_config_hash, so override runs are
    # distinct configs from default-tolerance runs (no row collision).
    eps     = haskey(opts, "eps") ? parse(Float64, opts["eps"]) : native_tol(problem_str)
    maxiter = universal_maxiter(problem_str, eps)
    prob_id = parse(Int, problem_str[2:end])
    prob    = get_problem(prob_id; dim = dim, n_seeds = n_seeds)
    T       = METHOD_TYPES[method_str]
    center  = T(Symbol(problem_str))
    inits   = prob.initial_points

    dims, build, labels = lhs_spec(center, problem_str)
    n_samples = haskey(opts, "samples") ? parse(Int, opts["samples"]) :
                default_samples(dims; quick = quick)

    println(tee, "=" ^ 78)
    println(tee, "  s20 LHS joint search — $method_str on $problem_str   $(Dates.now())")
    println(tee, "=" ^ 78)
    @printf(tee, "  problem      : %s (dim=%d, L=%.4f)\n", prob.name, prob.dim, prob.metadata.L)
    @printf(tee, "  eps          : %.1e\n", eps)
    @printf(tee, "  maxiter      : %d  (universal_maxiter; tolerance-aware)\n", maxiter)
    @printf(tee, "  seeds        : %d  (seed1..seed%d)\n", n_seeds, n_seeds)
    @printf(tee, "  LHS dims     : %d  (%s)\n", dims, join(labels, ", "))
    @printf(tee, "  LHS samples  : %d  (+1 preset centre; seed=%d)\n", n_samples, LHS_SEED)
    @printf(tee, "  centre preset: %s(:%s)  [%s]\n", method_str, problem_str, _tuned_str(center))

    # ── Build the config list: preset centre (idx 0) + LHS samples ──────────
    rng = MersenneTwister(LHS_SEED)
    H   = lhs_unit(rng, n_samples, dims)
    algs = Vector{Tuple{Int,T}}()              # (sample_idx, alg); idx 0 = centre
    push!(algs, (0, center))
    for i in 1:n_samples
        push!(algs, (i, with_overrides(center, build(@view H[i, :]))))
    end

    db     = open_db()
    run_id = "s20_$(Dates.format(now(), "yyyymmdd_HHMMSS"))"

    # ── Run the sweep ────────────────────────────────────────────────────────
    println(tee, "\n--- Running $(length(algs)) configs × $(length(inits)) seeds ---")
    registry = Tuple{Int,String,T}[]           # (sample_idx, config_hash, alg)
    n_runs = 0
    n_skip = 0
    n_done = 0
    n_cfg  = length(algs)
    for (idx, alg) in algs
        hash, hin = make_config_hash(alg, problem_str, eps, maxiter)
        push!(registry, (idx, hash, alg))
        ensure_config!(db, alg, problem_str, eps, maxiter, hash, hin)
        for ip in inits
            if !force && is_done(db, hash, problem_str, dim, ip.label; script = "s20")
                n_skip += 1
                continue
            end
            result, native = run_one(alg, prob, ip, eps, maxiter)
            insert_result!(db, hash, problem_str, dim, ip.label, ip.seed_idx,
                           run_id, result; script = "s20", native_residual = native)
            n_runs += 1
            verbose && @printf(tee, "    [%4d/%-4d] %-7s : iters=%d flag=%s\n",
                               idx, n_cfg - 1, ip.label, result.iterations, result.flag)
        end
        n_done += 1
        if !verbose && (n_done % max(1, n_cfg ÷ 10) == 0 || n_done == n_cfg)
            @printf(tee, "  ... %d/%d configs done (%d runs, %d skipped)\n",
                    n_done, n_cfg, n_runs, n_skip)
        end
    end
    @printf(tee, "  (%d new runs, %d skipped as already-done)\n", n_runs, n_skip)

    # ── Rank configs by median iterations (converged seeds only) ────────────
    rows = [(idx, hash, alg, summarize_config(db, hash, problem_str, dim))
            for (idx, hash, alg) in registry]
    # Sort key: converged configs first, by med_iters asc; ties by more conv.
    sort!(rows, by = r -> (r[4].n_conv > 0 ? r[4].med_iters : Inf,
                           -r[4].n_conv, r[1]))

    topK = min(15, length(rows))
    println(tee, "\n--- Top $topK configs (median iters over converged seeds) ---")
    @printf(tee, "  %5s %8s %10s %12s   %s\n", "idx", "conv", "med_iter", "med_cpu(s)", "params")
    for r in rows[1:topK]
        idx, _, alg, s = r
        iters_str = s.n_conv > 0 ? @sprintf("%.1f", s.med_iters) : "—"
        cpu_str   = s.n_conv > 0 ? @sprintf("%.4f", s.med_cpu)   : "—"
        tag       = idx == 0 ? "centre" : string(idx)
        @printf(tee, "  %5s %4d/%-3d %10s %12s   %s\n",
                tag, s.n_conv, s.n_total, iters_str, cpu_str, _tuned_str(alg))
    end

    # ── Best config: full paste-ready params (for s25 / manual inspection) ──
    best = rows[1]
    if best[4].n_conv == 0
        println(tee, "\n  (NO config converged within maxiter — widen ranges, raise --samples, or check the problem.)")
    else
        _, best_hash, best_alg, best_s = best
        best_nt = NamedTuple{fieldnames(T)}(map(f -> getfield(best_alg, f), fieldnames(T)))
        println(tee, "\n--- Best config ---")
        @printf(tee, "  sample idx   : %s\n", best[1] == 0 ? "centre" : string(best[1]))
        @printf(tee, "  config_hash  : %s\n", best_hash)
        @printf(tee, "  med_iters    : %.1f   (%d/%d seeds converged)\n",
                best_s.med_iters, best_s.n_conv, best_s.n_total)
        @printf(tee, "  med_cpu (s)  : %.4f\n", best_s.med_cpu)
        println(tee, "  paste-ready  : ", repr(best_nt))
        # Quick read on whether the centre preset is competitive.
        centre_row = rows[findfirst(r -> r[1] == 0, rows)]
        if centre_row[4].n_conv > 0 && best[1] != 0
            @printf(tee, "  speedup vs centre: %.1f× (centre med_iters=%.1f)\n",
                    centre_row[4].med_iters / best_s.med_iters, centre_row[4].med_iters)
        elseif centre_row[4].n_conv == 0
            println(tee, "  (preset centre did NOT converge — LHS found a working regime the manuscript default misses.)")
        end
    end

    if do_export
        expath = joinpath(JCODE_ROOT, "results", "s20_lhs_$(method_str)_$(problem_str).csv")
        n = export_results_csv(db, expath)
        @printf(tee, "\n  Exported %d total result rows to %s\n", n, expath)
    end

    println(tee, "\n" * "=" ^ 78)
    println(tee, "  s20 LHS complete: $method_str / $problem_str / dim=$dim")
    println(tee, "=" ^ 78)

    teardown_logging(tee, logpath)
    return true
end

main()
