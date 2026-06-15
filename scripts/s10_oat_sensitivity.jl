# ============================================================================
# s10: One-At-A-Time (OAT) parameter sensitivity sweep — one (method, problem)
# ============================================================================
#
# Screens which parameters of a solver matter on a given problem, and finds
# rough good values, by varying ONE parameter at a time around the method's
# per-problem preset (the "center" = source-paper defaults) while holding the
# rest fixed. Each (parameter, value) is one config; it is run on `--seeds`
# initial points and aggregated to median-iterations-to-converge. Results land
# in the DB tagged `script="s10"`; the joint optimization is s20 (LHS).
#
# Policy (shared with s20/s28/s30):
#   - Convergence rule: each problem's NATIVE source-paper criterion < native_tol
#     (D1 decision, supersedes B1's universal R_n). R_n/R̃_n still recorded.
#   - Budget: maxiter = universal_maxiter(problem, eps) — tolerance-aware,
#     = 3000 (P1/P2) / 1000 (P3) at eps=1e-6. Same as the benchmark (so we
#     optimize iterations-within-the-reported-budget).
#   - Non-converged runs (flag ≠ :converged) are excluded from the median and
#     shown via the conv-count column; constraint-violating grid points throw
#     in `solve` and are recorded as flag=:error.
#   - Default screening dim is the MID dim per problem (P1 m=20, P2 m=300,
#     P3 K=100); conditioning (and the τ_0 sweet spot) is dim-dependent, so the
#     dim is configurable. History is NOT collected (tuning runs are lean).
#
# CLI:
#   julia --project=jcode jcode/scripts/s10_oat_sensitivity.jl --method=EPCM --problem=P1
#   flags: --dim=D   --seeds=N   --quick (seeds=2, small dim)
#          --force (re-run rows already in the DB)   --export (CSV)   --verbose
#
# Output: console + jcode/results/logs/log_s10_oat_<method>_<problem>_<ts>.txt

include(joinpath(@__DIR__, "..", "src", "includes.jl"))

const METHOD_TYPES = Dict("EPCM" => EPCM, "MTTM" => MTTM, "IMTTM" => IMTTM,
                          "SFRBM" => SFRBM, "IFRAB" => IFRAB)
# Suite (2026-05-28): P1=Volterra SFP (dim=grid N), P2=ℓ1+quad (m), P3=control
# (K=100 fixed), P4=LASSO (signal N). Stopping = native source-paper criterion
# (D1) via NativeResStopping(prob.native_residual, native_tol(problem)).
const MID_DIM   = Dict("P1" => 128, "P2" => 300, "P3" => 100, "P4" => 512)
const QUICK_DIM = Dict("P1" => 32,  "P2" => 100, "P3" => 100, "P4" => 64)

# Relative multipliers for log-scaled params (tau_0, lambda_0): grid = center .* this.
const LOG_MULT = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0]

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

"""
    with_overrides(alg::T, ov::NamedTuple) -> T

Rebuild `alg` with the fields in `ov` overridden and all others copied from
`alg`. Uses the keyword constructor, so `ov`'s keys must be fields of `T`.
"""
function with_overrides(alg::T, ov::NamedTuple) where {T<:AbstractAlgorithm}
    base = NamedTuple{fieldnames(T)}(map(f -> getfield(alg, f), fieldnames(T)))
    return T(; merge(base, ov)...)
end

"Build OAT points for one scalar field: list of (label, override-NamedTuple)."
_pts(param::Symbol, values) =
    [(string(param) * "=" * string(v), NamedTuple{(param,)}((v,))) for v in values]

# ── Per-method OAT groups: Vector of (group_name, Vector{(point_label, overrides)}) ──
# Grids are designed to stay inside each method's admissibility envelope when the
# other params sit at the center; any straggler that violates a constraint is
# caught in `solve` and recorded as flag=:error (see `run_one`).

function oat_groups(alg::EPCM)
    return [
        ("tau_0",       _pts(:tau_0, alg.tau_0 .* LOG_MULT)),
        ("sigma_exp",   _pts(:sigma_exp, [0.6, 0.75, 0.9, 1.0])),
        ("sigma_scale", _pts(:sigma_scale, [0.1, 0.25, 0.5, 1.0])),
        ("delta",       vcat(_pts(:delta_exp, [1.25, 1.5, 2.0, 2.5, 3.0]),
                             [("delta_rule=:zero", (delta_rule = :zero,))])),
        ("gamma",       _pts(:gamma, [0.25, 0.5, 1.0, 1.5, 1.9])),
        ("vartheta_0",  _pts(:vartheta_0, [0.15, 0.2, 0.3, 0.4])),
        ("vartheta_1",  _pts(:vartheta_1, [0.05, 0.1, 0.15, 0.19])),
        ("zeta",        _pts(:zeta, [0.2, 0.4, 0.5, 0.6, 0.66])),
    ]
end

function oat_groups(alg::MTTM)
    return [
        ("lambda_0", _pts(:lambda_0, alg.lambda_0 .* LOG_MULT)),
        ("mu",       _pts(:mu, [0.3, 0.5, 0.7, 0.85, 0.95])),
    ]
end

function oat_groups(alg::IMTTM)
    return [
        ("lambda_0", _pts(:lambda_0, alg.lambda_0 .* LOG_MULT)),
        ("mu",       _pts(:mu, [0.3, 0.5, 0.7, 0.9])),
        ("theta",    _pts(:theta, [0.1, 0.3, 0.5, 0.7, 0.9])),
    ]
end

function oat_groups(alg::SFRBM)
    # Yao keeps λ_{-1} = λ_0; sweep them together. μ grid respects 2μ < 1/(1+θ)
    # at θ = center (=10 ⟹ μ < 1/22 ≈ 0.0455).
    lam = [("lambda_0=" * string(v), (lambda_0 = v, lambda_minus1 = v))
           for v in alg.lambda_0 .* LOG_MULT]
    return [
        ("lambda_0", lam),
        ("mu",       _pts(:mu, [1.0e-4, 1.0e-3, 1.0e-2, 0.04])),
        ("theta",    _pts(:theta, [1.0, 5.0, 10.0, 20.0])),
    ]
end

function oat_groups(alg::IFRAB)
    # vartheta_bar grid stays < (1/2 − r̄)/2 = 0.1 at r̄ = center (0.3);
    # rbar grid stays < 0.42 so ϑ̄ = center (0.04) remains admissible.
    return [
        ("delta_0",      _pts(:delta_0, [0.05, 0.1, 0.2, 0.4])),
        ("delta_1",      _pts(:delta_1, [0.1, 0.3, 0.5, 0.7])),
        ("rbar",         _pts(:rbar, [0.1, 0.2, 0.3, 0.4])),
        ("vartheta_bar", _pts(:vartheta_bar, [0.0, 0.02, 0.04, 0.06])),
    ]
end

"""
    run_one(alg, prob, ip, eps, maxiter) -> (SolverResult, native::Float64)

Solve a single (config, initial-point) run with no history. Catches the
admissibility `ArgumentError` thrown by `solve` for out-of-envelope grid
points, returning a flag=:error result so the OAT sweep continues.
"""
function run_one(alg, prob, ip, eps, maxiter)
    # eps = the problem's native tolerance; stop on the native criterion (D1).
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

"Aggregate a config's s10 result rows: (n_total, n_conv, med_iters, med_cpu)."
function summarize_config(db, hash, problem, dim)
    df = DBInterface.execute(db, """
        SELECT converged, iterations, cpu_time FROM results
        WHERE script='s10' AND config_hash=? AND problem=? AND dimension=?
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

    method_str  = get(opts, "method", "")
    problem_str = get(opts, "problem", "")
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

    logpath, tee, _ = setup_logging("s10_oat_$(method_str)_$(problem_str)")
    eps     = native_tol(problem_str)            # native source-paper tolerance (D1)
    maxiter = universal_maxiter(problem_str, eps)
    prob_id = parse(Int, problem_str[2:end])
    prob    = get_problem(prob_id; dim = dim, n_seeds = n_seeds)
    T       = METHOD_TYPES[method_str]
    center  = T(Symbol(problem_str))
    inits   = prob.initial_points
    groups  = oat_groups(center)

    println(tee, "=" ^ 78)
    println(tee, "  s10 OAT sensitivity — $method_str on $problem_str   $(Dates.now())")
    println(tee, "=" ^ 78)
    @printf(tee, "  problem      : %s (dim=%d, L=%.4f)\n", prob.name, prob.dim, prob.metadata.L)
    @printf(tee, "  eps          : %.1e\n", eps)
    @printf(tee, "  maxiter      : %d  (universal_maxiter; tolerance-aware)\n", maxiter)
    @printf(tee, "  seeds        : %d  (seed1..seed%d)\n", n_seeds, n_seeds)
    @printf(tee, "  center preset: %s(:%s)\n", method_str, problem_str)
    for f in fieldnames(T)
        @printf(tee, "      .%-13s = %s\n", string(f), repr(getfield(center, f)))
    end
    @printf(tee, "  OAT groups   : %d  (%s)\n",
            length(groups), join(first.(groups), ", "))

    db     = open_db()
    run_id = "s10_$(Dates.format(now(), "yyyymmdd_HHMMSS"))"

    # ── Run the sweep ──────────────────────────────────────────────────────
    println(tee, "\n--- Running OAT points ---")
    registry = Tuple{String,String,String}[]   # (group, point_label, config_hash)
    n_runs = 0
    n_skip = 0
    for (gname, points) in groups
        for (plabel, ov) in points
            alg        = with_overrides(center, ov)
            hash, hin  = make_config_hash(alg, problem_str, eps, maxiter)
            push!(registry, (gname, plabel, hash))
            ensure_config!(db, alg, problem_str, eps, maxiter, hash, hin)
            for ip in inits
                if !force && is_done(db, hash, problem_str, dim, ip.label; script = "s10")
                    n_skip += 1
                    continue
                end
                result, native = run_one(alg, prob, ip, eps, maxiter)
                insert_result!(db, hash, problem_str, dim, ip.label, ip.seed_idx,
                               run_id, result; script = "s10", native_residual = native)
                n_runs += 1
                verbose && @printf(tee, "    %-12s %-24s %-7s : iters=%d flag=%s\n",
                                   gname, plabel, ip.label, result.iterations, result.flag)
            end
        end
        @printf(tee, "  %-12s : %d points\n", gname, length(points))
    end
    @printf(tee, "  (%d new runs, %d skipped as already-done)\n", n_runs, n_skip)

    # ── Summary: per-group median-iters table, best value flagged ──────────
    println(tee, "\n--- OAT summary (median iterations over converged seeds) ---")
    gnames = unique(first.(registry))
    for g in gnames
        pts  = [(lbl, h) for (gg, lbl, h) in registry if gg == g]
        rows = [(lbl, summarize_config(db, h, problem_str, dim)) for (lbl, h) in pts]
        best_i, best_iters = 0, Inf
        for (i, (_, s)) in enumerate(rows)
            if s.n_conv > 0 && s.med_iters < best_iters
                best_i, best_iters = i, s.med_iters
            end
        end
        @printf(tee, "\n  [%s]\n", g)
        @printf(tee, "    %-26s %8s %10s %12s\n", "value", "conv", "med_iter", "med_cpu(s)")
        for (i, (lbl, s)) in enumerate(rows)
            iters_str = s.n_conv > 0 ? @sprintf("%.1f", s.med_iters) : "—"
            cpu_str   = s.n_conv > 0 ? @sprintf("%.4f", s.med_cpu)   : "—"
            mark      = i == best_i ? "  <- best" : ""
            @printf(tee, "    %-26s %4d/%-3d %10s %12s%s\n",
                    lbl, s.n_conv, s.n_total, iters_str, cpu_str, mark)
        end
        if best_i == 0
            println(tee, "    (no value converged within maxiter — likely needs a larger budget or different center)")
        end
    end

    if do_export
        expath = joinpath(JCODE_ROOT, "results", "s10_oat_$(method_str)_$(problem_str).csv")
        n = export_results_csv(db, expath)
        @printf(tee, "\n  Exported %d total result rows to %s\n", n, expath)
    end

    println(tee, "\n" * "=" ^ 78)
    println(tee, "  s10 OAT complete: $method_str / $problem_str / dim=$dim")
    println(tee, "=" ^ 78)

    teardown_logging(tee, logpath)
    return true
end

main()
