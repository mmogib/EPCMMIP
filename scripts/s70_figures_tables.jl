# ============================================================================
# s70: Figures & tables for the paper — from the s30 production benchmark
# ============================================================================
#
# Reads the s30 PRODUCTION rows (production=1; or production=0 with --quick) and
# produces, under results/figures/:
#   1. Per-problem comparison TABLES (console + LaTeX `tables.tex`):
#      median iterations / B-evals / CPU per (method, problem), best bolded,
#      DNC marked. The MTTM/P1 and MTTM/P5 α₁=1 zero-solution artifacts are
#      excluded (footnoted).
#   2. PERFORMANCE PROFILES (Dolan–Moré, BenchmarkProfiles) on three metrics —
#      iterations, B-evals, CPU — over all (problem, dim, seed) instances.
#      The B-eval profile is the single-call story (EPCM/IFRAB/SFRBM = 1 eval/iter
#      vs MTTM/IMTTM = 2/iter).
#   3. CONVERGENCE PANELS: native residual Rₙ vs iteration (log scale), one
#      subplot per problem, one line per method, from the `history` table
#      (the representative ref-dim/seed1 cell per method×problem).
#
# Metrics use the median over converged seeds & dims; a method with 0 converged
# seeds on a problem is "DNC" (and counts as a failure in the profiles).
#
# Usage:
#   julia --project=jcode jcode/scripts/s70_figures_tables.jl            # production=1
#   julia --project=jcode jcode/scripts/s70_figures_tables.jl --quick    # production=0
#   julia --project=jcode jcode/scripts/s70_figures_tables.jl --png      # also emit PNGs
# Output: results/figures/{tables.tex, profile_*.pdf, convergence_panels.pdf}
#
# NOTE: this script `using`s Plots + BenchmarkProfiles (heavy; ~first-call lag).
# The tables are written BEFORE any plotting, so a plotting-API hiccup never
# costs you the tables.

include(joinpath(@__DIR__, "..", "src", "includes.jl"))
using Plots, LaTeXStrings, BenchmarkProfiles
gr()

const S70_METHOD_ORDER = ["EPCM", "EFBFP", "AEFBFP", "MAEFBFP", "MTTM", "IMTTM", "IFRAB", "SFRBM", "VAFBS", "MDITSM", "MFRBSM", "RFBSM", "IRFBSM"]
const S70_PROBLEMS     = ["P1", "P2", "P3", "P4"]
# Known artifacts excluded from tables/profiles. MTTM/P1 and MTTM/P5: α₁=1 ⟹
# first iterate 0, which on zero-solution problems "converges" in ~2 iters
# trivially — not a real result.
const ARTIFACT_EXCLUDE = Set([("MTTM", "P1"), ("MTTM", "P5")])
# Robustness threshold: a method solving < ROBUST_FRAC of its (seed × dim) instances on a
# problem is "partial" — its median is daggered (†) and it is never bolded as best; the raw
# solved/total counts are reported separately in Table~\ref{tab:solved}.
const ROBUST_FRAC = 0.5
const FIGDIR = joinpath(JCODE_ROOT, "results", "figures")

# ── tiny ARG parser ──────────────────────────────────────────────────────────
function parse_cli(args)
    opts = Dict{String,String}(); flags = String[]
    for a in args
        startswith(a, "--") || continue
        kv = split(a[3:end], "="; limit = 2)
        length(kv) == 2 ? (opts[kv[1]] = kv[2]) : push!(flags, kv[1])
    end
    return opts, flags
end

# ── Data ─────────────────────────────────────────────────────────────────────
function load_results(db, production::Bool)
    df = DBInterface.execute(db, """
        SELECT c.method AS method, r.problem AS problem, r.dimension AS dim,
               r.init_point AS init, r.converged AS conv, r.iterations AS iters,
               r.f_evals AS fe, r.cpu_time AS cpu, c.eps AS eps, c.maxiter AS maxiter,
               r.config_hash AS config_hash
        FROM results r JOIN configs c ON r.config_hash = c.config_hash
        WHERE r.script='s30' AND r.production=?
    """, (production ? 1 : 0,)) |> DataFrame
    df  = filter(r -> is_canonical(r.problem, r.eps, r.maxiter), df)   # canonical config only
    cur = current_canonical_configs(db; production = production)
    return filter(r -> r.config_hash in cur, df)                       # pin to the live tuned config
end

"Median of metric `col` over the CONVERGED rows of (method m, problem P); Inf if none."
function med_conv(df, m::AbstractString, P::AbstractString, col::Symbol)
    (m, P) in ARTIFACT_EXCLUDE && return Inf
    sub = df[(df.method .== m) .& (df.problem .== P) .& (df.conv .== 1), :]
    nrow(sub) == 0 ? Inf : median(float.(sub[!, col]))
end

"(n_conv, n_total) for (method m, problem P)."
function conv_counts(df, m::AbstractString, P::AbstractString)
    (m, P) in ARTIFACT_EXCLUDE && return (0, 0)
    sub = df[(df.method .== m) .& (df.problem .== P), :]
    (sum(sub.conv .== 1), nrow(sub))
end

# ── 1. Tables (console + LaTeX) ──────────────────────────────────────────────
function print_console_tables(df, tee)
    println(tee, "\n" * "=" ^ 72)
    println(tee, "  Per-problem comparison (median over converged seeds & dims)")
    println(tee, "  DNC = no converged seed; excluded artifact(s): ",
            join(["$m/$p" for (m, p) in ARTIFACT_EXCLUDE], ", "))
    println(tee, "=" ^ 72)
    for P in S70_PROBLEMS
        println(tee, "\n[$P]")
        @printf(tee, "  %-6s %9s %10s %10s %12s\n", "method", "conv", "med_iter", "med_eval", "med_cpu(s)")
        for m in S70_METHOD_ORDER
            (m, P) in ARTIFACT_EXCLUDE && continue
            nc, nt = conv_counts(df, m, P)
            nt == 0 && continue
            mi = nc > 0 ? string(Int(round(med_conv(df, m, P, :iters)))) : "DNC"
            me = nc > 0 ? string(Int(round(med_conv(df, m, P, :fe))))    : "DNC"
            mc = nc > 0 ? @sprintf("%.4g", med_conv(df, m, P, :cpu))     : "—"
            @printf(tee, "  %-6s %5d/%-3d %10s %10s %12s\n", m, nc, nt, mi, me, mc)
        end
    end
end

"True if method `m` converged on at least ROBUST_FRAC of its instances on problem `P`."
function is_robust(df, m::AbstractString, P::AbstractString)
    nc, nt = conv_counts(df, m, P)
    nt > 0 && nc >= ROBUST_FRAC * nt
end

"One LaTeX tabular: methods × problems, cells = median of `metric` over solved instances.
Partial-convergence cells (solved < all) are daggered (†); `best` (bolded) is taken only over
ROBUST solvers (≥ ROBUST_FRAC solved), so a lucky-fast partial run is never bolded. DNC if none."
function latex_metric_table(io, df, metric::Symbol, fmt, caption::AbstractString, label::AbstractString)
    best = Dict(P => minimum((med_conv(df, m, P, metric) for m in S70_METHOD_ORDER if is_robust(df, m, P));
                             init = Inf) for P in S70_PROBLEMS)
    rows = String[]
    has_partial = false
    for m in S70_METHOD_ORDER
        cells = String[]
        for P in S70_PROBLEMS
            nc, nt = conv_counts(df, m, P)
            v = med_conv(df, m, P, metric)
            if (m, P) in ARTIFACT_EXCLUDE
                push!(cells, "--")
            elseif nc == 0 || !isfinite(v)
                push!(cells, "DNC")
            else
                s = fmt(v)
                if nc < nt                                  # partial convergence
                    s *= "\$^\\dagger\$"
                    has_partial = true
                end
                push!(cells, (is_robust(df, m, P) && v == best[P]) ? "\\textbf{$s}" : s)
            end
        end
        push!(rows, "$m & " * join(cells, " & ") * " \\\\")
    end
    println(io, "\\begin{table}[H]\\centering")
    println(io, "\\caption{$caption}\\label{$label}")
    println(io, "\\begin{tabular}{l" * "r"^length(S70_PROBLEMS) * "}")
    println(io, "\\toprule")
    println(io, "Method & " * join(S70_PROBLEMS, " & ") * " \\\\")
    println(io, "\\midrule")
    foreach(r -> println(io, r), rows)
    println(io, "\\bottomrule")
    println(io, "\\end{tabular}")
    has_partial && println(io, "\\par\\smallskip{\\footnotesize \$^\\dagger\$~median over a PARTIAL " *
                               "instance set --- the method solved only some seeds/dims; " *
                               "see Table~\\ref{tab:solved} for counts.}")
    println(io, "\\end{table}")
    println(io)
end

"LaTeX tabular of convergence robustness: cells = solved / total (seed × dim) instances."
function latex_solved_table(io, df)
    println(io, "\\begin{table}[H]\\centering")
    println(io, "\\caption{Convergence robustness: solved\\,/\\,total instances (seeds \$\\times\$ " *
                "dimensions) reaching the native tolerance. Medians in Tables~\\ref{tab:iters}--" *
                "\\ref{tab:cpu} are taken over each method's solved instances only.}\\label{tab:solved}")
    println(io, "\\begin{tabular}{l" * "r"^length(S70_PROBLEMS) * "}")
    println(io, "\\toprule")
    println(io, "Method & " * join(S70_PROBLEMS, " & ") * " \\\\")
    println(io, "\\midrule")
    for m in S70_METHOD_ORDER
        cells = String[]
        for P in S70_PROBLEMS
            if (m, P) in ARTIFACT_EXCLUDE
                push!(cells, "--")
            else
                nc, nt = conv_counts(df, m, P)
                push!(cells, "$nc/$nt")
            end
        end
        println(io, "$m & " * join(cells, " & ") * " \\\\")
    end
    println(io, "\\bottomrule")
    println(io, "\\end{tabular}")
    println(io, "\\end{table}")
    println(io)
end

function write_latex_tables(df, tee)
    path = joinpath(FIGDIR, "tables.tex")
    fmt_int = v -> string(Int(round(v)))
    fmt_cpu = v -> @sprintf("%.3g", v)
    open(path, "w") do io
        println(io, "% Auto-generated by s70_figures_tables.jl. Median over converged seeds & dims.")
        println(io, "% MTTM/P1 and MTTM/P5 excluded (α₁=1 zero-solution artifacts). DNC = no converged seed at native tol.")
        println(io, "% †: median over a PARTIAL instance set; see tab:solved for solved/total counts.")
        latex_solved_table(io, df)        # robustness first — frames the medians below
        latex_metric_table(io, df, :iters, fmt_int,
            "Median iterations to the native tolerance (best in \\textbf{bold}).", "tab:iters")
        latex_metric_table(io, df, :fe, fmt_int,
            "Median operator (\$B\$) evaluations --- single-call (1/iter) vs two-call (2/iter).", "tab:evals")
        latex_metric_table(io, df, :cpu, fmt_cpu,
            "Median CPU time (seconds).", "tab:cpu")
    end
    println(tee, "  wrote $(path)")
end

# ── 2. Performance profiles ──────────────────────────────────────────────────
"Cost matrix (instances × methods) for `metric`; Inf = failure (DNC/excluded)."
function cost_matrix(df, metric::Symbol)
    insts = unique([(r.problem, r.dim, r.init) for r in eachrow(df)])
    idx   = Dict(insts[i] => i for i in eachindex(insts))
    T     = fill(Inf, length(insts), length(S70_METHOD_ORDER))
    for r in eachrow(df)
        (r.method, r.problem) in ARTIFACT_EXCLUDE && continue
        r.conv == 1 || continue
        j = findfirst(==(r.method), S70_METHOD_ORDER)
        j === nothing && continue
        T[idx[(r.problem, r.dim, r.init)], j] = float(r[metric])
    end
    return T
end

function make_profiles(df, tee; png::Bool)
    for (metric, tag, title) in ((:iters, "iters", "Iterations"),
                                 (:fe,    "evals", "Operator evaluations"),
                                 (:cpu,   "cpu",   "CPU time"))
        T = cost_matrix(df, metric)
        plt = performance_profile(PlotsBackend(), T, S70_METHOD_ORDER;
                                  logscale = true, title = title,
                                  legend = :bottomright)
        savefig(plt, joinpath(FIGDIR, "profile_$(tag).pdf"))
        png && savefig(plt, joinpath(FIGDIR, "profile_$(tag).png"))
        println(tee, "  wrote profile_$(tag).pdf")
    end
end

# ── 3. Convergence panels ────────────────────────────────────────────────────
function make_convergence_panels(db, tee, production::Bool; png::Bool)
    hdf = DBInterface.execute(db, """
        SELECT c.method AS method, h.problem AS problem, h.k AS k, h.residual AS res,
               c.eps AS eps, c.maxiter AS maxiter, h.config_hash AS config_hash
        FROM history h JOIN configs c ON h.config_hash = c.config_hash
        WHERE h.script='s30' AND h.production=?
        ORDER BY c.method, h.problem, h.k
    """, (production ? 1 : 0,)) |> DataFrame
    hdf = filter(r -> is_canonical(r.problem, r.eps, r.maxiter), hdf)   # canonical config only
    cur = current_canonical_configs(db; production = production)
    hdf = filter(r -> r.config_hash in cur, hdf)                        # pin to the live tuned config
    if nrow(hdf) == 0
        println(tee, "  (no history rows for production=$(production ? 1 : 0) — skipping panels)")
        return
    end
    plt = plot(layout = (3, 2), size = (1000, 1100))
    for (pi, P) in enumerate(S70_PROBLEMS)
        plotted = false
        for m in S70_METHOD_ORDER
            (m, P) in ARTIFACT_EXCLUDE && continue
            sub = hdf[(hdf.method .== m) .& (hdf.problem .== P), :]
            nrow(sub) == 0 && continue
            res = max.(float.(sub.res), 1e-16)        # floor for log scale
            plot!(plt, sub.k, res; subplot = pi, yscale = :log10, label = m, lw = 1.5)
            plotted = true
        end
        title!(plt, P; subplot = pi)
        xlabel!(plt, "iteration k"; subplot = pi)
        ylabel!(plt, "residual Rₙ"; subplot = pi)
        plotted || annotate!(plt, 0.5, 0.5, text("no history", 9); subplot = pi)
    end
    savefig(plt, joinpath(FIGDIR, "convergence_panels.pdf"))
    png && savefig(plt, joinpath(FIGDIR, "convergence_panels.png"))
    println(tee, "  wrote convergence_panels.pdf")
end

# ── Main ─────────────────────────────────────────────────────────────────────
function main()
    opts, flags = parse_cli(ARGS)
    production  = !("quick" in flags)              # default: production=1; --quick ⟹ 0
    png         = "png" in flags
    mkpath(FIGDIR)

    logpath, tee, _ = setup_logging("s70_figures")
    db = open_db()
    df = load_results(db, production)

    println(tee, "=" ^ 72)
    println(tee, "  s70 figures & tables — production=$(production ? 1 : 0)   $(Dates.now())")
    println(tee, "  $(nrow(df)) s30 result rows; output → $(FIGDIR)")
    println(tee, "=" ^ 72)
    if nrow(df) == 0
        println(tee, "  (no s30 production=$(production ? 1 : 0) rows — run s30 first.)")
        teardown_logging(tee, logpath)
        return nothing
    end

    # Tables first (no plotting dependency) so they always land.
    print_console_tables(df, tee)
    write_latex_tables(df, tee)

    # Figures (guarded: a Plots/BenchmarkProfiles API hiccup won't drop the tables).
    println(tee, "\n--- figures ---")
    try
        make_profiles(df, tee; png = png)
    catch e
        println(tee, "  ⚠ performance profiles failed: ", sprint(showerror, e))
    end
    try
        make_convergence_panels(db, tee, production; png = png)
    catch e
        println(tee, "  ⚠ convergence panels failed: ", sprint(showerror, e))
    end

    println(tee, "\n  Done.")
    teardown_logging(tee, logpath)
    return nothing
end

main()
