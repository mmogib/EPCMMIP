# ============================================================================
# s25: Promote best LHS configs (ALL methods) → results/tuned_params.toml
# ============================================================================
#
# Reads s20 LHS rows (script='s20') for every method and selects, per
# (method, problem), the BEST config by a ROBUST criterion, then writes the
# winners to results/tuned_params.toml and prints paste-ready preset entries.
#
# ── Robust selection (fixes the s20 "best" display trap) ───────────────────
# Ranks by  (n_conv DESC, median_iters ASC, median_cpu ASC)  — convergence-
# robustness first. A 5/5 @ 552 beats a 1/5 @ 24. If NO config reaches the
# canonical tolerance in budget, falls back to the CLOSEST config (min median
# final native residual) as a best-effort config, flagged non-converged (so
# s30 always has a config and the paper reports an honest best-effort + DNC).
#
# ── Version + tolerance safety ─────────────────────────────────────────────
# `version(T)` is part of make_config_hash, and `eps` (= native tolerance) too.
# s25 filters to each method's CURRENT version AND the problem's CANONICAL
# native tolerance, so stale (old-version) or diagnostic (--eps) rows are never
# promoted.
#
# Generic over methods: every algorithm struct is Float64/Symbol-only (project
# rule), and configs.params_json stores field→repr(value); s25 reconstructs via
# fieldnames(T) (declaration order) without per-method hardcoding.
#
# Usage:  julia --project=jcode jcode/scripts/s25_promote_best_params.jl
# Run AFTER s20 has tuned all (method, problem) cells. Cells with no current-
# version s20 rows are skipped with a note. Read-only on the DB; writes only
# results/tuned_params.toml.

include(joinpath(@__DIR__, "..", "src", "includes.jl"))

const PROMOTE_METHODS  = (EPCM, MTTM, IMTTM, IFRAB, SFRBM)   # reporting order
const PROMOTE_PROBLEMS = ("P1", "P2", "P3", "P4")

_median(v) = isempty(v) ? Inf :
    (s = sort(v); n = length(s); iseven(n) ? (s[n ÷ 2] + s[n ÷ 2 + 1]) / 2 : s[(n + 1) ÷ 2])

# Reconstruct an ordered [field => value] list from configs.params_json
# (Dict field→repr(value)). Values are Float64 or Symbol (the only field types
# permitted in algorithm structs). Order follows fieldnames(T).
function parse_params(::Type{T}, pj::AbstractString) where {T}
    d = JSON3.read(pj, Dict{String,String})
    pairs = Pair{Symbol,Any}[]
    for f in fieldnames(T)
        v = d[string(f)]
        val = startswith(v, ":") ? Symbol(replace(v, ":" => "")) : parse(Float64, v)
        push!(pairs, f => val)
    end
    return pairs
end

# Aggregate s20 rows for one (method current-version, canonical eps, problem) cell.
function ranked_configs(db, method_name::AbstractString, cur_ver::AbstractString,
                        canon_eps::Float64, problem::AbstractString)
    df = DBInterface.execute(db, """
        SELECT r.config_hash AS hash, r.dimension AS dim, r.converged AS conv,
               r.iterations AS iters, r.cpu_time AS cpu, r.native_residual AS nat,
               c.params_json AS pj
        FROM results r JOIN configs c ON r.config_hash = c.config_hash
        WHERE r.script='s20' AND c.method=? AND c.version=? AND c.eps=? AND r.problem=?
    """, (method_name, cur_ver, canon_eps, problem)) |> DataFrame
    nrow(df) == 0 && return NamedTuple[]
    recs = NamedTuple[]
    for h in unique(df.hash)
        sub       = df[df.hash .== h, :]
        conv_rows = sub[sub.conv .== 1, :]
        n_conv    = nrow(conv_rows)
        nats      = Float64[float(x) for x in sub.nat if !(x === missing) && isfinite(float(x))]
        push!(recs, (
            hash = h, dim = sub.dim[1], pj = sub.pj[1],
            n_total = nrow(sub), n_conv = n_conv,
            med_iters = n_conv > 0 ? _median(float.(conv_rows.iters)) : Inf,
            med_cpu   = n_conv > 0 ? _median(float.(conv_rows.cpu))   : Inf,
            med_nat   = isempty(nats) ? Inf : _median(nats),
        ))
    end
    sort!(recs, by = r -> (-r.n_conv, r.med_iters, r.med_cpu))   # robust rank
    return recs
end

_toml_val(v::Symbol) = "\"$(v)\""
_toml_val(v::Real)   = string(v)
_preset_val(v::Symbol) = ":$(v)"
_preset_val(v::Real)   = @sprintf("%.6g", v)

function main()
    logpath, tee, _ = setup_logging("s25_promote")
    db = open_db()

    println(tee, "=" ^ 80)
    println(tee, "  s25 promote best params — ALL methods (robust: n_conv↓, med_iters↑)  $(Dates.now())")
    println(tee, "  Filters: current version(T) + canonical native tolerance per problem.")
    println(tee, "=" ^ 80)

    # chosen[(method_name, P)] = (T, ordered-params, winner-meta-with-:converged)
    chosen = Dict{Tuple{String,String},Any}()
    for T in PROMOTE_METHODS
        mname   = name(T)
        cur_ver = string(version(T))
        println(tee, "\n############  $mname  (v$cur_ver)  ############")
        for P in PROMOTE_PROBLEMS
            canon_eps = native_tol(P)
            recs = ranked_configs(db, mname, cur_ver, canon_eps, P)
            if isempty(recs)
                println(tee, "  $P: no $mname(v$cur_ver) s20 rows at eps=$canon_eps — SKIP (run s20 --method=$mname --problem=$P).")
                continue
            end
            @printf(tee, "  --- %s/%s : top configs ---\n", mname, P)
            @printf(tee, "    %-14s %7s %10s %12s %12s\n", "hash", "n_conv", "med_iter", "med_cpu(s)", "med_nat")
            for r in recs[1:min(3, length(recs))]
                @printf(tee, "    %-14s %4d/%-2d %10s %12.4f %12.3e\n",
                        r.hash, r.n_conv, r.n_total,
                        isfinite(r.med_iters) ? string(Int(round(r.med_iters))) : "—", r.med_cpu, r.med_nat)
            end
            local w, conv_flag
            if recs[1].n_conv == 0
                w = sort(recs, by = r -> r.med_nat)[1]      # closest-residual fallback
                conv_flag = false
                println(tee, "    ⚠ NO config reached eps=$canon_eps in budget — promoting CLOSEST-residual ",
                        "(med native res=$(round(w.med_nat, sigdigits=3)), hash $(w.hash)). Benchmark will show DNC here.")
            else
                w = recs[1]
                conv_flag = true
                w.n_conv < w.n_total &&
                    println(tee, "    ⚠ best converged $(w.n_conv)/$(w.n_total) seeds (not fully robust); promoting most-robust.")
            end
            chosen[(mname, P)] = (T, parse_params(T, w.pj), merge(w, (converged = conv_flag,)))
        end
    end

    # ── write results/tuned_params.toml ──────────────────────────────────────
    tomlpath = joinpath(JCODE_ROOT, "results", "tuned_params.toml")
    open(tomlpath, "w") do io
        println(io, "# tuned_params.toml — best LHS configs per (method, problem).")
        println(io, "# s25 robust selection (n_conv↓, med_iters↑; closest-residual fallback when DNC).")
        println(io, "# Generated $(Dates.now()) from script='s20' rows at canonical native tolerances.")
        for T in PROMOTE_METHODS
            mname = name(T)
            for P in PROMOTE_PROBLEMS
                haskey(chosen, (mname, P)) || continue
                _, params, w = chosen[(mname, P)]
                println(io, "\n[$mname.$P]")
                for (f, val) in params
                    println(io, "$f = $(_toml_val(val))")
                end
                println(io, "# meta: converged=$(w.converged)  n_conv=$(w.n_conv)/$(w.n_total)  " *
                            "med_iters=$(isfinite(w.med_iters) ? Int(round(w.med_iters)) : -1)  " *
                            "med_native_res=$(isfinite(w.med_nat) ? w.med_nat : -1)  dim=$(w.dim)  hash=$(w.hash)")
            end
        end
    end
    println(tee, "\n  Wrote $(tomlpath)  ($(length(chosen)) (method,problem) cell(s)).")

    # ── write results/tuned_params.jl (machine-readable; s30 `include`s it; no TOML dep) ──
    jlpath = joinpath(JCODE_ROOT, "results", "tuned_params.jl")
    open(jlpath, "w") do io
        println(io, "# tuned_params.jl — generated by s25 ($(Dates.now())). Do not edit by hand (re-run s25).")
        println(io, "# (method_name, problem) → NamedTuple of tuned params; TUNED_CONVERGED flags DNC-at-native-tol cells.")
        println(io, "const TUNED_PARAMS = Dict{Tuple{String,String},NamedTuple}(")
        for T in PROMOTE_METHODS, P in PROMOTE_PROBLEMS
            haskey(chosen, (name(T), P)) || continue
            _, params, _ = chosen[(name(T), P)]
            body = join(["$f = $(repr(val))" for (f, val) in params], ", ")
            println(io, "    (\"$(name(T))\", \"$P\") => ($body,),")
        end
        println(io, ")")
        println(io, "const TUNED_CONVERGED = Dict{Tuple{String,String},Bool}(")
        for T in PROMOTE_METHODS, P in PROMOTE_PROBLEMS
            haskey(chosen, (name(T), P)) || continue
            _, _, w = chosen[(name(T), P)]
            println(io, "    (\"$(name(T))\", \"$P\") => $(w.converged),")
        end
        println(io, ")")
    end
    println(tee, "  Wrote $(jlpath)  (machine-readable; s30 includes it).")

    # ── paste-ready preset entries (per method) ──────────────────────────────
    for T in PROMOTE_METHODS
        mname = name(T)
        cells = [(P, chosen[(mname, P)]) for P in PROMOTE_PROBLEMS if haskey(chosen, (mname, P))]
        isempty(cells) && continue
        println(tee, "\n--- paste-ready $(mname)_PRESETS entries ---")
        for (P, (_, params, w)) in cells
            body = join(["$f=$(_preset_val(val))" for (f, val) in params], ", ")
            @printf(tee, "    :%s => (%s),%s\n", P, body, w.converged ? "" : "   # DNC at native tol — best-effort")
        end
    end

    println(tee, "\n  NOTE: run s25 only after s20 has tuned all (method,problem) cells; re-run to refresh.")
    teardown_logging(tee, logpath)
    return chosen
end

main()
