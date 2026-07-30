# ============================================================================
# s28: Validate baselines at their SOURCE-PAPER settings (:paper presets)
# ============================================================================
#
# Sanity gate before the s30 benchmark. Each baseline is run at its `:paper`
# preset (the authors' recommended settings) on the suite. Our suite problems
# ARE these methods' source-paper examples, so each method has a HOME problem
# where it must behave sanely:
#
#   IMTTM (Tan2022a Alg 3.3)      home: P1 (Ex 5.2 Volterra SFP), P4 (Ex 5.3 LASSO)
#   SFRBM (Yao2024 Alg 2)         home: P2 (Ex 4.2 ℓ1+quadratic)
#   IFRAB (Izuchukwu2023 Alg 4.5) home: P3 (Ex 6.2 optimal control)
#   MTTM  (Gibali2018 Alg 1)      home: — (Gibali's example is NOT in the suite)
#
# Interpretation:
#   - A method that CONVERGES (or at least makes monotone progress, no blow-up)
#     at :paper on its HOME problem ⟹ implementation is sound; its failure on
#     OFF-home problems is then a genuine method-vs-problem result (or an LHS
#     range issue in s20), NOT a bug.
#   - A method that DIVERGES / stalls at :paper on its OWN home ⟹ implementation
#     bug — fix before trusting any s20/s30 numbers for it.
#
# This does NOT claim exact reproduction of the papers' reported iteration
# counts (our dims/initials/tolerance differ); it validates SANE behavior at the
# authors' settings on their own problem class. Compare the home-cell numbers to
# the source papers by eye.
#
# Usage:  julia --project=jcode jcode/scripts/s28_validate_baselines.jl
#         julia --project=jcode jcode/scripts/s28_validate_baselines.jl --seeds=5 --force
# Output: console + results/logs/log_s28_validate_<ts>.txt ; rows tagged script='s28'.

include(joinpath(@__DIR__, "..", "src", "includes.jl"))
using Statistics

const S28_BASELINES = (MTTM, IMTTM, IFRAB, SFRBM, VAFBS, MDITSM, RFBSM, IRFBSM)  # EPCM has no :paper preset
const S28_PROBLEMS  = ("P1", "P2", "P3", "P4")
const S28_REFDIM    = Dict("P1" => 128, "P2" => 300, "P3" => 100, "P4" => 256)
const S28_HOME      = Dict(                               # problems that are each method's source-paper home
    "MTTM"  => String[],                  # Gibali2018 example not in suite
    "IMTTM" => ["P1", "P4"],              # Tan2022a Ex 5.2 / 5.3
    "IFRAB" => ["P3"],                    # Izuchukwu2023 Ex 6.2
    "SFRBM" => ["P2"],                    # Yao2024 Ex 4.2
    "VAFBS" => ["P4"],                    # Thong2019 compressed-sensing / LASSO model
    "MDITSM" => ["P4"],                   # Wang2023 compressed-sensing / LASSO class match
    "RFBSM" => ["P4"],                    # Cholamjiak2021 signal-recovery / LASSO class match
    "IRFBSM" => ["P4"],                   # Cholamjiak2021 signal-recovery / LASSO class match
)

function parse_cli(args)
    opts = Dict{String,String}(); flags = String[]
    for a in args
        startswith(a, "--") || continue
        kv = split(a[3:end], "="; limit = 2)
        length(kv) == 2 ? (opts[kv[1]] = kv[2]) : push!(flags, kv[1])
    end
    return opts, flags
end

function main()
    opts, flags = parse_cli(ARGS)
    force   = "force" in flags
    n_seeds = haskey(opts, "seeds") ? parse(Int, opts["seeds"]) : 3

    logpath, tee, _ = setup_logging("s28_validate")
    db     = open_db()
    run_id = "s28_$(Dates.format(now(), "yyyymmdd_HHMMSS"))"

    println(tee, "=" ^ 82)
    println(tee, "  s28 baseline validation @ :paper presets   $(Dates.now())   seeds=$n_seeds")
    println(tee, "  ★ = method's source-paper HOME problem (must behave sanely there).")
    println(tee, "=" ^ 82)
    @printf(tee, "  %-6s %-4s %-3s %8s %10s %14s %-10s\n",
            "method", "prob", "★", "conv", "med_iter", "med_native", "flag")
    println(tee, "  " * "-" ^ 64)

    for T in S28_BASELINES
        mname = name(T)
        home  = get(S28_HOME, mname, String[])
        for P in S28_PROBLEMS
            prob_id = parse(Int, P[2:end])
            eps     = native_tol(P)
            maxiter = universal_maxiter(P, eps)
            prob = P == "P3" ? get_problem(3; n_seeds = n_seeds) :
                               get_problem(prob_id; dim = S28_REFDIM[P], n_seeds = n_seeds)
            rdim = prob.dim
            alg  = T(:paper)
            hash, hin = make_config_hash(alg, P, eps, maxiter)
            ensure_config!(db, alg, P, eps, maxiter, hash, hin)

            iters_conv = Float64[]; nats = Float64[]; flags = Symbol[]
            for init in get_initial_points(prob)
                if !force && is_done(db, hash, P, rdim, init.label; script = "s28")
                    continue
                end
                nrec = NativeResRecorder(prob.native_residual)
                stopping = (NativeResStopping(prob.native_residual, eps),
                            MaxIterStopping(maxiter), NanStopping())
                result = solve(alg, prob, copy(init.x0); stopping = stopping, observers = (nrec,))
                insert_result!(db, hash, P, rdim, init.label, init.seed_idx, run_id, result;
                               script = "s28", native_residual = nrec.value)
                result.converged && push!(iters_conv, float(result.iterations))
                isfinite(nrec.value) && push!(nats, nrec.value)
                push!(flags, result.flag)
            end
            # if everything was skipped (already done), pull a quick count from DB
            df = DBInterface.execute(db, """
                SELECT converged, iterations, native_residual, flag FROM results
                WHERE script='s28' AND config_hash=? AND problem=? AND dimension=?
            """, (hash, P, rdim)) |> DataFrame
            n_total = nrow(df)
            n_conv  = sum(df.converged .== 1)
            mi = n_conv > 0 ? string(Int(round(median(float.(df[df.converged .== 1, :].iterations))))) : "—"
            natvals = Float64[float(x) for x in df.native_residual if !(x === missing) && isfinite(float(x))]
            mn = isempty(natvals) ? NaN : median(natvals)
            star = P in home ? "★" : ""
            flagstr = isempty(flags) ? "(cached)" : string(flags[1])
            @printf(tee, "  %-6s %-4s %-3s %4d/%-3d %10s %14.3e %-10s\n",
                    mname, P, star, n_conv, n_total, mi, mn, flagstr)
        end
        # home-cell verdict line
        if !isempty(home)
            homestr = join(home, ",")
            srcname = mname == "IMTTM" ? "Tan2022a" : mname == "IFRAB" ? "Izuchukwu2023" :
                      mname == "SFRBM" ? "Yao2024" : mname == "VAFBS" ? "Thong2019" :
                      mname == "MDITSM" ? "Wang2023" :
                      (mname == "RFBSM" || mname == "IRFBSM") ? "Cholamjiak2021" : "source"
            println(tee, "    → $mname home = $homestr: the ★ row(s) must converge / stay stable (compare to $srcname).")
        else
            println(tee, "    → $mname home (Gibali2018) is NOT in the suite — rows are off-home behavior only (no source-cell validation).")
        end
    end

    println(tee, "\n  Interpretation: a ★ row that DIVERGES/stalls ⟹ implementation bug (fix before trusting s20/s30).")
    println(tee, "  A ★ row that converges ⟹ implementation sound; off-home failures are genuine (or s20-range issues).")
    println(tee, "  Rows tagged script='s28'.")
    teardown_logging(tee, logpath)
    return nothing
end

main()
