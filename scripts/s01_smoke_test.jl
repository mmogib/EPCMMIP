# ============================================================================
# s01: Smoke Test — verify all current smoke-listed solvers run end-to-end
# ============================================================================
#
# Sanity check that exercises every layer of the stack across the full
# current smoke-listed solver × 5-problem matrix at the smallest dimension of each problem
# (P1 N=32, P2 m=100, P3 K=100, P4 N=64, P5 N=64), seed1, maxiter = 100:
#
#   Section 1 : Build the five problems (smallest dim each); report name/dim/L.
#   Section 2 : solve() matrix — current smoke-listed solvers on P1–P5.
#               Each solver uses its per-problem preset :P{k}.
#               Reports converged?, flag, iters, B-evals, both residuals, native.
#               Asserts each cell returns a SolverResult with a valid flag
#               (it does NOT assert convergence — see note below). The P2
#               soft-thresholding resolvent exercises the ρ-dependent resolvent
#               path that P1/P3 (clip, ρ-independent) do not.
#   Section 3 : Config-hash uniqueness — all (method × problem) configs
#               must hash distinctly; plus within-EPCM preset distinctness and
#               prob_id-participates-in-hash micro-checks.
#   Section 4 : DB integration — ensure_config!/insert_result! for all 25 cells,
#               insert_history! once (method-agnostic IterRecord path), and
#               is_done + script-discriminator checks. Rows tagged script="s01"
#               so smoke data stays out of the s30 benchmark namespace.
#
# Usage:  julia --project=jcode jcode/scripts/s01_smoke_test.jl
# Output: console + jcode/results/logs/log_smoke_test_<timestamp>.txt
#
# Convergence is intentionally NOT asserted: small dim × tight tolerance ×
# maxiter=100 may not converge, and some baselines on off-home problems may
# diverge (caught gracefully by NanStopping → flag=:nan). That is diagnostic
# information, not a failure. The script asserts that every solve(...) returns
# a SolverResult without crashing, that hashes are unique, and that DB writes
# succeed.

include(joinpath(@__DIR__, "..", "src", "includes.jl"))

# Algorithm types in canonical reporting order. Each exposes :P1/:P2/:P3/:P4/:P5 presets
# (EPCM has no :paper; the baseline competitors do, but the smoke uses the per-problem
# presets so every cell is a valid (method, problem) configuration).
const SMOKE_ALG_TYPES = (EPCM, MAEFBFP, MTTM, IMTTM, SFRBM, IFRAB, VAFBS, MDITSM, MFRBSM, RFBSM, IRFBSM)

# Stopping uses each problem's NATIVE source-paper criterion (D1) via
# NativeResStopping(prob.native_residual, native_tol(label)) — native_tol from
# benchmark.jl. maxiter is kept small (smoke = quick sanity, not convergence).
const SMOKE_MAXITER = 100


function main()
    logpath, tee, _ = setup_logging("smoke_test")

    println(tee, "=" ^ 75)
    println(tee, "  EPCMMIP Smoke Test — $(length(SMOKE_ALG_TYPES)) solvers × 5 problems  $(Dates.now())")
    println(tee, "=" ^ 75)

    all_ok = true

    # ══════════════════════════════════════════════════════════════════════
# Section 1: Build the five problems (smallest dim each)
    # ══════════════════════════════════════════════════════════════════════
    println(tee, "\n--- Section 1: Problem setup (smallest dim per problem) ---")
    problems = NamedTuple[]
    try
        push!(problems, (id = 1, label = "P1", prob = get_problem(1; dim = 32,  n_seeds = 3)))
        push!(problems, (id = 2, label = "P2", prob = get_problem(2; dim = 100, n_seeds = 3)))
        push!(problems, (id = 3, label = "P3", prob = get_problem(3;            n_seeds = 3)))
        push!(problems, (id = 4, label = "P4", prob = get_problem(4; dim = 64,  n_seeds = 3)))
        push!(problems, (id = 5, label = "P5", prob = get_problem(5; dim = 64,  n_seeds = 3)))
        for p in problems
            @printf(tee, "  %-3s %-22s dim=%-4d L=%.4f  init_pts=%d\n",
                    p.label, p.prob.name, p.prob.dim, p.prob.metadata.L,
                    length(p.prob.initial_points))
        end
    catch e
        all_ok = false
        println(tee, "  PROBLEM SETUP FAILED:")
        showerror(tee, e, catch_backtrace())
    end

    # ══════════════════════════════════════════════════════════════════════
    # Section 2: solve() matrix — all registered solvers × 5 problems
    # ══════════════════════════════════════════════════════════════════════
    println(tee, "\n--- Section 2: solve() matrix — $(length(SMOKE_ALG_TYPES)) solvers × 5 problems ---")
    cells = NamedTuple[]
    n_expected_cells = length(SMOKE_ALG_TYPES) * length(problems)
    if !isempty(problems)
        @printf(tee, "  %-6s %-4s %-5s %-6s %-10s %6s %8s %12s %12s %12s\n",
                "method", "prob", "dim", "conv", "flag", "iters", "f_evals",
                "R_n", "Rt_n", "native")
        println(tee, "  " * "-" ^ 92)
        for p in problems
            preset = Symbol("P$(p.id)")
            init   = first(get_initial_points(p.prob))
            for T in SMOKE_ALG_TYPES
                cell_ok = true
                try
                    alg  = T(preset)
                    hist = HistoryCallback()
                    nrec = NativeResRecorder(p.prob.native_residual)
                    # Primary stopper = native source-paper criterion (D1).
                    stopping = (NativeResStopping(p.prob.native_residual, native_tol(p.label)),
                                MaxIterStopping(SMOKE_MAXITER),
                                NanStopping())

                    result = solve(alg, p.prob, copy(init.x0);
                                   stopping = stopping, observers = (hist, nrec))

                    valid_flag = result.flag in
                        (:converged, :maxiter, :nan, :stagnation, :error)
                    if !(result isa SolverResult) || !valid_flag ||
                       result.iterations < 1 || result.f_evals < 1
                        cell_ok = false
                    end

                    @printf(tee,
                        "  %-6s %-4s %-5d %-6s %-10s %6d %8d %12.4e %12.4e %12.4e\n",
                        name(T), p.label, p.prob.dim, string(result.converged),
                        string(result.flag), result.iterations, result.f_evals,
                        result.residual, result.scaled_residual, nrec.value)

                    push!(cells, (method = name(T), problem = p.label, dim = p.prob.dim,
                                  alg = alg, result = result, hist = hist,
                                  nrec = nrec, init = init))
                catch e
                    cell_ok = false
                    @printf(tee, "  %-6s %-4s %-5d  *** EXCEPTION ***\n",
                            name(T), p.label, p.prob.dim)
                    showerror(tee, e, catch_backtrace())
                    println(tee)
                end
                all_ok &= cell_ok
            end
        end
        @printf(tee, "\n  %d/%d cells returned a valid SolverResult.\n",
                length(cells), n_expected_cells)
        println(tee, "  Note: convergence is NOT asserted (smoke maxiter=100; stop = native criterion).")
        println(tee, "        native: P1 E_n=‖(I−P_C)x‖²+‖Bx‖² · P2/P4 ‖x−x_prev‖ · P3 0.5‖z−clip(z−Bz)‖² · P5 ‖x−J^A_1(x−Bx)‖.")
        println(tee, "        (P1=Volterra SFP, P2=ℓ1+quad, P3=control, P4=LASSO, P5=ℓ₂ positive-part example.)")
    end

    # ══════════════════════════════════════════════════════════════════════
    # Section 3: Config-hash uniqueness (methods × problems must be distinct)
    # ══════════════════════════════════════════════════════════════════════
    println(tee, "\n--- Section 3: Config-hash sanity ---")
    try
        # (a) All (method × problem) configs must hash distinctly. The
        # convergence tolerance fed to the hash is the problem's native_tol (D1).
        cell_hashes = Tuple{String,String,String}[]   # (method, problem, hash)
        for p in problems
            preset = Symbol("P$(p.id)")
            for T in SMOKE_ALG_TYPES
                h, _ = make_config_hash(T(preset), p.label, native_tol(p.label), SMOKE_MAXITER)
                push!(cell_hashes, (name(T), p.label, h))
            end
        end
        n_distinct = length(unique(getindex.(cell_hashes, 3)))
        @printf(tee, "  method×problem hashes: %d configs, %d distinct\n",
                length(cell_hashes), n_distinct)
        if n_distinct == length(cell_hashes) && !isempty(cell_hashes)
            println(tee, "  Uniqueness across methods × problems: PASS")
        else
            all_ok = false
            println(tee, "  Uniqueness across methods × problems: FAIL (hash collision)")
            for (m, pr, h) in cell_hashes
                @printf(tee, "    %-6s %-3s -> %s\n", m, pr, h)
            end
        end

        # (b) EPCM :P1/:P2/:P3/:P4/:P5 must be distinct (per-problem τ_0 may differ).
        epcm_h = [make_config_hash(EPCM(pp), string(pp), native_tol(string(pp)), SMOKE_MAXITER)[1]
                  for pp in (:P1, :P2, :P3, :P4, :P5)]
        if length(unique(epcm_h)) == 5
            println(tee, "  EPCM :P1/:P2/:P3/:P4/:P5 distinct: PASS")
        else
            all_ok = false
            println(tee, "  EPCM :P1/:P2/:P3/:P4/:P5 distinct: FAIL")
        end

        # (c) prob_id participates in the hash (same alg+params+eps, different prob).
        h1, _ = make_config_hash(EPCM(:P1), "P1", 1.0e-6, SMOKE_MAXITER)
        h2, _ = make_config_hash(EPCM(:P1), "P2", 1.0e-6, SMOKE_MAXITER)
        if h1 != h2
            println(tee, "  prob_id-in-hash: PASS (EPCM(:P1) hashes differ for prob=P1 vs P2)")
        else
            all_ok = false
            println(tee, "  prob_id-in-hash: FAIL (hash ignores prob_id)")
        end
    catch e
        all_ok = false
        println(tee, "  HASH-SANITY FAILED:")
        showerror(tee, e, catch_backtrace())
    end

    # ══════════════════════════════════════════════════════════════════════
    # Section 4: DB integration (all cells; rows tagged script="s01")
    # ══════════════════════════════════════════════════════════════════════
    println(tee, "\n--- Section 4: DB integration (script=\"s01\") ---")
    if !isempty(cells)
        try
            db     = open_db()   # main results/experiments.db
            run_id = "smoke_$(Dates.format(now(), "yyyymmdd_HHMMSS"))"

            n_inserted = 0
            for c in cells
                eps_c  = native_tol(c.problem)
                h, hin = make_config_hash(c.alg, c.problem, eps_c, SMOKE_MAXITER)
                ensure_config!(db, c.alg, c.problem, eps_c, SMOKE_MAXITER, h, hin)
                insert_result!(db, h, c.problem, c.dim, c.init.label, c.init.seed_idx,
                               run_id, c.result;
                               script = "s01", native_residual = c.nrec.value)
                n_inserted += 1
            end
            @printf(tee, "  Configs + results inserted: %d/%d cells (run_id=%s)\n",
                    n_inserted, n_expected_cells, run_id)

            # History-insert path is method-agnostic (IterRecord); exercise once.
            rep    = first(cells)
            h_rep, _ = make_config_hash(rep.alg, rep.problem, native_tol(rep.problem), SMOKE_MAXITER)
            insert_history!(db, h_rep, rep.problem, rep.dim, rep.init.label,
                            rep.init.seed_idx, rep.hist.history; script = "s01")
            @printf(tee, "  History inserted: %d rows for %s/%s (one transaction)\n",
                    length(rep.hist.history), rep.method, rep.problem)

            # is_done check on the representative cell.
            if is_done(db, h_rep, rep.problem, rep.dim, rep.init.label; script = "s01")
                println(tee, "  is_done (script=s01): PASS")
            else
                all_ok = false
                println(tee, "  is_done (script=s01): FAIL (just-inserted row not found)")
            end

            # Script discriminator: the s01 row must NOT satisfy an s30 query.
            if !is_done(db, h_rep, rep.problem, rep.dim, rep.init.label; script = "s30")
                println(tee, "  script-discriminator: PASS (s01 row absent from s30 namespace)")
            else
                all_ok = false
                println(tee, "  script-discriminator: FAIL (s01 row leaked into s30 namespace)")
            end
        catch e
            all_ok = false
            println(tee, "  DB INTEGRATION FAILED:")
            showerror(tee, e, catch_backtrace())
        end
    else
        println(tee, "  Skipped (no cells available from Section 2).")
    end

    # ══════════════════════════════════════════════════════════════════════
    # Final
    # ══════════════════════════════════════════════════════════════════════
    println(tee, "\n" * "=" ^ 75)
    println(tee, all_ok ? "  ALL SMOKE CHECKS PASSED" : "  SOME SMOKE CHECKS FAILED")
    println(tee, "=" ^ 75)

    teardown_logging(tee, logpath)
    return all_ok
end

main()
