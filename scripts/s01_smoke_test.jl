# ============================================================================
# s01: Smoke Test — verify all solvers run on a simple problem
# ============================================================================
#
# Quick sanity check before long experiments. Tests:
#   Part 1: All solvers converge on a simple problem
#   Part 2: Config hash uniqueness (no collisions across solvers/params)
#
# Usage:  julia --project=jcode jcode/scripts/s01_smoke_test.jl
#
# Output: console + jcode/results/logs/smoke_test_<timestamp>.log
# ============================================================================

# Style B (Flat Include) load pattern
include(joinpath(@__DIR__, "..", "src", "includes.jl"))

function main()
    # ── Logging ──────────────────────────────────────────────────────────
    logpath, tee, logfile = setup_logging("smoke_test")

    println(tee, "=" ^ 70)
    println(tee, "  EPCMMIP Smoke Test — $(Dates.now())")
    println(tee, "=" ^ 70)

    all_pass = true

    # ══════════════════════════════════════════════════════════════════════
    # PART 1: Solver convergence
    # ══════════════════════════════════════════════════════════════════════

    println(tee, "\n--- Part 1: Solver Convergence ---")

    # Setup: pick one easy problem at small dimension once problems.jl exists.
    # Suggested smoke-test target: Tan2022a Example 5.1 at m = 5.
    # prob = get_problem(:tan_vip; m=5, seed=42)
    # x0 = fill(0.0, prob.dim)
    eps_test = 1e-9
    maxiter_test = 1000

    # Solvers to test: (name, function, version, defaults)
    # ── Fill in once algorithms are implemented in src/algorithm.jl ──────
    solvers = [
        # ("EPCM",   solve_epcm,   EPCM_VERSION,   EPCM_DEFAULTS),
        # ("MTTM",   solve_mttm,   MTTM_VERSION,   MTTM_DEFAULTS),
        # ("IMTTM",  solve_imttm,  IMTTM_VERSION,  IMTTM_DEFAULTS),
        # ("IFRAB",  solve_ifrab,  IFRAB_VERSION,  IFRAB_DEFAULTS),
        # ("SFRBM",  solve_sfrbm,  SFRBM_VERSION,  SFRBM_DEFAULTS),
    ]

    println(tee)
    @printf(tee, "  %-12s  %5s  %5s  %6s  %12s  %8s  %s\n",
            "Algorithm", "Conv", "Iter", "FEval", "Residual", "Time(s)", "Flag")
    println(tee, "  " * "-" ^ 65)

    for (name, solver_fn, version, defaults) in solvers
        try
            # Inclusion form: solver receives (resolvent_A, B, x0)
            # result = solver_fn(prob.resolvent_A, prob.B, copy(x0);
            #                    defaults..., eps=eps_test, maxiter=maxiter_test)
            error("solver wiring not implemented yet — see src/algorithm.jl")
        catch e
            all_pass = false
            msg = sprint(showerror, e, catch_backtrace())
            @printf(tee, "  %-12s  ERROR: %s\n", name, first(msg, 120))
        end
    end

    println(tee, "  " * "-" ^ 65)
    println(tee, isempty(solvers) ?
            "  NO SOLVERS CONFIGURED — add entries to the `solvers` list once src/algorithm.jl exists" :
            (all_pass ? "  ALL SOLVERS PASSED" : "  SOME SOLVERS FAILED"))

    # ══════════════════════════════════════════════════════════════════════
    # PART 2: Config hash uniqueness
    # ══════════════════════════════════════════════════════════════════════

    if !isempty(solvers)
        println(tee, "\n--- Part 2: Config Hash Uniqueness ---")

        # Cross-solver: all hashes must be distinct
        hashes = Dict{String,String}()
        hash_ok = true
        for (name, _, version, defaults) in solvers
            h, _ = make_config_hash(name, version, defaults, eps_test, maxiter_test)
            println(tee, "  $name v$version → $h")
            if haskey(hashes, h)
                println(tee, "  COLLISION: $name collides with $(hashes[h])!")
                hash_ok = false
                all_pass = false
            end
            hashes[h] = name
        end
        println(tee, "  Cross-solver: $(hash_ok ? "PASS ($(length(hashes)) distinct)" : "FAIL")")

        # Parameter sensitivity: perturbing one param must change the hash
        name, _, version, defaults = solvers[1]
        base_h, _ = make_config_hash(name, version, defaults, eps_test, maxiter_test)
        println(tee, "\n  Parameter sensitivity ($name):")
        param_ok = true
        for (k, v) in pairs(defaults)
            v isa Number || continue   # skip non-numeric params (functions, symbols)
            perturbed = merge(defaults, NamedTuple{(k,)}((v * 1.01,)))
            h2, _ = make_config_hash(name, version, perturbed, eps_test, maxiter_test)
            if h2 == base_h
                println(tee, "    $k: FAIL — hash unchanged!")
                param_ok = false
                all_pass = false
            end
        end
        println(tee, "  Param sensitivity: $(param_ok ? "PASS" : "FAIL")")

        # Version sensitivity
        h_v2, _ = make_config_hash(name, "99.99.99", defaults, eps_test, maxiter_test)
        ver_ok = h_v2 != base_h
        println(tee, "  Version sensitivity: $(ver_ok ? "PASS" : "FAIL")")
        if !ver_ok; all_pass = false; end
    end

    # ══════════════════════════════════════════════════════════════════════
    # FINAL
    # ══════════════════════════════════════════════════════════════════════

    println(tee, "\n" * "=" ^ 70)
    println(tee, all_pass ? "  ALL CHECKS PASSED" : "  SOME CHECKS FAILED")
    println(tee, "=" ^ 70)

    teardown_logging(tee, logpath)
    return all_pass
end

main()
