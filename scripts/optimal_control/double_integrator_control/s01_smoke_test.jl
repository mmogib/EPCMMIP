# ============================================================================
# s01: Smoke Test — verify the double-integrator smoke methods run end-to-end
# ============================================================================
#
# Small sanity check for the double-integrator optimal-control problem.
# This is not the full benchmark.
#
# Smoke configuration:
#   - problem      : local double-integrator optimal control
#   - mesh size    : K = 100
#   - initial data : seed1 only
#   - methods      : AEFBFP, IFRAB
#   - presets      : both methods temporarily use the legacy control preset :P3
#                    only as a parameter seed; the problem itself is local here
#   - stopping     : native residual <= 1e-5 for 2 consecutive iterations,
#                    or maxiter = 4000, or NaN detection
#
# What this script does:
#   Before Section 1 :
#       load the shared source files from src/includes.jl, then open the smoke
#       log and the local optimal-control DB.
#   Section 1 : Build the smoke problem (K=100) and print its basic data.
#   Section 2 : Define the stopping rule explicitly.
#   Section 3 : Hash sanity — verify the smoke configs get distinct
#               config_hash values before we write any DB rows.
#   Section 4 : Run the smoke methods and check that solve(...) returns a
#               usable SolverResult without crashing.
#   Section 5 : DB integration — write smoke rows under script="s01" and check
#               that they do not leak into the s30 benchmark namespace.
#   Final     : Print one clear PASS/FAIL summary line.
#
# What this script does NOT do:
#   - no parameter search
#   - no tuned-winner import
#   - no multi-dimension benchmark sweep
#   - no convergence plots
#   - no figures or tables

include(joinpath(@__DIR__, "..", "..", "..", "src", "includes.jl"))
include(joinpath(@__DIR__, "problem_definition.jl"))

const SMOKE_METHOD_TYPES = (AEFBFP, IFRAB)
const SMOKE_MAXITER = 4000

function main()
    # ========================================================================
    # Before Section 1: setup
    # ========================================================================
    #
    # At this point the file has already loaded the shared project source via
    # src/includes.jl. We now do the run-level setup:
    #   - create the smoke log directory
    #   - open the smoke log
    #   - open/create the local DB
    #   - create a run id
    #   - initialize the overall PASS/FAIL state
    problem_name = DOUBLE_INTEGRATOR_SPEC.problem_name
    result_root  = joinpath(JCODE_ROOT, "results", "optimal_control", problem_name)
    logdir       = joinpath(result_root, "logs")
    db_path      = joinpath(result_root, "experiments.db")

    mkpath(logdir)
    logpath, tee, _ = setup_logging("s01_smoke_test"; logdir = logdir)
    db = open_db(db_path)

    run_id = "s01_" * Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
    all_ok = true
    rows = NamedTuple[]

    try
        dim = 100
        eps = 1.0e-5
        consec = 2
        preset = :P3

        println(tee, "="^78)
        println(tee, "  smoke test: $(problem_name)")
        println(tee, "="^78)
        println(tee, "  methods       : $(join(name.(SMOKE_METHOD_TYPES), ", "))")
        println(tee, "  preset        : $(preset)")
        println(tee, "  mesh size K   : $(dim)")
        println(tee, "  initial points: 1 (seed1 only)")
        @printf(tee, "  eps           : %.1e\n", eps)
        println(tee, "  maxiter       : $(SMOKE_MAXITER)")
        println(tee, "  log directory : $(logdir)")
        println(tee, "  local DB path : $(db_path)")

        # ====================================================================
        # Section 1: Build smoke problem
        # ====================================================================
        println(tee, "\n--- Section 1: Build smoke problem ---")
        # Build the local double-integrator optimal-control problem at K = 100.
        prob = build_double_integrator_problem(dim; n_inits = 1)
        # Use only the first seeded initial point for this smoke run.
        init = first(get_initial_points(prob))
        @printf(tee, "  problem name  : %s\n", prob.name)
        @printf(tee, "  control dim   : %d\n", prob.dim)
        @printf(tee, "  step size h   : %.4f\n", prob.metadata.h)
        @printf(tee, "  Lipschitz L   : %.4f\n", prob.metadata.L)
        @printf(tee, "  init label    : %s\n", init.label)
        println(tee, "  input vector  : z in R^K")

        # ====================================================================
        # Section 2: Define stopping rule explicitly
        # ====================================================================
        println(tee, "\n--- Section 2: Stopping rule ---")
        println(tee, "  Stopping tuple used by solve(...):")
        println(tee, "    (NativeResStopping(prob.native_residual, $(eps); consec=$(consec)),")
        println(tee, "     MaxIterStopping($(SMOKE_MAXITER)),")
        println(tee, "     NanStopping())")
        println(tee, "  Primary stop : native residual <= $(eps) for $(consec) consecutive iterations")
        println(tee, "  Native residual for this problem:")
        println(tee, "    0.5 * ||z - clip(z - B(z))||^2")
        println(tee, "  Backup stop  : stop after $(SMOKE_MAXITER) iterations if tolerance is not hit")
        println(tee, "  Safety stop  : stop if a non-finite value appears")

        # ====================================================================
        # Section 3: Hash sanity
        # ====================================================================
        #
        # Use:
        #   The DB stores experiment rows by config_hash. If two different
        #   configs get the same hash, results can mix or overwrite each other.
        println(tee, "\n--- Section 3: Hash sanity ---")
        method_specs = NamedTuple[]
        hashes = String[]
        for T in SMOKE_METHOD_TYPES
            # Build the fixed smoke config for this method from preset :P3.
            alg = T(preset)
            # Hash = DB identity for (method, params, problem, eps, maxiter).
            hash, hash_input = make_config_hash(alg, problem_name, eps, SMOKE_MAXITER)
            push!(method_specs, (method = name(T), alg = alg, hash = hash, hash_input = hash_input))
            push!(hashes, hash)
            println(tee, "  $(name(T)): hash=$(hash)")
        end

        # Different methods should not collapse to the same hash.
        if length(unique(hashes)) == length(hashes)
            println(tee, "  method-wise hash uniqueness: PASS")
        else
            all_ok = false
            println(tee, "  method-wise hash uniqueness: FAIL")
        end

        # The problem id should also affect the hash for the same algorithm.
        ref_alg = first(method_specs).alg
        h1, _ = make_config_hash(ref_alg, problem_name, eps, SMOKE_MAXITER)
        h2, _ = make_config_hash(ref_alg, "other_problem", eps, SMOKE_MAXITER)
        if h1 != h2
            println(tee, "  problem-id participates in hash: PASS")
        else
            all_ok = false
            println(tee, "  problem-id participates in hash: FAIL")
        end

        # ====================================================================
        # Section 4: Run smoke methods
        # ====================================================================
        println(tee, "\n--- Section 4: Run smoke methods ---")
        @printf(tee, "  %-8s %-6s %-10s %6s %8s %12s %12s %12s\n",
                "method", "conv", "flag", "iters", "f_evals", "R_n", "Rt_n", "native")
        println(tee, "  " * "-"^90)

        # Build the stopping callbacks once for this single smoke problem.
        stopping = (
            NativeResStopping(prob.native_residual, eps; consec = consec),
            MaxIterStopping(SMOKE_MAXITER),
            NanStopping(),
        )

        for spec in method_specs
            # Track whether this individual method passed the smoke checks.
            cell_ok = true
            try
                # Register the config in the DB before writing any result row.
                ensure_config!(db, spec.alg, problem_name, eps, SMOKE_MAXITER, spec.hash, spec.hash_input)

                # Record the native residual while the solver runs.
                nrec = NativeResRecorder(prob.native_residual)
                # Run the solver on the smoke initial point.
                result = solve(spec.alg, prob, copy(init.x0); stopping = stopping, observers = (nrec,))
                native_residual = nrec.value

                # Smoke requirement: return a usable SolverResult; convergence is not required.
                valid_flag = result.flag in (:converged, :maxiter, :nan, :stagnation, :error)
                cell_ok = (result isa SolverResult) && valid_flag && result.iterations >= 1 && result.f_evals >= 1

                # Write the smoke result row under the s01 namespace only.
                insert_result!(db, spec.hash, problem_name, prob.dim, init.label, init.seed_idx, run_id, result;
                               script = "s01", native_residual = native_residual, production = false)

                @printf(tee,
                        "  %-8s %-6s %-10s %6d %8d %12.4e %12.4e %12.4e\n",
                        spec.method, string(result.converged), string(result.flag),
                        result.iterations, result.f_evals, result.residual,
                        result.scaled_residual, native_residual)

                # Save method/hash pairs for the DB verification section.
                push!(rows, (method = spec.method, hash = spec.hash))
            catch e
                cell_ok = false
                println(tee, "  $(spec.method) ERROR:")
                showerror(tee, e, catch_backtrace())
                println(tee)
            end
            # Fold the method result into the overall smoke status.
            all_ok &= cell_ok
        end

        println(tee, "\n  Note: this smoke test checks that the code runs cleanly.")
        println(tee, "  It does not require the methods to converge.")

        # ====================================================================
        # Section 5: DB integration
        # ====================================================================
        println(tee, "\n--- Section 5: DB integration (script=\"s01\") ---")
        for row in rows
            # Check that the smoke row was written under script="s01".
            if is_done(db, row.hash, problem_name, prob.dim, init.label; script = "s01", production = false)
                println(tee, "  $(row.method): s01 row found")
            else
                all_ok = false
                println(tee, "  $(row.method): s01 row missing")
            end

            # Check that the same row does not appear in the s30 benchmark namespace.
            if !is_done(db, row.hash, problem_name, prob.dim, init.label; script = "s30", production = false)
                println(tee, "  $(row.method): s30 namespace clean")
            else
                all_ok = false
                println(tee, "  $(row.method): s01 row leaked into s30 namespace")
            end
        end

        # ====================================================================
        # Final
        # ====================================================================
        println(tee, "\n" * "="^78)
        println(tee, all_ok ? "  ALL SMOKE CHECKS PASSED" : "  SOME SMOKE CHECKS FAILED")
        println(tee, "="^78)
    finally
        # Always close the log and DB even if one method throws an error.
        teardown_logging(tee, logpath)
        SQLite.close(db)
    end

    return all_ok
end

main()
