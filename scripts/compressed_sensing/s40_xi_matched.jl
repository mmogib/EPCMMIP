# Production xi-matched AEFBFP companion for compressed-sensing Case 1.
# This script is deliberately plot-free.

include(joinpath(@__DIR__, "s30_benchmark.jl"))

const CS_XI_MATCHED_SCRIPT = "s40_xi_matched"
const CS_XI_MATCHED_EXPERIMENT = "009_xi_matched"
const CS_XI_MATCHED_DIR = joinpath(RESULT_ROOT, "supplementary_009")

function cs_xi_matched_hash(alg, case_id, cfg, source_hash)
    _, base = make_config_hash(alg, case_id, cfg.eps, cfg.maxiter)
    input = string(base, "|protocol=supplementary_009_xi_matched",
                   "|source_config_hash=", source_hash,
                   "|xi_k=(k+1)^-2|all_other_parameters=fixed",
                   "|stopping=successive_displacement|consec=", cfg.consec,
                   "|reps=", cfg.reps, "|warmup=2")
    return bytes2hex(sha256(input))[1:12], input
end

function cs_write_xi_matched_csv(db, source_hash, companion_hash, path)
    rows = DBInterface.execute(db, """
        SELECT CASE WHEN r.config_hash=? THEN 'production' ELSE 'xi_matched' END AS variant,
               r.config_hash, r.init_point, r.seed_idx, r.flag, r.converged,
               r.iterations, r.f_evals, r.cpu_time,
               m.objective, m.reconstruction_mse, m.common_residual
        FROM results r
        JOIN cs_final_metrics m
          ON m.config_hash=r.config_hash AND m.problem=r.problem
         AND m.dimension=r.dimension AND m.init_point=r.init_point
         AND m.script=r.script AND m.production=r.production
        WHERE r.problem=? AND r.production=1
          AND ((r.script='s30' AND r.config_hash=?) OR
               (r.script=? AND r.config_hash=?))
        ORDER BY r.seed_idx, variant DESC
    """, (source_hash, DEFAULT_CASES[1].problem, source_hash,
           CS_XI_MATCHED_SCRIPT, companion_hash)) |> DataFrame
    CSV.write(path, rows)
    return rows
end

function cs_xi_matched_main()
    runtime = configure_reproducible_runtime!()
    mkpath(CS_XI_MATCHED_DIR)
    mkpath(LOGDIR)
    logpath, tee, _ = setup_logging(CS_XI_MATCHED_SCRIPT; logdir = LOGDIR)
    db = open_db(DB_PATH)
    ensure_local_tables!(db)
    ensure_supplementary_009_tables!(db)
    run_id = CS_XI_MATCHED_SCRIPT * "_" * Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
    cfg = (eps = EPS_REF, maxiter = NMAX_REF, consec = 2, reps = 3,
           production = true)
    case = DEFAULT_CASES[1]
    try
        prob = build_manuscript_problem(case; case_index = 1, gamma = GAMMA_REF,
                                         n_inits = 10)
        production_alg = AEFBFP(; MANUSCRIPT_AEFBFP_PARAMS...)
        source_hash, _ = cs_config_hash(production_alg, case.problem, cfg)
        params = merge(MANUSCRIPT_AEFBFP_PARAMS, (xi_exp = 2.0,))
        alg = AEFBFP(; params...)
        hash, hash_input = cs_xi_matched_hash(alg, case.problem, cfg, source_hash)
        ensure_config!(db, alg, case.problem, cfg.eps, cfg.maxiter, hash, hash_input)

        manifest_path = joinpath(CS_XI_MATCHED_DIR, "$(run_id).json")
        write_run_manifest(manifest_path;
            runtime = runtime,
            protocol = (name = "supplementary_009_xi_matched",
                        source_config_hash = source_hash,
                        stopping = "successive_displacement",
                        tolerance = cfg.eps, consecutive_hits = cfg.consec,
                        maxiter = cfg.maxiter, warmup_iterations = 2,
                        repetitions = cfg.reps),
            seeds = prob.metadata.seeds,
            hashes = prob.metadata.hashes,
            parameters = (case = case, algorithm = sprint(show, alg),
                          xi_sequence = "(k+1)^-2",
                          all_other_parameters = "production AEFBFP"),
            project_manifest = joinpath(JCODE_ROOT, "Manifest.toml"))
        println(tee, "run_id=$run_id hash=$hash source_hash=$source_hash")
        println(tee, "manifest=$manifest_path")

        solve(alg, prob, copy(first(prob.initial_points).x0);
              stopping = (MaxIterStopping(2), NanStopping()), observers = ())
        for init in prob.initial_points
            if is_done(db, hash, case.problem, case.N, init.label;
                       script = CS_XI_MATCHED_SCRIPT, production = true)
                println(tee, "skip $(init.label)")
                continue
            end
            repetitions = SolverResult[]
            native_values = Float64[]
            for _ in 1:cfg.reps
                nrec = NativeResRecorder(prob.native_residual)
                result = solve(alg, prob, copy(init.x0);
                    stopping = make_stopping(prob, cfg.eps, cfg.maxiter;
                                             consec = cfg.consec),
                    observers = (nrec,))
                push!(repetitions, result)
                push!(native_values, nrec.value)
            end
            result = supplementary_median_result(repetitions)
            all(==(first(native_values)), native_values) ||
                error("Native-residual repetition mismatch for $(init.label)")
            retain_supplementary_repetitions!(db, repetitions;
                experiment = CS_XI_MATCHED_EXPERIMENT, config_hash = hash,
                family = "compressed_sensing", method = "AEFBFP",
                problem = case.problem, dimension = case.N,
                init_point = init.label, seed_idx = init.seed_idx, run_id = run_id)
            insert_result!(db, hash, case.problem, case.N, init.label, init.seed_idx,
                           run_id, result; script = CS_XI_MATCHED_SCRIPT,
                           native_residual = first(native_values), production = true)
            insert_final_metrics!(db, hash, case.problem, case.N, init.label,
                                  init.seed_idx;
                objective = cs_objective(result.x, prob),
                reconstruction_mse = reconstruction_mse(result.x, prob.metadata.x_star),
                common_residual = cs_scaled_optimality_residual(result.x, prob),
                script = CS_XI_MATCHED_SCRIPT, production = true)
            @printf(tee, "%s flag=%s iter=%d F=%d native=%.3e cpu=%.6f\n",
                    init.label, string(result.flag), result.iterations,
                    result.f_evals, first(native_values), result.cpu_time)
        end

        csv_path = joinpath(CS_XI_MATCHED_DIR, "cs_case1_side_by_side.csv")
        rows = cs_write_xi_matched_csv(db, source_hash, hash, csv_path)
        nrow(rows) == 20 || error("Expected 20 side-by-side rows; found $(nrow(rows))")
        println(tee, "side_by_side_csv=$csv_path")
    finally
        teardown_logging(tee, logpath)
        SQLite.close(db)
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    cs_xi_matched_main()
end
