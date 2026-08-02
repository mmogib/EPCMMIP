# Production xi-matched AEFBFP companion on the duplicated-identity game.
# This script is deliberately plot-free.

include(joinpath(@__DIR__, "s30_benchmark.jl"))

const GAME_XI_MATCHED_SCRIPT = "s40_xi_matched"
const GAME_XI_MATCHED_EXPERIMENT = "009_xi_matched"
const GAME_XI_MATCHED_DIR = joinpath(GAME_RESULT_ROOT, "supplementary_009")

function game_xi_matched_hash(alg, prob, cfg, source_hash)
    _, base = make_config_hash(alg, prob.name, cfg.eps, cfg.maxiter)
    input = string(base, "|protocol=supplementary_009_xi_matched",
                   "|source_config_hash=", source_hash,
                   "|xi_k=(k+1)^-2|all_other_parameters=fixed",
                   "|stopping=unit_natural_residual|consec=", cfg.consec,
                   "|reps=", cfg.reps, "|warmup=2")
    return bytes2hex(sha256(input))[1:12], input
end

function game_write_xi_matched_csv(db, source_hash, companion_hash, problem, path)
    rows = DBInterface.execute(db, """
        SELECT CASE WHEN r.config_hash=? THEN 'production_cap_1e5' ELSE 'xi_matched' END AS variant,
               r.config_hash, r.init_point, r.seed_idx, r.flag, r.converged,
               r.iterations, r.f_evals, r.cpu_time,
               gm.natural_residual, gm.duality_gap, gm.min_branch_fraction,
               a.distance_to_projection, a.displacement, a.distance_to_solution,
               a.projection_distance_error, a.certificate, a.target_hash
        FROM results r
        JOIN game_final_metrics gm
          ON gm.config_hash=r.config_hash AND gm.problem=r.problem
         AND gm.dimension=r.dimension AND gm.init_point=r.init_point
         AND gm.script=r.script AND gm.production=r.production
        JOIN game_anchor_metrics a
          ON a.config_hash=r.config_hash AND a.problem=r.problem
         AND a.dimension=r.dimension AND a.init_point=r.init_point
         AND a.script=r.script AND a.production=r.production
        WHERE r.problem=? AND r.production=1
          AND ((r.script='s30' AND r.config_hash=?) OR
               (r.script=? AND r.config_hash=?))
        ORDER BY r.seed_idx, variant DESC
    """, (source_hash, problem, source_hash,
           GAME_XI_MATCHED_SCRIPT, companion_hash)) |> DataFrame
    CSV.write(path, rows)
    return rows
end

function game_xi_matched_main()
    runtime = configure_reproducible_runtime!()
    cfg = (eps = 1.0e-6, maxiter = 100_000, consec = 2, reps = 3,
           pilot = false, production = true)
    mkpath(GAME_XI_MATCHED_DIR)
    mkpath(GAME_LOGDIR)
    logpath, tee, _ = setup_logging(GAME_XI_MATCHED_SCRIPT; logdir = GAME_LOGDIR)
    db = open_db(GAME_DB_PATH)
    ensure_game_tables!(db)
    ensure_supplementary_009_tables!(db)
    run_id = GAME_XI_MATCHED_SCRIPT * "_" * Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
    try
        prob = build_duplicated_identity_game(GAME_DEGENERATE_Q; n_inits = 10)
        production_alg = AEFBFP(; GAME_AEFBFP_PARAMS...)
        source_hash, _ = game_config_hash(production_alg, prob, cfg)
        params = merge(GAME_AEFBFP_PARAMS, (xi_exp = 2.0,))
        alg = AEFBFP(; params...)
        hash, hash_input = game_xi_matched_hash(alg, prob, cfg, source_hash)
        ensure_config!(db, alg, prob.name, cfg.eps, cfg.maxiter, hash, hash_input)
        manifest_path = joinpath(GAME_XI_MATCHED_DIR, "$(run_id).json")
        write_run_manifest(manifest_path;
            runtime = runtime,
            protocol = (name = "supplementary_009_xi_matched",
                        source_config_hash = source_hash,
                        stopping = "unit-step natural residual",
                        tolerance = cfg.eps, consecutive_hits = cfg.consec,
                        maxiter = cfg.maxiter, warmup_iterations = 2,
                        repetitions = cfg.reps),
            seeds = prob.metadata.seeds,
            hashes = prob.metadata.hashes,
            parameters = (instance = prob.name,
                          matrix_size = [prob.metadata.m, prob.metadata.n],
                          algorithm = sprint(show, alg),
                          xi_sequence = "(k+1)^-2",
                          all_other_parameters = "production AEFBFP"),
            project_manifest = joinpath(JCODE_ROOT, "Manifest.toml"))
        println(tee, "run_id=$run_id hash=$hash source_hash=$source_hash")
        println(tee, "manifest=$manifest_path")
        warm_up_game_method!(alg, prob)

        for init in prob.initial_points
            if is_done(db, hash, prob.name, prob.dim, init.label;
                       script = GAME_XI_MATCHED_SCRIPT, production = true)
                println(tee, "skip $(init.label)")
                continue
            end
            repetitions = SolverResult[]
            for _ in 1:cfg.reps
                push!(repetitions, solve(alg, prob, copy(init.x0);
                    stopping = game_stopping(prob, cfg), observers = ()))
            end
            result = supplementary_median_result(repetitions)
            collect_history = init.seed_idx == 1
            diagnostic, natural_records, tau_records, branch_fraction =
                run_game_diagnostic_pass(alg, prob, init, cfg;
                                         collect_history = collect_history)
            supplementary_signature(diagnostic) == supplementary_signature(result) ||
                error("Diagnostic signature mismatch for $(init.label)")
            natural_residual = prob.native_residual(result.x, Float64[])
            gap = game_duality_gap(prob, result.x)
            anchor = anchor_diagnostics(prob, init.x0, result.x)
            retain_supplementary_repetitions!(db, repetitions;
                experiment = GAME_XI_MATCHED_EXPERIMENT, config_hash = hash,
                family = "saddle_point", method = "AEFBFP",
                problem = prob.name, dimension = prob.dim,
                init_point = init.label, seed_idx = init.seed_idx,
                run_id = run_id)
            insert_result!(db, hash, prob.name, prob.dim, init.label, init.seed_idx,
                           run_id, result; script = GAME_XI_MATCHED_SCRIPT,
                           native_residual = natural_residual, production = true)
            insert_game_metrics!(db, hash, prob.name, prob.dim, init, run_id,
                                 result, natural_residual, gap, branch_fraction;
                                 script = GAME_XI_MATCHED_SCRIPT,
                                 production = true)
            insert_game_history!(db, hash, prob.name, prob.dim, init,
                                 natural_records, tau_records;
                                 script = GAME_XI_MATCHED_SCRIPT,
                                 production = true)
            insert_anchor_metrics!(db, hash, prob.name, prob.dim, init, anchor;
                                   script = GAME_XI_MATCHED_SCRIPT,
                                   production = true)
            @printf(tee, "%s flag=%s iter=%d F=%d nat=%.3e gap=%.3e dproj=%.3e cert=%.3e\n",
                    init.label, string(result.flag), result.iterations,
                    result.f_evals, natural_residual, gap,
                    anchor.distance_to_projection, anchor.certificate)
        end

        csv_path = joinpath(GAME_XI_MATCHED_DIR,
                            "duplicated_identity_side_by_side.csv")
        rows = game_write_xi_matched_csv(db, source_hash, hash, prob.name, csv_path)
        nrow(rows) == 20 || error("Expected 20 side-by-side rows; found $(nrow(rows))")
        println(tee, "side_by_side_csv=$csv_path")
    finally
        teardown_logging(tee, logpath)
        SQLite.close(db)
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    game_xi_matched_main()
end
