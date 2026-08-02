# Shared, plot-free production xi-matched AEFBFP companion for optimal control.

const OC_XI_MATCHED_SCRIPT = "s40_xi_matched"
const OC_XI_MATCHED_EXPERIMENT = "009_xi_matched"

function oc_xi_matched_hash(alg, spec, cfg, source_hash)
    _, base = make_config_hash(alg, spec.problem_name, cfg.eps, cfg.maxiter)
    input = string(base, "|protocol=supplementary_009_xi_matched",
                   "|source_config_hash=", source_hash,
                   "|xi_k=(k+1)^-2|all_other_parameters=fixed",
                   "|consec=", cfg.consec, "|reps=", cfg.reps, "|warmup=2")
    return bytes2hex(sha256(input))[1:12], input
end

function oc_write_xi_matched_csv(db, spec, dims, source_hash, companion_hash, path)
    placeholders = join(fill("?", length(dims)), ",")
    params = Any[source_hash, spec.problem_name]
    append!(params, dims)
    append!(params, [source_hash, OC_XI_MATCHED_SCRIPT, companion_hash])
    rows = DBInterface.execute(db, """
        SELECT CASE WHEN r.config_hash=? THEN 'production' ELSE 'xi_matched' END AS variant,
               r.config_hash, r.dimension, r.init_point, r.seed_idx, r.flag,
               r.converged, r.iterations, r.f_evals, r.cpu_time,
               r.native_residual
        FROM results r
        WHERE r.problem=? AND r.dimension IN ($placeholders) AND r.production=1
          AND ((r.script='s30' AND r.config_hash=?) OR
               (r.script=? AND r.config_hash=?))
        ORDER BY r.dimension, r.seed_idx, variant DESC
    """, Tuple(params)) |> DataFrame
    CSV.write(path, rows)
    return rows
end

function oc_xi_matched_main(spec, dims::Vector{Int})
    runtime = configure_reproducible_runtime!()
    cfg = (eps = OC_EPS_REF, maxiter = OC_NMAX_REF, consec = 2, reps = 3,
           production = true)
    outdir = joinpath(result_root(spec), "supplementary_009")
    mkpath(outdir)
    mkpath(local_logdir(spec))
    logpath, tee, _ = setup_logging(OC_XI_MATCHED_SCRIPT; logdir = local_logdir(spec))
    db = open_db(local_db_path(spec))
    ensure_local_tables!(db)
    ensure_supplementary_009_tables!(db)
    run_id = OC_XI_MATCHED_SCRIPT * "_" * Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
    try
        production_alg = build_algorithm(db, spec, "AEFBFP")
        source_hash, _ = oc_config_hash(production_alg, spec, cfg)
        params = merge(OC_SHARED_AEFBFP_PARAMS, (xi_exp = 2.0,))
        alg = AEFBFP(; params...)
        hash, hash_input = oc_xi_matched_hash(alg, spec, cfg, source_hash)
        ensure_config!(db, alg, spec.problem_name, cfg.eps, cfg.maxiter,
                       hash, hash_input)
        println(tee, "run_id=$run_id hash=$hash source_hash=$source_hash")

        for dim in dims
            prob = spec.build_problem(dim; n_inits = 10)
            manifest_path = joinpath(outdir, "K$(dim)_$(run_id).json")
            write_run_manifest(manifest_path;
                runtime = runtime,
                protocol = (name = "supplementary_009_xi_matched",
                            source_config_hash = source_hash,
                            stopping = "native unweighted norm",
                            tolerance = cfg.eps, consecutive_hits = cfg.consec,
                            maxiter = cfg.maxiter, warmup_iterations = 2,
                            repetitions = cfg.reps),
                seeds = prob.metadata.seeds,
                hashes = prob.metadata.hashes,
                parameters = (dimension = dim, algorithm = sprint(show, alg),
                              xi_sequence = "(k+1)^-2",
                              all_other_parameters = "production AEFBFP"),
                project_manifest = joinpath(JCODE_ROOT, "Manifest.toml"))
            println(tee, "K=$dim manifest=$manifest_path")
            solve(alg, prob, copy(first(prob.initial_points).x0);
                  stopping = (MaxIterStopping(2), NanStopping()), observers = ())

            for init in prob.initial_points
                if is_done(db, hash, spec.problem_name, dim, init.label;
                           script = OC_XI_MATCHED_SCRIPT, production = true)
                    println(tee, "skip K=$dim $(init.label)")
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
                    error("Native-residual repetition mismatch for K=$dim/$(init.label)")
                retain_supplementary_repetitions!(db, repetitions;
                    experiment = OC_XI_MATCHED_EXPERIMENT, config_hash = hash,
                    family = spec.problem_name, method = "AEFBFP",
                    problem = spec.problem_name, dimension = dim,
                    init_point = init.label, seed_idx = init.seed_idx,
                    run_id = run_id)
                insert_result!(db, hash, spec.problem_name, dim, init.label,
                               init.seed_idx, run_id, result;
                               script = OC_XI_MATCHED_SCRIPT,
                               native_residual = first(native_values),
                               production = true)
                @printf(tee, "K=%d %s flag=%s iter=%d F=%d native=%.3e cpu=%.6f\n",
                        dim, init.label, string(result.flag), result.iterations,
                        result.f_evals, first(native_values), result.cpu_time)
            end
        end

        csv_path = joinpath(outdir, "$(spec.problem_name)_side_by_side.csv")
        rows = oc_write_xi_matched_csv(db, spec, dims, source_hash, hash, csv_path)
        expected_rows = 2 * 10 * length(dims)
        nrow(rows) == expected_rows ||
            error("Expected $expected_rows side-by-side rows; found $(nrow(rows))")
        println(tee, "side_by_side_csv=$csv_path")
    finally
        teardown_logging(tee, logpath)
        SQLite.close(db)
    end
    return nothing
end
