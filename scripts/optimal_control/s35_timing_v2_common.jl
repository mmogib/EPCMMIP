# Shared implementation for the two optimal-control timing-v2 entry points.

function oc_timing_config(args)
    opts, flags = parse_cli(args)
    quick = "quick" in flags
    dims = haskey(opts, "dims") ? parse_int_list(opts["dims"]) : collect(OC_DEFAULT_DIMS)
    methods = haskey(opts, "methods") ? parse_method_list(opts["methods"]) :
              [name(T) for T in OC_METHOD_TYPES]
    starts = haskey(opts, "starts") ? parse(Int, opts["starts"]) : OC_BENCH_INITS
    quick && (dims = [first(dims)]; methods = [first(methods)]; starts = 1)
    1 <= starts <= OC_BENCH_INITS || error("starts must lie in 1:$(OC_BENCH_INITS)")
    min_batch_seconds = quick ? 0.005 : 0.1
    return (quick = quick, dims = dims, methods = methods, starts = starts,
            summary = "summary" in flags, min_batch_seconds = min_batch_seconds,
            warmups = 2, repetitions = 3, eps = OC_EPS_REF,
            maxiter = OC_NMAX_REF, consec = 2)
end

function oc_expected_timing_row(db, source_hash, spec, dim, init_label)
    rows = DBInterface.execute(db, """
        SELECT flag, iterations, f_evals, seed_idx
        FROM results
        WHERE script='s30' AND production=1 AND config_hash=?
          AND problem=? AND dimension=? AND init_point=?
    """, (source_hash, spec.problem_name, dim, init_label)) |> DataFrame
    nrow(rows) == 1 || error("Expected one production row for $(spec.problem_name)/K=$dim/$init_label/$source_hash; found $(nrow(rows))")
    return rows[1, :]
end

function print_oc_timing_summary(db, protocol_hash, tee, spec)
    summary = timing_v2_summary(db; protocol_hash = protocol_hash)
    nrow(summary) == 0 && return println(tee, "No timing-v2 rows for $protocol_hash")
    sort!(summary, [:dimension, :method, :seed_idx])
    println(tee, "\nK method seed median_ms [q1,q3] [min,max] signature")
    for row in eachrow(summary)
        @printf(tee, "%d %-8s %2d %.6f [%.6f,%.6f] [%.6f,%.6f] (%s,%d,%d)\n",
                row.dimension, row.method, row.seed_idx, row.median_ms,
                row.q1_ms, row.q3_ms, row.min_ms, row.max_ms,
                row.expected_flag, row.expected_iterations, row.expected_f_evals)
    end
    path = joinpath(result_root(spec), "timing_v2_summary_$(protocol_hash).csv")
    CSV.write(path, summary)
    println(tee, "summary_csv=$path")
end

function oc_timing_v2_main(spec, family::String, args = ARGS)
    runtime = configure_reproducible_runtime!()
    cfg = oc_timing_config(args)
    protocol_hash = timing_v2_protocol_hash(
        warmups = cfg.warmups, repetitions = cfg.repetitions,
        min_batch_seconds = cfg.min_batch_seconds)
    logpath, tee, _ = setup_logging("s35_timing_v2"; logdir = local_logdir(spec))
    db = open_db(local_db_path(spec))
    ensure_timing_v2_table!(db)
    run_id = "timing_v2_" * Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
    try
        if cfg.summary
            print_oc_timing_summary(db, protocol_hash, tee, spec)
            return nothing
        end
        source_cfg = (eps = cfg.eps, maxiter = cfg.maxiter,
                      reps = cfg.repetitions, consec = cfg.consec)
        definitive_hashes = current_benchmark_hashes(
            db, spec; production = true, eps = cfg.eps, maxiter = cfg.maxiter)
        isempty(definitive_hashes) && error("No definitive OC source hashes for $(spec.problem_name)")
        println(tee, "family=$family protocol_hash=$protocol_hash")

        for dim in cfg.dims
            prob = spec.build_problem(dim; n_inits = OC_BENCH_INITS)
            algorithms = Dict{String,String}()
            source_map = Dict{String,String}()
            for method in cfg.methods
                alg = build_algorithm(db, spec, method)
                source_hash, _ = oc_config_hash(alg, spec, source_cfg)
                source_hash in definitive_hashes || error("Computed source hash $source_hash is not definitive for $method")
                algorithms[method] = sprint(show, alg)
                source_map[method] = source_hash
            end
            manifest_path = joinpath(local_manifestdir(spec), "K$(dim)_$(run_id).json")
            write_run_manifest(manifest_path;
                runtime = runtime,
                protocol = (name = "timing_v2", protocol_hash = protocol_hash,
                            clock = "time_ns", warmups = 2, repetitions = 3,
                            min_batch_seconds = cfg.min_batch_seconds,
                            monitoring = false, elapsed_recording = false,
                            source_script = "s30",
                            signature = ["flag", "iterations", "f_evals"]),
                seeds = prob.metadata.seeds,
                hashes = merge(prob.metadata.hashes, (source_configs = source_map,)),
                parameters = (dimension = dim, methods = cfg.methods,
                              algorithms = algorithms, tolerance = cfg.eps,
                              maxiter = cfg.maxiter, consecutive_hits = cfg.consec),
                project_manifest = joinpath(JCODE_ROOT, "Manifest.toml"))
            println(tee, "K=$dim manifest=$manifest_path")

            for method in cfg.methods
                alg = build_algorithm(db, spec, method)
                source_hash = source_map[method]
                for seed_idx in 1:cfg.starts
                    init = prob.initial_points[seed_idx]
                    expected_row = oc_expected_timing_row(
                        db, source_hash, spec, dim, init.label)
                    Int(expected_row.seed_idx) == seed_idx || error("Seed mismatch for K=$dim/$method/$(init.label)")
                    expected = (Symbol(expected_row.flag), Int(expected_row.iterations),
                                Int(expected_row.f_evals))
                    if timing_v2_done(db, protocol_hash, source_hash,
                                      spec.problem_name, dim, init.label, true;
                                      repetitions = cfg.repetitions)
                        println(tee, "skip K=$dim $method $(init.label)")
                        continue
                    end
                    solve_once = () -> solve(alg, prob, copy(init.x0);
                        stopping = make_stopping(prob, cfg.eps, cfg.maxiter;
                                                 consec = cfg.consec),
                        observers = (), monitor_residual = false,
                        record_elapsed = false)
                    batches = run_timing_v2(solve_once, expected;
                        warmups = 2, repetitions = 3,
                        min_batch_seconds = cfg.min_batch_seconds)
                    for batch in batches
                        insert_timing_v2_repetition!(db, TimingV2Repetition(
                            protocol_hash = protocol_hash, family = family,
                            method = method, source_config_hash = source_hash,
                            problem = spec.problem_name, dimension = dim,
                            init_point = init.label, seed_idx = seed_idx,
                            repetition = batch.repetition,
                            batch_size = batch.batch_size,
                            total_ns = batch.total_ns,
                            ms_per_solve = batch.ms_per_solve,
                            expected_flag = expected[1],
                            expected_iterations = expected[2],
                            expected_f_evals = expected[3], run_id = run_id))
                    end
                    @printf(tee, "K=%d %-8s %s %.6f ms\n", dim, method,
                            init.label,
                            median(batch.ms_per_solve for batch in batches))
                end
            end
        end
        print_oc_timing_summary(db, protocol_hash, tee, spec)
    finally
        teardown_logging(tee, logpath)
        SQLite.close(db)
    end
    return nothing
end
