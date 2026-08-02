# External monitoring-free timing pass for definitive matrix games.

include(joinpath(@__DIR__, "s30_benchmark.jl"))

function game_timing_config(args)
    opts, flags = parse_game_cli(args)
    quick = "quick" in flags
    instances = haskey(opts, "instances") ?
        parse_game_list(opts["instances"], GAME_INSTANCE_NAMES, "instance") :
        copy(GAME_INSTANCE_NAMES)
    methods = haskey(opts, "methods") ?
        parse_game_list(opts["methods"], GAME_METHOD_NAMES, "method") :
        copy(GAME_METHOD_NAMES)
    starts = haskey(opts, "starts") ? parse(Int, opts["starts"]) : 10
    quick && (instances = [first(instances)]; methods = [first(methods)]; starts = 1)
    1 <= starts <= 10 || error("starts must lie in 1:10")
    min_batch_seconds = quick ? 0.005 : 0.1
    return (quick = quick, instances = instances, methods = methods,
            starts = starts, summary = "summary" in flags,
            min_batch_seconds = min_batch_seconds, warmups = 2,
            repetitions = 3, reps = 3, eps = 1.0e-6, maxiter = 10_000,
            consec = 2, pilot = false, production = true)
end

function game_expected_timing_row(db, source_hash, problem, dim, init_label)
    rows = DBInterface.execute(db, """
        SELECT flag, iterations, f_evals, seed_idx
        FROM results
        WHERE script='s30' AND production=1 AND config_hash=?
          AND problem=? AND dimension=? AND init_point=?
    """, (source_hash, problem, dim, init_label)) |> DataFrame
    nrow(rows) == 1 || error("Expected one production row for $problem/$init_label/$source_hash; found $(nrow(rows))")
    return rows[1, :]
end

function print_game_timing_summary_v2(db, protocol_hash, tee)
    summary = timing_v2_summary(db; protocol_hash = protocol_hash)
    nrow(summary) == 0 && return println(tee, "No timing-v2 rows for $protocol_hash")
    sort!(summary, [:problem, :method, :seed_idx])
    println(tee, "\nproblem method seed median_ms [q1,q3] [min,max] signature")
    for row in eachrow(summary)
        @printf(tee, "%s %-8s %2d %.6f [%.6f,%.6f] [%.6f,%.6f] (%s,%d,%d)\n",
                row.problem, row.method, row.seed_idx, row.median_ms,
                row.q1_ms, row.q3_ms, row.min_ms, row.max_ms,
                row.expected_flag, row.expected_iterations, row.expected_f_evals)
    end
    path = joinpath(GAME_RESULT_ROOT, "timing_v2_summary_$(protocol_hash).csv")
    CSV.write(path, summary)
    println(tee, "summary_csv=$path")
end

function game_timing_v2_main(args = ARGS)
    runtime = configure_reproducible_runtime!()
    cfg = game_timing_config(args)
    protocol_hash = timing_v2_protocol_hash(
        warmups = cfg.warmups, repetitions = cfg.repetitions,
        min_batch_seconds = cfg.min_batch_seconds)
    logpath, tee, _ = setup_logging("s35_timing_v2"; logdir = GAME_LOGDIR)
    db = open_db(GAME_DB_PATH)
    ensure_timing_v2_table!(db)
    run_id = "timing_v2_" * Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
    try
        if cfg.summary
            print_game_timing_summary_v2(db, protocol_hash, tee)
            return nothing
        end
        println(tee, "protocol_hash=$protocol_hash")
        for instance in cfg.instances
            prob = build_game_instance(instance; n_inits = 10)
            algorithms = Dict{String,String}()
            source_map = Dict{String,String}()
            for method in cfg.methods
                alg = build_game_algorithm(method, prob)
                source_hash, _ = game_config_hash(alg, prob, cfg)
                algorithms[method] = sprint(show, alg)
                source_map[method] = source_hash
            end
            manifest_path = joinpath(GAME_MANIFESTDIR, "$(instance)_$(run_id).json")
            write_run_manifest(manifest_path;
                runtime = runtime,
                protocol = (name = "timing_v2", protocol_hash = protocol_hash,
                            clock = "time_ns", warmups = 2, repetitions = 3,
                            min_batch_seconds = cfg.min_batch_seconds,
                            monitoring = false, elapsed_recording = false,
                            source_script = "s30",
                            signature = ["flag", "iterations", "f_evals"]),
                seeds = prob.metadata.seeds,
                hashes = merge(prob.metadata.hashes,
                               (source_configs = source_map,)),
                parameters = (instance = instance,
                              matrix_size = [prob.metadata.m, prob.metadata.n],
                              spectral_norm = prob.metadata.L,
                              methods = cfg.methods, algorithms = algorithms,
                              tolerance = cfg.eps, maxiter = cfg.maxiter,
                              consecutive_hits = cfg.consec),
                project_manifest = joinpath(JCODE_ROOT, "Manifest.toml"))
            println(tee, "$instance manifest=$manifest_path")

            for method in cfg.methods
                alg = build_game_algorithm(method, prob)
                source_hash = source_map[method]
                for seed_idx in 1:cfg.starts
                    init = prob.initial_points[seed_idx]
                    expected_row = game_expected_timing_row(
                        db, source_hash, prob.name, prob.dim, init.label)
                    Int(expected_row.seed_idx) == seed_idx || error("Seed mismatch for $instance/$method/$(init.label)")
                    expected = (Symbol(expected_row.flag), Int(expected_row.iterations),
                                Int(expected_row.f_evals))
                    if timing_v2_done(db, protocol_hash, source_hash, prob.name,
                                      prob.dim, init.label, true;
                                      repetitions = cfg.repetitions)
                        println(tee, "skip $instance $method $(init.label)")
                        continue
                    end
                    solve_once = () -> solve(alg, prob, copy(init.x0);
                        stopping = game_stopping(prob, cfg), observers = (),
                        monitor_residual = false, record_elapsed = false)
                    batches = run_timing_v2(solve_once, expected;
                        warmups = 2, repetitions = 3,
                        min_batch_seconds = cfg.min_batch_seconds)
                    for batch in batches
                        insert_timing_v2_repetition!(db, TimingV2Repetition(
                            protocol_hash = protocol_hash, family = "saddle_point",
                            method = method, source_config_hash = source_hash,
                            problem = prob.name, dimension = prob.dim,
                            init_point = init.label, seed_idx = seed_idx,
                            repetition = batch.repetition,
                            batch_size = batch.batch_size,
                            total_ns = batch.total_ns,
                            ms_per_solve = batch.ms_per_solve,
                            expected_flag = expected[1],
                            expected_iterations = expected[2],
                            expected_f_evals = expected[3], run_id = run_id))
                    end
                    @printf(tee, "%s %-8s %s %.6f ms\n", instance, method,
                            init.label,
                            median(batch.ms_per_solve for batch in batches))
                end
            end
        end
        print_game_timing_summary_v2(db, protocol_hash, tee)
    finally
        teardown_logging(tee, logpath)
        SQLite.close(db)
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    game_timing_v2_main()
end
