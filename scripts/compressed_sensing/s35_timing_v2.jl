# External monitoring-free timing pass for definitive compressed sensing.

include(joinpath(@__DIR__, "s30_benchmark.jl"))

function cs_timing_config(args)
    opts, flags = parse_cli(args)
    quick = "quick" in flags
    cases = haskey(opts, "cases") ? parse_case_list(opts["cases"]) :
            [case.problem for case in DEFAULT_CASES]
    methods = haskey(opts, "methods") ? parse_method_list(opts["methods"]) :
              [name(T) for T in METHOD_TYPES]
    starts = haskey(opts, "starts") ? parse(Int, opts["starts"]) : DEFAULT_INITIAL_POINTS
    summary = "summary" in flags
    quick && (cases = [first(cases)]; methods = [first(methods)]; starts = 1)
    1 <= starts <= DEFAULT_INITIAL_POINTS || error("starts must lie in 1:$(DEFAULT_INITIAL_POINTS)")
    min_batch_seconds = quick ? 0.005 : 0.1
    return (quick = quick, cases = cases, methods = methods, starts = starts,
            summary = summary, min_batch_seconds = min_batch_seconds,
            warmups = 2, repetitions = 3, eps = EPS_REF,
            maxiter = NMAX_REF, consec = 2)
end

function cs_expected_timing_row(db, hash, case_id, init_label)
    rows = DBInterface.execute(db, """
        SELECT flag, iterations, f_evals, seed_idx
        FROM results
        WHERE script='s30' AND production=1 AND config_hash=?
          AND problem=? AND init_point=?
    """, (hash, case_id, init_label)) |> DataFrame
    nrow(rows) == 1 || error("Expected one production row for $case_id/$init_label/$hash; found $(nrow(rows))")
    return rows[1, :]
end

function print_cs_timing_summary(db, protocol_hash, tee)
    summary = timing_v2_summary(db; protocol_hash = protocol_hash)
    nrow(summary) == 0 && return println(tee, "No timing-v2 rows for $protocol_hash")
    sort!(summary, [:problem, :method, :seed_idx])
    println(tee, "\ncase method seed median_ms [q1,q3] [min,max] signature")
    for row in eachrow(summary)
        @printf(tee, "%s %-8s %2d %.6f [%.6f,%.6f] [%.6f,%.6f] (%s,%d,%d)\n",
                row.problem, row.method, row.seed_idx, row.median_ms,
                row.q1_ms, row.q3_ms, row.min_ms, row.max_ms,
                row.expected_flag, row.expected_iterations, row.expected_f_evals)
    end
    path = joinpath(RESULT_ROOT, "timing_v2_summary_$(protocol_hash).csv")
    CSV.write(path, summary)
    println(tee, "summary_csv=$path")
end

function cs_timing_v2_main(args = ARGS)
    runtime = configure_reproducible_runtime!()
    cfg = cs_timing_config(args)
    protocol_hash = timing_v2_protocol_hash(
        warmups = cfg.warmups,
        repetitions = cfg.repetitions,
        min_batch_seconds = cfg.min_batch_seconds,
    )
    logpath, tee, _ = setup_logging("s35_timing_v2"; logdir = LOGDIR)
    db = open_db(DB_PATH)
    ensure_timing_v2_table!(db)
    run_id = "timing_v2_" * Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
    try
        if cfg.summary
            print_cs_timing_summary(db, protocol_hash, tee)
            return nothing
        end
        source_cfg = (production = true, eps = cfg.eps, maxiter = cfg.maxiter,
                      cases = cfg.cases)
        source_hashes = current_method_hashes(db, true, cfg.eps, cfg.maxiter, cfg.cases)
        isempty(source_hashes) && error("No definitive compressed-sensing source hashes found")
        manifest_path = joinpath(RESULT_ROOT, "manifests", "$(run_id).json")
        write_run_manifest(manifest_path;
            runtime = runtime,
            protocol = (name = "timing_v2", protocol_hash = protocol_hash,
                        clock = "time_ns", warmups = 2, repetitions = 3,
                        min_batch_seconds = cfg.min_batch_seconds,
                        monitoring = false, elapsed_recording = false,
                        source_script = "s30",
                        signature = ["flag", "iterations", "f_evals"]),
            seeds = (seed_base = MANUSCRIPT_SEED_BASE, starts = collect(1:cfg.starts)),
            hashes = (source_configs = Dict("$(k[1])|$(k[2])" => v for (k, v) in source_hashes),),
            parameters = (cases = cfg.cases, methods = cfg.methods,
                          tolerance = cfg.eps, maxiter = cfg.maxiter,
                          consecutive_hits = cfg.consec),
            project_manifest = joinpath(JCODE_ROOT, "Manifest.toml"))
        println(tee, "protocol_hash=$protocol_hash manifest=$manifest_path")

        for case_id in cfg.cases, method in cfg.methods
            source_hash = get(source_hashes, (case_id, method), nothing)
            source_hash === nothing && error("Missing source hash for $case_id/$method")
            alg = build_algorithm(db, method)
            for seed_idx in 1:cfg.starts
                selected = manuscript_problem_start(case_id, seed_idx; gamma = GAMMA_REF)
                prob, init = selected.prob, selected.init
                expected_row = cs_expected_timing_row(db, source_hash, case_id, init.label)
                expected = (Symbol(expected_row.flag), Int(expected_row.iterations),
                            Int(expected_row.f_evals))
                Int(expected_row.seed_idx) == seed_idx || error("Seed mismatch for $case_id/$method/$(init.label)")
                if timing_v2_done(db, protocol_hash, source_hash, case_id, prob.dim,
                                  init.label, true; repetitions = cfg.repetitions)
                    println(tee, "skip $case_id $method $(init.label)")
                    continue
                end
                solve_once = () -> solve(alg, prob, copy(init.x0);
                    stopping = make_stopping(prob, cfg.eps, cfg.maxiter; consec = cfg.consec),
                    observers = (), monitor_residual = false, record_elapsed = false)
                batches = run_timing_v2(solve_once, expected;
                    warmups = 2, repetitions = 3,
                    min_batch_seconds = cfg.min_batch_seconds)
                for batch in batches
                    insert_timing_v2_repetition!(db, TimingV2Repetition(
                        protocol_hash = protocol_hash, family = "compressed_sensing",
                        method = method, source_config_hash = source_hash,
                        problem = case_id, dimension = prob.dim,
                        init_point = init.label, seed_idx = seed_idx,
                        repetition = batch.repetition, batch_size = batch.batch_size,
                        total_ns = batch.total_ns, ms_per_solve = batch.ms_per_solve,
                        expected_flag = expected[1], expected_iterations = expected[2],
                        expected_f_evals = expected[3], run_id = run_id))
                end
                @printf(tee, "%s %-8s %s %.6f ms\n", case_id, method, init.label,
                        median(batch.ms_per_solve for batch in batches))
            end
        end
        print_cs_timing_summary(db, protocol_hash, tee)
    finally
        teardown_logging(tee, logpath)
        SQLite.close(db)
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    cs_timing_v2_main()
end
