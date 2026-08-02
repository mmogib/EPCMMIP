# Definitive non-cocoercive saddle-point benchmark (numerical entrypoint).

include(joinpath(@__DIR__, "..", "..", "src", "includes.jl"))
include(joinpath(@__DIR__, "problem_definition.jl"))
include(joinpath(@__DIR__, "halpern_forward_backward.jl"))
include(joinpath(@__DIR__, "diagnostics.jl"))

const GAME_RESULT_ROOT = joinpath(JCODE_ROOT, "results", "saddle_point")
const GAME_DB_PATH = joinpath(GAME_RESULT_ROOT, "experiments.db")
const GAME_LOGDIR = joinpath(GAME_RESULT_ROOT, "logs")
const GAME_MANIFESTDIR = joinpath(GAME_RESULT_ROOT, "manifests")
const GAME_INSTANCE_NAMES = ["random_100", "random_500", "random_1000", "duplicated_identity_q100"]
const GAME_METHOD_NAMES = ["AEFBFP", "VAFBS", "MDITSM", "RFBSM", "IRFBSM", "IFRAB", "HFB"]
const GAME_METHOD_TYPES = Dict(
    "VAFBS" => VAFBS,
    "MDITSM" => MDITSM,
    "RFBSM" => RFBSM,
    "IRFBSM" => IRFBSM,
    "IFRAB" => IFRAB,
)
const GAME_AEFBFP_PARAMS = (
    mu = 0.32,
    tau_0 = 0.05,
    xi_rule = :power,
    sigma_rule = :power,
    xi_exp = 1.11,
    sigma_exp = 0.97,
    sigma_scale = 0.024,
)

function parse_game_cli(args)
    opts = Dict{String,String}()
    flags = Set{String}()
    for arg in args
        startswith(arg, "--") || continue
        parts = split(arg[3:end], "="; limit = 2)
        length(parts) == 2 ? (opts[parts[1]] = parts[2]) : push!(flags, parts[1])
    end
    return opts, flags
end

function parse_game_list(text::String, allowed, label::String)
    values = filter(value -> !isempty(value), strip.(split(text, ",")))
    isempty(values) && throw(ArgumentError("Expected at least one $label."))
    for value in values
        value in allowed || throw(ArgumentError("Unknown $label '$value'."))
    end
    return values
end

function read_game_config(args)
    opts, flags = parse_game_cli(args)
    quick = "quick" in flags
    pilot = "pilot-1000" in flags
    quick && pilot && throw(ArgumentError("--quick and --pilot-1000 are mutually exclusive."))
    default_instances = pilot ? ["random_1000"] : (quick ? ["random_100"] : copy(GAME_INSTANCE_NAMES))
    instances = haskey(opts, "instances") ?
        parse_game_list(opts["instances"], GAME_INSTANCE_NAMES, "instance") : default_instances
    methods = haskey(opts, "methods") ?
        parse_game_list(opts["methods"], GAME_METHOD_NAMES, "method") : copy(GAME_METHOD_NAMES)
    initial_points = haskey(opts, "initial-points") ? parse(Int, opts["initial-points"]) : (quick || pilot ? 1 : 10)
    eps_value = haskey(opts, "eps") ? parse(Float64, opts["eps"]) : 1.0e-6
    maxiter = haskey(opts, "maxiter") ? parse(Int, opts["maxiter"]) : (quick ? 20 : 10_000)
    consec = haskey(opts, "consec") ? parse(Int, opts["consec"]) : 2
    reps = haskey(opts, "reps") ? parse(Int, opts["reps"]) : (quick ? 1 : 3)
    initial_points >= 1 || throw(ArgumentError("initial-points must be >= 1"))
    eps_value > 0 || throw(ArgumentError("eps must be > 0"))
    maxiter >= 1 || throw(ArgumentError("maxiter must be >= 1"))
    consec >= 1 || throw(ArgumentError("consec must be >= 1"))
    reps >= 1 || throw(ArgumentError("reps must be >= 1"))
    return (
        quick = quick,
        pilot = pilot,
        production = !(quick || pilot),
        instances = instances,
        methods = methods,
        initial_points = initial_points,
        eps = eps_value,
        maxiter = maxiter,
        consec = consec,
        reps = reps,
        force = "force" in flags,
        summary = "summary" in flags,
    )
end

function build_game_algorithm(method::AbstractString, prob::TestProblem)
    method == "AEFBFP" && return AEFBFP(; GAME_AEFBFP_PARAMS...)
    method == "HFB" && return HalpernForwardBackward(0.99 / prob.metadata.L)
    return GAME_METHOD_TYPES[String(method)](:paper)
end

game_result_signature(result::SolverResult) = (result.flag, result.iterations, result.f_evals)

function game_median_cpu_result(results::AbstractVector{<:SolverResult})
    isempty(results) && error("Cannot aggregate an empty repetition set.")
    signature = game_result_signature(first(results))
    all(game_result_signature(result) == signature for result in results) ||
        error("Game repetition signature mismatch: expected $signature, got $([game_result_signature(r) for r in results])")
    representative = first(results)
    return make_result(
        converged = representative.converged,
        iterations = representative.iterations,
        f_evals = representative.f_evals,
        cpu_time = median(result.cpu_time for result in results),
        x = copy(representative.x),
        flag = representative.flag,
        history = copy(representative.history),
        residual = representative.residual,
        scaled_residual = representative.scaled_residual,
    )
end

function game_config_hash(alg::AbstractAlgorithm, prob::TestProblem, cfg)
    _, base = make_config_hash(alg, prob.name, cfg.eps, cfg.maxiter)
    input = string(base,
                   "|protocol=game_manuscript_v1",
                   "|matrix_sha256=", prob.metadata.hashes.matrix,
                   "|stopping=unit_natural_residual",
                   "|consec=", cfg.consec,
                   "|reps=", cfg.reps,
                   "|cpu=median",
                   "|warmup=2")
    return bytes2hex(sha256(input))[1:12], input
end

game_stopping(prob, cfg) = (
    NativeResStopping(prob.native_residual, cfg.eps; consec = cfg.consec),
    MaxIterStopping(cfg.maxiter),
    NanStopping(),
)

function warm_up_game_method!(alg, prob)
    solve(alg, prob, copy(first(prob.initial_points).x0);
          stopping = (MaxIterStopping(2), NanStopping()), observers = ())
    return nothing
end

function write_game_manifest(prob, cfg, run_id, runtime)
    algorithms = Dict(method => sprint(show, build_game_algorithm(method, prob)) for method in cfg.methods)
    path = joinpath(GAME_MANIFESTDIR, "$(prob.name)_$(run_id).json")
    return write_run_manifest(
        path;
        runtime = runtime,
        protocol = (
            name = "game_manuscript_v1",
            stopping = "unit-step natural residual",
            tolerance = cfg.eps,
            consecutive_hits = cfg.consec,
            maxiter = cfg.maxiter,
            repetitions = cfg.reps,
            warmup_iterations = 2,
            cpu_aggregation = "median",
            monitor_B_calls_counted = false,
            pilot = cfg.pilot,
        ),
        seeds = prob.metadata.seeds,
        hashes = prob.metadata.hashes,
        parameters = (
            matrix_size = [prob.metadata.m, prob.metadata.n],
            spectral_norm = prob.metadata.L,
            algorithms = algorithms,
        ),
        project_manifest = joinpath(JCODE_ROOT, "Manifest.toml"),
    )
end

function run_game_diagnostic_pass(alg, prob, init, cfg; collect_history::Bool)
    natural = GameNaturalHistory(prob.native_residual)
    if alg isa AEFBFP
        tau = AEFBFPTauObserver(alg.xi_exp, alg.tau_0)
        observers = collect_history ? (natural, tau) : (tau,)
        result = solve(alg, prob, copy(init.x0);
                       stopping = game_stopping(prob, cfg), observers = observers)
        return result, natural.records, tau.records, min_branch_fraction(tau)
    end
    collect_history || error("A non-AEFBFP diagnostic pass is only needed for a stored representative history.")
    result = solve(alg, prob, copy(init.x0);
                   stopping = game_stopping(prob, cfg), observers = (natural,))
    return result, natural.records, TauBranchRecord[], NaN
end

function run_game_cell!(db, prob, cfg, run_id, script_tag, method, init)
    alg = build_game_algorithm(method, prob)
    hash, hash_input = game_config_hash(alg, prob, cfg)
    ensure_config!(db, alg, prob.name, cfg.eps, cfg.maxiter, hash, hash_input)
    if !cfg.force && is_done(db, hash, prob.name, prob.dim, init.label;
                             script = script_tag, production = cfg.production)
        return (status = :skipped,)
    end

    repetitions = SolverResult[]
    try
        for _ in 1:cfg.reps
            push!(repetitions, solve(alg, prob, copy(init.x0);
                                     stopping = game_stopping(prob, cfg), observers = ()))
        end
        result = game_median_cpu_result(repetitions)
        collect_history = init.seed_idx == 1
        natural_records = GameNaturalRecord[]
        tau_records = TauBranchRecord[]
        branch_fraction = NaN
        if alg isa AEFBFP || collect_history
            diagnostic, natural_records, tau_records, branch_fraction =
                run_game_diagnostic_pass(alg, prob, init, cfg; collect_history = collect_history)
            game_result_signature(diagnostic) == game_result_signature(result) ||
                error("Diagnostic-pass signature mismatch: timed=$(game_result_signature(result)), diagnostic=$(game_result_signature(diagnostic))")
        end

        natural_residual = prob.native_residual(result.x, Float64[])
        gap = game_duality_gap(prob, result.x)
        insert_result!(db, hash, prob.name, prob.dim, init.label, init.seed_idx,
                       String(run_id), result; script = script_tag,
                       native_residual = natural_residual, production = cfg.production)
        insert_game_metrics!(db, hash, prob.name, prob.dim, init, run_id, result,
                             natural_residual, gap, branch_fraction;
                             script = script_tag, production = cfg.production)
        insert_game_history!(db, hash, prob.name, prob.dim, init,
                             natural_records, tau_records;
                             script = script_tag, production = cfg.production)
        if prob.metadata.instance_kind === :duplicated_identity
            anchor = anchor_diagnostics(prob, init.x0, result.x)
            insert_anchor_metrics!(db, hash, prob.name, prob.dim, init, anchor;
                                   script = script_tag, production = cfg.production)
        end
        return (status = :ran, result = result, natural_residual = natural_residual,
                gap = gap, branch_fraction = branch_fraction)
    catch err
        return (status = :error, error_text = sprint(showerror, err, catch_backtrace()))
    end
end

function print_game_summary(db, tee; run_id = nothing)
    where_clause = run_id === nothing ? "" : "WHERE r.run_id = ?"
    params = run_id === nothing ? () : (run_id,)
    df = DBInterface.execute(db, """
        SELECT r.problem, c.method, r.converged, r.iterations, r.f_evals,
               r.cpu_time, gm.duality_gap
        FROM results r
        JOIN configs c ON c.config_hash = r.config_hash
        JOIN game_final_metrics gm
          ON gm.config_hash=r.config_hash AND gm.problem=r.problem
         AND gm.dimension=r.dimension AND gm.init_point=r.init_point
         AND gm.script=r.script AND gm.production=r.production
        $where_clause
    """, params) |> DataFrame
    nrow(df) == 0 && return println(tee, "No game rows found.")
    println(tee, "\n--- game benchmark summary ---")
    for problem in unique(String.(df.problem))
        println(tee, "[$problem]")
        for method in GAME_METHOD_NAMES
            sub = df[(df.problem .== problem) .& (df.method .== method), :]
            nrow(sub) == 0 && continue
            solved = sub[sub.converged .== 1, :]
            if nrow(solved) == 0
                @printf(tee, "  %-8s %2d/%-2d DNC\n", method, 0, nrow(sub))
            else
                @printf(tee, "  %-8s %2d/%-2d iter=%8.1f fe=%8.1f gap=%9.2e cpu=%8.3f\n",
                        method, nrow(solved), nrow(sub), median(Float64.(solved.iterations)),
                        median(Float64.(solved.f_evals)), median(Float64.(solved.duality_gap)),
                        median(Float64.(solved.cpu_time)))
            end
        end
    end
end

function game_benchmark_main(args = ARGS)
    runtime = configure_reproducible_runtime!()
    cfg = read_game_config(args)
    cfg.production && cfg.reps != 3 && error("Production game runs require exactly 3 repetitions.")
    cfg.pilot && cfg.reps != 3 && error("The 1000x1000 pilot requires exactly 3 repetitions.")
    mkpath(GAME_LOGDIR)
    mkpath(GAME_MANIFESTDIR)
    logpath, tee, _ = setup_logging("s30_benchmark"; logdir = GAME_LOGDIR)
    db = open_db(GAME_DB_PATH)
    ensure_game_tables!(db)
    run_id = (cfg.pilot ? "pilot1000_" : "s30_") * Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
    script_tag = cfg.pilot ? "s30_pilot_1000" : "s30"
    try
        if cfg.summary
            print_game_summary(db, tee)
            return nothing
        end
        println(tee, "="^78)
        println(tee, "  saddle-point matrix-game benchmark")
        println(tee, "="^78)
        println(tee, "  instances : $(join(cfg.instances, ", "))")
        println(tee, "  methods   : $(join(cfg.methods, ", "))")
        println(tee, "  starts    : $(cfg.initial_points)")
        println(tee, "  stopping  : unit natural residual <= $(cfg.eps), $(cfg.consec) consecutive, cap $(cfg.maxiter)")
        println(tee, "  timing    : warm-up 2; $(cfg.reps) repetitions; median CPU")
        println(tee, "  runtime   : Julia $(runtime.julia_version), Julia threads=$(runtime.julia_threads), BLAS threads=$(runtime.blas_threads)")
        println(tee, "  run_id    : $run_id")

        for instance in cfg.instances
            prob = build_game_instance(instance; n_inits = cfg.initial_points)
            manifest = write_game_manifest(prob, cfg, run_id, runtime)
            println(tee, "\n[$instance] size=$(prob.metadata.m)x$(prob.metadata.n) L=$(prob.metadata.L)")
            println(tee, "  matrix_sha256=$(prob.metadata.hashes.matrix)")
            println(tee, "  manifest=$manifest")
            for method in cfg.methods
                alg = build_game_algorithm(method, prob)
                warm_up_game_method!(alg, prob)
                for init in prob.initial_points
                    cell = run_game_cell!(db, prob, cfg, run_id, script_tag, method, init)
                    if cell.status === :skipped
                        println(tee, "  $method init=$(init.label) skipped (checkpoint present)")
                    elseif cell.status === :error
                        println(tee, "  $method init=$(init.label) ANOMALY: $(cell.error_text)")
                    else
                        branch = isfinite(cell.branch_fraction) ? @sprintf(" branch=%.3f", cell.branch_fraction) : ""
                        @printf(tee, "  %-8s init=%-6s conv=%-5s iter=%5d fe=%6d nat=%9.2e gap=%9.2e cpu=%8.3f%s\n",
                                method, init.label, string(cell.result.converged),
                                cell.result.iterations, cell.result.f_evals,
                                cell.natural_residual, cell.gap, cell.result.cpu_time, branch)
                    end
                end
            end
        end
        print_game_summary(db, tee; run_id = run_id)
    finally
        teardown_logging(tee, logpath)
        SQLite.close(db)
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    game_benchmark_main()
end
