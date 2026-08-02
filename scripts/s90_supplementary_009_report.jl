# Plot-free audit/report generator for the production data commissioned in 009.

include(joinpath(@__DIR__, "..", "src", "includes.jl"))

const REPORT_009_ROOT = joinpath(JCODE_ROOT, "results", "supplementary_009")
const REPORT_009_TIMING_HASH = timing_v2_protocol_hash(
    warmups = 2, repetitions = 3, min_batch_seconds = 0.1)

function quantile_or_missing(values, p)
    isempty(values) && return missing
    return quantile(Float64.(values), p)
end

function write_family_timing_reports(db_path, family, outdir)
    db = open_db(db_path)
    try
        ensure_timing_v2_table!(db)
        per_start = timing_v2_summary(db; protocol_hash = REPORT_009_TIMING_HASH)
        if nrow(per_start) == 0
            println("BLOCKED timing-v2 missing: $family")
            return false
        end
        sort!(per_start, [:problem, :dimension, :method, :seed_idx])
        CSV.write(joinpath(outdir, "$(family)_timing_v2_per_start.csv"), per_start)

        rows = NamedTuple[]
        for group in groupby(per_start, [:problem, :dimension, :method])
            solved = group[group.expected_flag .== "converged", :]
            all_medians = Float64.(group.median_ms)
            medians = Float64.(solved.median_ms)
            push!(rows, (
                family = family,
                problem = String(group.problem[1]),
                dimension = Int(group.dimension[1]),
                method = String(group.method[1]),
                timed_starts = nrow(group),
                successful_starts = nrow(solved),
                retained_repetitions = sum(Int.(group.repetitions)),
                all_start_median_ms = quantile_or_missing(all_medians, 0.5),
                all_start_q1_ms = quantile_or_missing(all_medians, 0.25),
                all_start_q3_ms = quantile_or_missing(all_medians, 0.75),
                all_start_min_ms = minimum(all_medians),
                all_start_max_ms = maximum(all_medians),
                median_ms = quantile_or_missing(medians, 0.5),
                q1_ms = quantile_or_missing(medians, 0.25),
                q3_ms = quantile_or_missing(medians, 0.75),
                min_ms = isempty(medians) ? missing : minimum(medians),
                max_ms = isempty(medians) ? missing : maximum(medians),
            ))
        end
        cells = DataFrame(rows)
        sort!(cells, [:problem, :dimension, :method])
        CSV.write(joinpath(outdir, "$(family)_timing_v2_cells.csv"), cells)

        gate = DBInterface.execute(db, """
            SELECT COUNT(*) AS retained_rows,
                   COUNT(DISTINCT source_config_hash || '|' || problem || '|' ||
                         dimension || '|' || init_point) AS timed_starts,
                   MIN(total_ns) AS minimum_batch_ns,
                   MIN(batch_size) AS minimum_batch_size,
                   MAX(batch_size) AS maximum_batch_size
            FROM timing_v2_repetitions WHERE protocol_hash=?
        """, (REPORT_009_TIMING_HASH,)) |> DataFrame
        CSV.write(joinpath(outdir, "$(family)_timing_v2_gate.csv"), gate)
        return true
    finally
        SQLite.close(db)
    end
end

function write_extended_game_reports(db_path, outdir)
    db = open_db(db_path)
    try
        rows = DBInterface.execute(db, """
            SELECT c.method, r.config_hash, r.problem, r.dimension,
                   r.init_point, r.seed_idx, r.flag, r.converged,
                   r.iterations, r.f_evals, r.cpu_time,
                   gm.natural_residual, gm.duality_gap,
                   a.distance_to_projection, a.displacement,
                   a.distance_to_solution, a.projection_distance_error,
                   a.certificate, a.target_hash
            FROM results r JOIN configs c ON c.config_hash=r.config_hash
            JOIN game_final_metrics gm
              ON gm.config_hash=r.config_hash AND gm.problem=r.problem
             AND gm.dimension=r.dimension AND gm.init_point=r.init_point
             AND gm.script=r.script AND gm.production=r.production
            JOIN game_anchor_metrics a
              ON a.config_hash=r.config_hash AND a.problem=r.problem
             AND a.dimension=r.dimension AND a.init_point=r.init_point
             AND a.script=r.script AND a.production=r.production
            WHERE r.script='s30' AND r.production=1
              AND r.problem='duplicated_identity_q100' AND c.maxiter=100000
              AND c.hash_input LIKE '%protocol=game_manuscript_v1%'
            ORDER BY c.method, r.seed_idx
        """) |> DataFrame
        if nrow(rows) != 70
            println("BLOCKED extended duplicated-identity rows: expected 70, found $(nrow(rows))")
            return false
        end
        CSV.write(joinpath(outdir, "duplicated_identity_cap_1e5_per_start.csv"), rows)
        anchor_methods = Set(["AEFBFP", "IFRAB", "RFBSM"])
        anchors = rows[in.(rows.method, Ref(anchor_methods)), :]
        nrow(anchors) == 30 || error("Expected 30 requested anchor rows")
        CSV.write(joinpath(outdir, "duplicated_identity_cap_1e5_anchors.csv"), anchors)

        summary_rows = NamedTuple[]
        for group in groupby(rows, :method)
            solved = group[group.converged .== 1, :]
            push!(summary_rows, (
                method = String(group.method[1]),
                successes = nrow(solved), total = nrow(group),
                median_iterations = median(Float64.(group.iterations)),
                min_iterations = minimum(Int.(group.iterations)),
                max_iterations = maximum(Int.(group.iterations)),
                median_f_evals = median(Float64.(group.f_evals)),
                median_natural_residual = median(Float64.(group.natural_residual)),
                min_natural_residual = minimum(Float64.(group.natural_residual)),
                max_natural_residual = maximum(Float64.(group.natural_residual)),
                median_duality_gap = median(Float64.(group.duality_gap)),
                min_duality_gap = minimum(Float64.(group.duality_gap)),
                max_duality_gap = maximum(Float64.(group.duality_gap)),
            ))
        end
        CSV.write(joinpath(outdir, "duplicated_identity_cap_1e5_summary.csv"),
                  DataFrame(summary_rows))
        return true
    finally
        SQLite.close(db)
    end
end

function write_expensive_b_reports(db_path, outdir)
    db = open_db(db_path)
    try
        source = DBInterface.execute(db, """
            SELECT c.method, r.config_hash, r.problem, r.dimension,
                   r.init_point, r.seed_idx, r.flag, r.converged,
                   r.iterations, r.f_evals, r.cpu_time AS legacy_cpu_time,
                   r.native_residual
            FROM results r JOIN configs c ON c.config_hash=r.config_hash
            WHERE r.script='s30' AND r.production=1
              AND r.problem='double_integrator_control'
              AND r.dimension=10000 AND c.maxiter=4000
              AND c.hash_input LIKE '%protocol=oc_manuscript_v1%'
            ORDER BY c.method, r.seed_idx
        """) |> DataFrame
        if nrow(source) != 60
            println("BLOCKED expensive-B rows: expected 60, found $(nrow(source))")
            return false
        end
        timing = timing_v2_summary(db; protocol_hash = REPORT_009_TIMING_HASH)
        timing = timing[(timing.problem .== "double_integrator_control") .&
                        (timing.dimension .== 10000), :]
        select!(timing, :source_config_hash => :config_hash, :problem, :dimension,
                :init_point, :median_ms => :timing_ms,
                :q1_ms => :timing_q1_ms, :q3_ms => :timing_q3_ms,
                :min_ms => :timing_min_ms, :max_ms => :timing_max_ms,
                :repetitions => :timing_repetitions)
        joined = leftjoin(source, timing;
                          on = [:config_hash, :problem, :dimension, :init_point])
        any(ismissing.(joined.timing_ms)) &&
            error("Missing timing-v2 rows for expensive-B results")
        CSV.write(joinpath(outdir, "double_integrator_K10000_per_start.csv"), joined)
        return true
    finally
        SQLite.close(db)
    end
end

function write_supplementary_repetition_gate(families, outdir)
    rows = DataFrame[]
    for (db_family, db_path) in families
        db = open_db(db_path)
        try
            ensure_supplementary_009_tables!(db)
            gate = DBInterface.execute(db, """
                WITH per_start AS (
                    SELECT experiment, family, method, problem, dimension,
                           init_point, seed_idx, run_id,
                           COUNT(*) AS repetitions,
                           MIN(repetition) AS first_repetition,
                           MAX(repetition) AS last_repetition
                    FROM supplementary_009_repetitions
                    GROUP BY experiment, family, method, problem, dimension,
                             init_point, seed_idx, run_id
                )
                SELECT ? AS database_family, experiment, family, method,
                       problem, dimension,
                       COUNT(*) AS starts,
                       COUNT(DISTINCT seed_idx) AS distinct_seeds,
                       SUM(repetitions) AS retained_repetitions,
                       MIN(repetitions) AS minimum_repetitions_per_start,
                       MAX(repetitions) AS maximum_repetitions_per_start,
                       MIN(first_repetition) AS minimum_repetition_index,
                       MAX(last_repetition) AS maximum_repetition_index,
                       MIN(seed_idx) AS minimum_seed_index,
                       MAX(seed_idx) AS maximum_seed_index
                FROM per_start
                GROUP BY experiment, family, method, problem, dimension
                ORDER BY experiment, problem, dimension, method
            """, (db_family,)) |> DataFrame
            nrow(gate) > 0 && push!(rows, gate)
        finally
            SQLite.close(db)
        end
    end
    isempty(rows) && error("No supplementary-009 repetitions found")
    CSV.write(joinpath(outdir, "supplementary_repetitions_gate.csv"), vcat(rows...))
    return true
end

function supplementary_009_report_main()
    configure_reproducible_runtime!()
    mkpath(REPORT_009_ROOT)
    families = [
        ("compressed_sensing", joinpath(JCODE_ROOT, "results", "compressed_sensing", "experiments.db")),
        ("double_integrator", joinpath(JCODE_ROOT, "results", "optimal_control", "double_integrator_control", "experiments.db")),
        ("harmonic_oscillator", joinpath(JCODE_ROOT, "results", "optimal_control", "harmonic_oscillator", "experiments.db")),
        ("saddle_point", joinpath(JCODE_ROOT, "results", "saddle_point", "experiments.db")),
    ]
    println("timing_protocol_hash=$REPORT_009_TIMING_HASH")
    for (family, db_path) in families
        write_family_timing_reports(db_path, family, REPORT_009_ROOT)
    end
    game_db = last(families)[2]
    di_db = families[2][2]
    write_extended_game_reports(game_db, REPORT_009_ROOT)
    write_expensive_b_reports(di_db, REPORT_009_ROOT)
    write_supplementary_repetition_gate(families, REPORT_009_ROOT)
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    supplementary_009_report_main()
end
