# External, monitoring-free timing protocol used by the 009 supplementary pass.

const TIMING_V2_VERSION = "2.0.0"

timing_signature(result::SolverResult) =
    (result.flag, result.iterations, result.f_evals)

Base.@kwdef struct TimingV2Batch
    repetition::Int
    batch_size::Int
    total_ns::Int
    ms_per_solve::Float64
    signature::Tuple{Symbol,Int,Int}
end

Base.@kwdef struct TimingV2Repetition
    protocol_hash::String
    family::String
    method::String
    source_config_hash::String
    source_script::String = "s30"
    problem::String
    dimension::Int
    init_point::String
    seed_idx::Int
    production::Bool = true
    repetition::Int
    batch_size::Int
    total_ns::Int
    ms_per_solve::Float64
    expected_flag::Symbol
    expected_iterations::Int
    expected_f_evals::Int
    run_id::String
    created_at::String = Dates.format(Dates.now(), dateformat"yyyy-mm-ddTHH:MM:SS.sss")
end

function timing_v2_protocol_hash(; warmups::Int, repetitions::Int,
                                 min_batch_seconds::Float64)
    warmups >= 1 || throw(ArgumentError("warmups must be at least 1"))
    repetitions >= 3 || throw(ArgumentError("repetitions must be at least 3"))
    min_batch_seconds > 0 || throw(ArgumentError("min_batch_seconds must be positive"))
    input = string(
        "timing_v2|version=", TIMING_V2_VERSION,
        "|clock=time_ns|warmups=", warmups,
        "|repetitions=", repetitions,
        "|min_batch_seconds=", repr(min_batch_seconds),
        "|monitor_residual=false|record_elapsed=false|observers=none",
    )
    return bytes2hex(sha256(input))[1:12]
end

function _checked_timing_solve(solve_once, expected::Tuple{Symbol,Int,Int})
    result = solve_once()
    actual = timing_signature(result)
    actual == expected || error(
        "Timing-v2 signature mismatch: expected $(expected), got $(actual)",
    )
    return result
end

function _timed_checked_batch(solve_once, expected::Tuple{Symbol,Int,Int},
                              batch_size::Int)
    batch_size >= 1 || throw(ArgumentError("batch_size must be positive"))
    start_ns = time_ns()
    for _ in 1:batch_size
        _checked_timing_solve(solve_once, expected)
    end
    return Int(time_ns() - start_ns)
end

function _grow_timing_batch(batch_size::Int, elapsed_ns::Int, target_ns::Int)
    elapsed_ns > 0 || return max(2, 2batch_size)
    estimate = ceil(Int, 1.10 * batch_size * target_ns / elapsed_ns)
    return max(batch_size + 1, estimate)
end

"""
    run_timing_v2(solve_once, expected; warmups=2, repetitions=3,
                  min_batch_seconds=0.1)

Warm the supplied deterministic solve closure, calibrate a batch, then retain
externally measured repetitions. Every solve is checked against the expected
`(flag, iterations, f_evals)` signature. The closure is responsible for
creating fresh mutable stopping callbacks on every call.
"""
function run_timing_v2(solve_once, expected::Tuple{Symbol,Int,Int};
                       warmups::Int = 2,
                       repetitions::Int = 3,
                       min_batch_seconds::Float64 = 0.1)
    warmups >= 1 || throw(ArgumentError("warmups must be at least 1"))
    repetitions >= 1 || throw(ArgumentError("repetitions must be positive"))
    min_batch_seconds > 0 || throw(ArgumentError("min_batch_seconds must be positive"))

    for _ in 1:warmups
        _checked_timing_solve(solve_once, expected)
    end

    target_ns = ceil(Int, min_batch_seconds * 1.0e9)
    batch_size = 1
    while true
        elapsed_ns = _timed_checked_batch(solve_once, expected, batch_size)
        elapsed_ns >= target_ns && break
        batch_size = _grow_timing_batch(batch_size, elapsed_ns, target_ns)
    end

    rows = TimingV2Batch[]
    for repetition in 1:repetitions
        while true
            elapsed_ns = _timed_checked_batch(solve_once, expected, batch_size)
            if elapsed_ns >= target_ns
                push!(rows, TimingV2Batch(
                    repetition = repetition,
                    batch_size = batch_size,
                    total_ns = elapsed_ns,
                    ms_per_solve = elapsed_ns / batch_size / 1.0e6,
                    signature = expected,
                ))
                break
            end
            batch_size = _grow_timing_batch(batch_size, elapsed_ns, target_ns)
        end
    end
    return rows
end

function ensure_timing_v2_table!(db)
    DBInterface.execute(db, """
        CREATE TABLE IF NOT EXISTS timing_v2_repetitions (
            protocol_hash       TEXT NOT NULL,
            family              TEXT NOT NULL,
            method              TEXT NOT NULL,
            source_config_hash  TEXT NOT NULL,
            source_script       TEXT NOT NULL,
            problem             TEXT NOT NULL,
            dimension           INTEGER NOT NULL,
            init_point          TEXT NOT NULL,
            seed_idx            INTEGER NOT NULL,
            production          INTEGER NOT NULL,
            repetition          INTEGER NOT NULL,
            batch_size          INTEGER NOT NULL,
            total_ns            INTEGER NOT NULL,
            ms_per_solve        REAL NOT NULL,
            expected_flag       TEXT NOT NULL,
            expected_iterations INTEGER NOT NULL,
            expected_f_evals    INTEGER NOT NULL,
            run_id              TEXT NOT NULL,
            created_at          TEXT NOT NULL,
            PRIMARY KEY (
                protocol_hash, source_config_hash, source_script,
                problem, dimension, init_point, production, repetition
            )
        )
    """)
    return nothing
end

function insert_timing_v2_repetition!(db, row::TimingV2Repetition)
    row.repetition >= 1 || throw(ArgumentError("repetition must be positive"))
    row.batch_size >= 1 || throw(ArgumentError("batch_size must be positive"))
    row.total_ns > 0 || throw(ArgumentError("total_ns must be positive"))
    row.ms_per_solve > 0 || throw(ArgumentError("ms_per_solve must be positive"))
    DBInterface.execute(db, """
        INSERT OR IGNORE INTO timing_v2_repetitions (
            protocol_hash, family, method, source_config_hash, source_script,
            problem, dimension, init_point, seed_idx, production, repetition,
            batch_size, total_ns, ms_per_solve, expected_flag,
            expected_iterations, expected_f_evals, run_id, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        row.protocol_hash, row.family, row.method, row.source_config_hash,
        row.source_script, row.problem, row.dimension, row.init_point,
        row.seed_idx, row.production ? 1 : 0, row.repetition, row.batch_size,
        row.total_ns, row.ms_per_solve, string(row.expected_flag),
        row.expected_iterations, row.expected_f_evals, row.run_id,
        row.created_at,
    ))
    return nothing
end

function timing_v2_done(db, protocol_hash::String, source_config_hash::String,
                        problem::String, dimension::Int, init_point::String,
                        production::Bool; repetitions::Int = 3,
                        source_script::String = "s30")
    rows = DBInterface.execute(db, """
        SELECT COUNT(DISTINCT repetition) AS n
        FROM timing_v2_repetitions
        WHERE protocol_hash=? AND source_config_hash=? AND source_script=?
          AND problem=? AND dimension=? AND init_point=? AND production=?
    """, (protocol_hash, source_config_hash, source_script, problem,
           dimension, init_point, production ? 1 : 0)) |> DataFrame
    return nrow(rows) == 1 && Int(rows.n[1]) >= repetitions
end

function timing_v2_summary(db; protocol_hash::Union{Nothing,String} = nothing)
    df = if protocol_hash === nothing
        DBInterface.execute(db, "SELECT * FROM timing_v2_repetitions") |> DataFrame
    else
        DBInterface.execute(
            db,
            "SELECT * FROM timing_v2_repetitions WHERE protocol_hash=?",
            (protocol_hash,),
        ) |> DataFrame
    end
    nrow(df) == 0 && return DataFrame()

    keys = [:protocol_hash, :family, :method, :source_config_hash,
            :source_script, :problem, :dimension, :init_point, :seed_idx,
            :production, :expected_flag, :expected_iterations,
            :expected_f_evals]
    rows = NamedTuple[]
    for group in groupby(df, keys)
        values = sort(Float64.(group.ms_per_solve))
        push!(rows, (
            protocol_hash = String(group.protocol_hash[1]),
            family = String(group.family[1]),
            method = String(group.method[1]),
            source_config_hash = String(group.source_config_hash[1]),
            source_script = String(group.source_script[1]),
            problem = String(group.problem[1]),
            dimension = Int(group.dimension[1]),
            init_point = String(group.init_point[1]),
            seed_idx = Int(group.seed_idx[1]),
            production = Int(group.production[1]),
            expected_flag = String(group.expected_flag[1]),
            expected_iterations = Int(group.expected_iterations[1]),
            expected_f_evals = Int(group.expected_f_evals[1]),
            repetitions = length(values),
            min_ms = minimum(values),
            q1_ms = quantile(values, 0.25),
            median_ms = median(values),
            q3_ms = quantile(values, 0.75),
            max_ms = maximum(values),
        ))
    end
    return DataFrame(rows)
end

"""
    attach_timing_v2(db, source_rows; warmups=2, repetitions=3,
                     min_batch_seconds=0.1)

Attach one externally timed median (milliseconds per solve) to every definitive
source row. Missing, duplicated, or under-replicated timing rows are fatal: a
publication table must never silently fall back to the legacy in-solver clock.
"""
function attach_timing_v2(db, source_rows::DataFrame;
                          warmups::Int = 2,
                          repetitions::Int = 3,
                          min_batch_seconds::Float64 = 0.1,
                          protocol_hash::Union{Nothing,String} = nothing)
    required = [:config_hash, :problem, :dimension, :init_point]
    missing_columns = setdiff(required, propertynames(source_rows))
    isempty(missing_columns) || error(
        "Cannot attach timing-v2 rows; source columns are missing: $(missing_columns)",
    )

    expected_protocol_hash = timing_v2_protocol_hash(
        warmups = warmups,
        repetitions = repetitions,
        min_batch_seconds = min_batch_seconds,
    )
    protocol_hash === nothing && (protocol_hash = expected_protocol_hash)
    protocol_hash == expected_protocol_hash || error(
        "Timing-v2 protocol hash $(protocol_hash) does not match the supplied settings $(expected_protocol_hash).",
    )
    timing = timing_v2_summary(db; protocol_hash = protocol_hash)
    nrow(timing) > 0 || error(
        "No timing-v2 rows found for production protocol $(protocol_hash). " *
        "Run the family s35_timing_v2.jl script first.",
    )
    timing = timing[timing.production .== 1, :]
    all(timing.repetitions .>= repetitions) || error(
        "Timing-v2 protocol $(protocol_hash) contains under-replicated starts.",
    )
    select!(timing, :source_config_hash => :config_hash, :problem, :dimension,
            :init_point, :median_ms => :timing_ms, :q1_ms => :timing_q1_ms,
            :q3_ms => :timing_q3_ms, :repetitions => :timing_repetitions)
    unique_timing = unique(timing, required)
    nrow(timing) == nrow(unique_timing) || error(
        "Timing-v2 protocol $(protocol_hash) has duplicate source keys.",
    )
    timing = unique_timing

    joined = leftjoin(source_rows, timing; on = required)
    missing_mask = ismissing.(joined.timing_ms)
    if any(missing_mask)
        sample = first(joined[missing_mask, required], min(5, count(missing_mask)))
        error("Missing production timing-v2 rows for $(count(missing_mask)) source rows; sample=$(sample)")
    end
    return joined
end
