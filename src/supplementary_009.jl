# Shared persistence helpers for the production supplementary runs commissioned
# in channels/claude_to_codex/009_supplementary-runs.md.

const SUPPLEMENTARY_009_REPETITIONS_SQL = """
CREATE TABLE IF NOT EXISTS supplementary_009_repetitions (
    experiment       TEXT NOT NULL,
    config_hash      TEXT NOT NULL,
    family           TEXT NOT NULL,
    method           TEXT NOT NULL,
    problem          TEXT NOT NULL,
    dimension        INTEGER NOT NULL,
    init_point       TEXT NOT NULL,
    seed_idx         INTEGER NOT NULL,
    repetition       INTEGER NOT NULL,
    flag             TEXT NOT NULL,
    iterations       INTEGER NOT NULL,
    f_evals          INTEGER NOT NULL,
    cpu_time         REAL NOT NULL,
    run_id           TEXT NOT NULL,
    created_at       TEXT NOT NULL,
    PRIMARY KEY (
        experiment, config_hash, problem, dimension, init_point, repetition
    )
)
"""

ensure_supplementary_009_tables!(db) =
    (DBInterface.execute(db, SUPPLEMENTARY_009_REPETITIONS_SQL); nothing)

supplementary_signature(result::SolverResult) =
    (result.flag, result.iterations, result.f_evals)

function supplementary_median_result(results::AbstractVector{<:SolverResult})
    isempty(results) && error("Cannot aggregate an empty supplementary repetition set.")
    signature = supplementary_signature(first(results))
    all(supplementary_signature(result) == signature for result in results) ||
        error("Supplementary repetition signature mismatch: expected $(signature), " *
              "got $([supplementary_signature(result) for result in results])")
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

function retain_supplementary_repetitions!(db, results;
        experiment::String, config_hash::String, family::String,
        method::String, problem::String, dimension::Int, init_point::String,
        seed_idx::Int, run_id::String)
    ensure_supplementary_009_tables!(db)
    for (repetition, result) in enumerate(results)
        DBInterface.execute(db, """
            INSERT OR REPLACE INTO supplementary_009_repetitions
            (experiment, config_hash, family, method, problem, dimension,
             init_point, seed_idx, repetition, flag, iterations, f_evals,
             cpu_time, run_id, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (experiment, config_hash, family, method, problem, dimension,
               init_point, seed_idx, repetition, string(result.flag),
               result.iterations, result.f_evals, result.cpu_time, run_id,
               Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS")))
    end
    return nothing
end
