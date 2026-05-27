# benchmark.jl — Database infrastructure for experiment management
#
# Provides: SQLite DB setup, content-addressable config hashing, CRUD
# operations, CSV export, and summary printing. Shared across all scripts.
#
# DB file: results/experiments.db (single file for ALL experiments)
# Storage: SQLite with WAL mode for concurrent read safety.
#
# Usage:
#   db = open_db()                                  # open or create DB
#   hash, input = make_config_hash("EPCM", "1.0.0", params, eps, maxiter)
#   ensure_config!(db, hash, "EPCM", "1.0.0", params, eps, maxiter, input)
#   is_done(db, hash, "P1", 50, "rand_seed42")      # check if result exists
#   insert_result!(db, hash, "P1", 50, "rand_seed42", run_id, result)
#   insert_history!(db, hash, "P1", 50, "rand_seed42", result.history)
#   export_results_csv(db, "results/export.csv")
#   print_summary(db, tee, ["EPCM", "MTTM", "IMTTM", "IFRAB", "SFRBM"])

# Dependencies are loaded via deps.jl (included before this file).

# ============================================================================
# Constants
# ============================================================================

const DB_PATH = joinpath(JCODE_ROOT, "results", "experiments.db")

# ============================================================================
# Database Setup
# ============================================================================

"""
    open_db(path=DB_PATH) -> SQLite.DB

Open (or create) the experiments database. Creates tables if they don't exist.
Enables WAL mode for safe concurrent reads.
"""
function open_db(path::String=DB_PATH)
    mkpath(dirname(path))
    db = SQLite.DB(path)
    DBInterface.execute(db, "PRAGMA journal_mode=WAL")
    _create_tables!(db)
    return db
end

function _create_tables!(db)
    # Algorithm configuration identity
    DBInterface.execute(db, """
        CREATE TABLE IF NOT EXISTS configs (
            config_hash TEXT PRIMARY KEY,
            method      TEXT NOT NULL,
            version     TEXT NOT NULL,
            params_json TEXT NOT NULL,
            eps         REAL NOT NULL,
            maxiter     INTEGER NOT NULL,
            hash_input  TEXT NOT NULL,
            created_at  TEXT NOT NULL
        )
    """)

    # One row per solver call
    DBInterface.execute(db, """
        CREATE TABLE IF NOT EXISTS results (
            config_hash TEXT NOT NULL,
            problem     TEXT NOT NULL,
            dimension   INTEGER NOT NULL,
            init_point  TEXT NOT NULL,
            run_id      TEXT NOT NULL,
            converged   INTEGER NOT NULL,
            iterations  INTEGER,
            f_evals     INTEGER,
            cpu_time    REAL,
            flag        TEXT,
            residual    REAL,
            created_at  TEXT NOT NULL,
            PRIMARY KEY (config_hash, problem, dimension, init_point),
            FOREIGN KEY (config_hash) REFERENCES configs(config_hash)
        )
    """)

    # Per-iteration history (only for tracked runs)
    DBInterface.execute(db, """
        CREATE TABLE IF NOT EXISTS history (
            config_hash TEXT NOT NULL,
            problem     TEXT NOT NULL,
            dimension   INTEGER NOT NULL,
            init_point  TEXT NOT NULL,
            k           INTEGER NOT NULL,
            f_evals     INTEGER,
            elapsed     REAL,
            residual    REAL,
            step_size   REAL,
            PRIMARY KEY (config_hash, problem, dimension, init_point, k),
            FOREIGN KEY (config_hash, problem, dimension, init_point)
                REFERENCES results(config_hash, problem, dimension, init_point)
        )
    """)
end

# ============================================================================
# Config Hashing
# ============================================================================

"""
    make_config_hash(method, version, params, eps, maxiter)
    -> (hash::String, hash_input::String)

Content-addressable experiment ID. The SAME NamedTuple `params` must be
both hashed here AND splatted to the solver — zero divergence possible.

Input format: "METHOD|vVERSION|param1=val1|param2=val2|...|eps=X|maxiter=Y"
Output: first 12 hex chars of SHA-256.
"""
function make_config_hash(method::String, version::String, params::NamedTuple,
                          eps::Float64, maxiter::Int)
    parts = sort(["$k=$v" for (k, v) in pairs(params)])
    input = "$method|v$version|" * join(parts, "|") * "|eps=$eps|maxiter=$maxiter"
    hash = bytes2hex(sha256(input))[1:12]
    return (hash, input)
end

# ============================================================================
# Config Registry
# ============================================================================

"""
    ensure_config!(db, hash, method, version, params, eps, maxiter, hash_input)

Insert config into the `configs` table if it doesn't already exist.
"""
function ensure_config!(db, hash::String, method::String, version::String,
                        params::NamedTuple, eps::Float64, maxiter::Int,
                        hash_input::String)
    existing = DBInterface.execute(db,
        "SELECT 1 FROM configs WHERE config_hash = ?", (hash,)) |> DataFrame
    nrow(existing) > 0 && return

    params_json = JSON3.write(Dict(pairs(params)))
    DBInterface.execute(db, """
        INSERT INTO configs (config_hash, method, version, params_json, eps, maxiter, hash_input, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (hash, method, version, params_json, eps, maxiter, hash_input,
          Dates.format(now(), "yyyy-mm-dd HH:MM:SS")))
end

# ============================================================================
# Result CRUD
# ============================================================================

"""
    is_done(db, config_hash, problem, dimension, init_point) -> Bool

Check if a result row exists for this (config, problem, dim, init) combination.
"""
function is_done(db, hash::String, problem::String, dim::Int, init::String)
    r = DBInterface.execute(db,
        "SELECT 1 FROM results WHERE config_hash=? AND problem=? AND dimension=? AND init_point=?",
        (hash, problem, dim, init)) |> DataFrame
    return nrow(r) > 0
end

"""
    insert_result!(db, config_hash, problem, dimension, init_point, run_id, result)

Insert or replace a result row. `result` must be a `SolverResult` (or have
the same field names).
"""
function insert_result!(db, hash::String, problem::String, dim::Int,
                        init::String, run_id::String, result)
    DBInterface.execute(db, """
        INSERT OR REPLACE INTO results
            (config_hash, problem, dimension, init_point, run_id,
             converged, iterations, f_evals, cpu_time, flag, residual, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (hash, problem, dim, init, run_id,
          result.converged ? 1 : 0, result.iterations, result.f_evals,
          result.cpu_time, string(result.flag), result.residual,
          Dates.format(now(), "yyyy-mm-dd HH:MM:SS")))
end

"""
    insert_history!(db, config_hash, problem, dimension, init_point, history)

Bulk-insert per-iteration history records. Deletes existing history for this
combination first (safe for re-runs with --force).
"""
function insert_history!(db, hash::String, problem::String, dim::Int,
                         init::String, history::Vector{IterRecord})
    isempty(history) && return

    # Delete existing history for this combination
    DBInterface.execute(db, """
        DELETE FROM history
        WHERE config_hash=? AND problem=? AND dimension=? AND init_point=?
    """, (hash, problem, dim, init))

    # Bulk insert
    for h in history
        DBInterface.execute(db, """
            INSERT INTO history
                (config_hash, problem, dimension, init_point, k, f_evals, elapsed, residual, step_size)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (hash, problem, dim, init, h.k, h.f_evals, h.elapsed, h.residual, h.step_size))
    end
end

# ============================================================================
# Export & Summary
# ============================================================================

"""
    export_results_csv(db, path) -> Int

Export all results (joined with configs) to a CSV file. Returns row count.
"""
function export_results_csv(db, path::String)
    df = DBInterface.execute(db, """
        SELECT c.method, c.version, c.params_json, c.eps, c.maxiter,
               r.problem, r.dimension, r.init_point,
               r.converged, r.iterations, r.f_evals, r.cpu_time, r.flag, r.residual,
               r.run_id, r.created_at
        FROM results r
        JOIN configs c ON r.config_hash = c.config_hash
        ORDER BY c.method, r.problem, r.dimension, r.init_point
    """) |> DataFrame
    CSV.write(path, df)
    return nrow(df)
end

"""
    print_summary(db, tee, method_order)

Print aggregate statistics per method to the TeeIO stream.
"""
function print_summary(db, tee, method_order::Vector{String})
    println(tee, "\n" * "=" ^ 70)
    println(tee, "  Summary")
    println(tee, "=" ^ 70)

    @printf(tee, "\n  %-12s  %6s  %6s  %7s  %8s  %8s  %8s\n",
            "Method", "Total", "Conv", "Rate%", "Med_IT", "Med_FE", "Med_CPU")
    println(tee, "  " * "-" ^ 62)

    for m in method_order
        df = DBInterface.execute(db, """
            SELECT r.converged, r.iterations, r.f_evals, r.cpu_time
            FROM results r JOIN configs c ON r.config_hash = c.config_hash
            WHERE c.method = ?
        """, (m,)) |> DataFrame

        nrow(df) == 0 && continue
        n_total = nrow(df)
        conv = filter(r -> r.converged == 1, df)
        n_conv = nrow(conv)
        rate = 100.0 * n_conv / n_total
        med_it  = n_conv > 0 ? median(conv.iterations) : NaN
        med_fe  = n_conv > 0 ? median(conv.f_evals) : NaN
        med_cpu = n_conv > 0 ? median(conv.cpu_time) : NaN

        @printf(tee, "  %-12s  %6d  %6d  %6.1f%%  %8.0f  %8.0f  %8.4f\n",
                m, n_total, n_conv, rate, med_it, med_fe, med_cpu)
    end

    println(tee, "  " * "-" ^ 62)
    println(tee, "  DB: $(DB_PATH)")
end
