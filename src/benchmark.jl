# benchmark.jl — Database infrastructure for experiment management.
#
# Provides: SQLite DB setup, content-addressable config hashing (dispatched
# on AbstractAlgorithm), CRUD operations, CSV export, and summary printing.
# Shared across all scripts (s01, s10, s20, s28, s30, s70).
#
# DB file: results/experiments.db (single file for ALL experiments — s10/s20
# tuning runs, s28 baseline-validation, s30 benchmark; the `script TEXT`
# discriminator column on results/history lets each script's rows coexist).
#
# Storage: SQLite with WAL mode for concurrent read safety.
#
# Usage:
#   db = open_db()                                                      # open or create
#   alg = EPCM(:P1)                                                     # algorithm instance
#   hash, input = make_config_hash(alg, "P1", 1e-6, 3000)               # SHA-256 prefix
#   ensure_config!(db, alg, "P1", 1e-6, 3000, hash, input)              # idempotent insert
#   is_done(db, hash, "P1", 5, "seed1"; script="s30")                   # skip check
#   insert_result!(db, hash, "P1", 5, "seed1", 1, run_id, result;       # write row
#                  script="s30", native_residual=Tol_n)
#   insert_history!(db, hash, "P1", 5, "seed1", 1, result.history;      # write per-iter
#                   script="s30")
#   export_results_csv(db, "results/export.csv")                        # dump CSV
#   print_summary(db, tee, ["EPCM", "MTTM", "IMTTM", "IFRAB", "SFRBM"]) # aggregate
#
# Dependencies are loaded via deps.jl (included before this file).
# Requires AbstractAlgorithm + name/version traits from algorithm_types.jl.

# ============================================================================
# Constants
# ============================================================================

const DB_PATH = joinpath(JCODE_ROOT, "results", "experiments.db")

"""
Recognised values for the `flag` column on the `results` table — the symbol
set produced by stopping callbacks plus `:error` (outer benchmark-wrapper
catch). See `callbacks.jl` for the source-of-truth list.
"""
const RECOGNISED_FLAGS = Set([:converged, :maxiter, :nan, :stagnation, :error])

# ============================================================================
# Iteration-budget (N_max) policy — tolerance-aware
# ============================================================================
#
# The universal stopping rule is R_n < eps; N_max is the shared safety/budget
# cap (uniform across methods for a given (problem, eps) — fairness). The cap
# is calibrated to the manuscript values at the reference tolerance eps = 1e-6
# (N_max = 3000 for P1/P2, 1000 for P3; script_port_plan.md Q2) and scales as
# log10(1/eps): under strong monotonicity (P1, P2) these methods converge
# linearly, so iterations-to-eps ∝ log(1/eps), making log-scaling the right
# law. P3 (rank-1 M, only monotone) may be sublinear — watch s10 for systematic
# capping there. Used by s10/s20/s28/s30 so the budget is consistent.

"Reference tolerance at which `NMAX_REF` is calibrated."
const EPS_REF = 1.0e-6

"Iteration-budget cap per problem at `EPS_REF` (suite revised 2026-05-28)."
const NMAX_REF = Dict("P1" => 3000, "P2" => 3000, "P3" => 1000, "P4" => 3000)

"""
Native (source-paper) stopping thresholds per problem — the PRIMARY convergence
criterion (D1 decision 2026-05-28, supersedes the universal R_n<1e-6 rule):
  - P1 (Volterra SFP, Tan2022a):  E_n = ‖(I−P_C)x‖²+‖B(x)‖² < 1e-5
  - P2 (ℓ1+quad, Yao2024):        ‖x_{k+1}−x_k‖ < 1e-6
  - P3 (control, Izuchukwu2023):  Tol_n = 0.5‖z−clip(z−Bz)‖² < 1e-6
  - P4 (LASSO, Tan2022a):         ‖x_{k+1}−x_k‖ < 1e-6
Each problem's `native_residual` closure computes the corresponding quantity;
scripts stop on `NativeResStopping(prob.native_residual, NATIVE_TOL[prob])` and
still RECORD R_n/R̃_n as secondary columns.
"""
const NATIVE_TOL = Dict("P1" => 1.0e-5, "P2" => 1.0e-6, "P3" => 1.0e-6, "P4" => 1.0e-6)

"""
    native_tol(problem::AbstractString) -> Float64

Native (source-paper) stopping threshold for `problem` (see `NATIVE_TOL`).
"""
function native_tol(problem::AbstractString)
    haskey(NATIVE_TOL, problem) ||
        throw(ArgumentError("native_tol: unknown problem '$problem'; known: $(join(sort(collect(keys(NATIVE_TOL))), ", "))"))
    return NATIVE_TOL[problem]
end

"""
    universal_maxiter(problem::AbstractString, eps::Float64) -> Int

Tolerance-aware iteration budget: `round(NMAX_REF[problem] · log10(1/eps)/log10(1/EPS_REF))`.
Reproduces the manuscript N_max exactly at `eps = EPS_REF = 1e-6` (3000 for
P1/P2, 1000 for P3) and scales ∝ log10(1/eps) for other tolerances. Uniform
across all methods for a given (problem, eps).
"""
function universal_maxiter(problem::AbstractString, eps::Float64)
    haskey(NMAX_REF, problem) ||
        throw(ArgumentError("universal_maxiter: unknown problem '$problem'; known: $(join(sort(collect(keys(NMAX_REF))), ", "))"))
    eps > 0 || throw(ArgumentError("universal_maxiter: eps must be > 0, got $eps"))
    return round(Int, NMAX_REF[problem] * (log10(1 / eps) / log10(1 / EPS_REF)))
end

# ============================================================================
# Database setup
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
    # ─── configs: one row per (algorithm, eps, maxiter) — script-agnostic ───
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

    # ─── results: one row per (script, config, problem, dim, init) ──────────
    DBInterface.execute(db, """
        CREATE TABLE IF NOT EXISTS results (
            script           TEXT NOT NULL DEFAULT 's30',
            config_hash      TEXT NOT NULL,
            problem          TEXT NOT NULL,
            dimension        INTEGER NOT NULL,
            init_point       TEXT NOT NULL,
            seed_idx         INTEGER NOT NULL,
            run_id           TEXT NOT NULL,
            converged        INTEGER NOT NULL,
            iterations       INTEGER,
            f_evals          INTEGER,
            cpu_time         REAL,
            flag             TEXT,
            residual         REAL,
            scaled_residual  REAL,
            native_residual  REAL,
            created_at       TEXT NOT NULL,
            PRIMARY KEY (script, config_hash, problem, dimension, init_point),
            FOREIGN KEY (config_hash) REFERENCES configs(config_hash)
        )
    """)

    # ─── history: per-iteration data, only for runs with HistoryCallback ────
    DBInterface.execute(db, """
        CREATE TABLE IF NOT EXISTS history (
            script           TEXT NOT NULL DEFAULT 's30',
            config_hash      TEXT NOT NULL,
            problem          TEXT NOT NULL,
            dimension        INTEGER NOT NULL,
            init_point       TEXT NOT NULL,
            seed_idx         INTEGER NOT NULL,
            k                INTEGER NOT NULL,
            f_evals          INTEGER,
            elapsed          REAL,
            residual         REAL,
            scaled_residual  REAL,
            step_size        REAL,
            PRIMARY KEY (script, config_hash, problem, dimension, init_point, k),
            FOREIGN KEY (script, config_hash, problem, dimension, init_point)
                REFERENCES results(script, config_hash, problem, dimension, init_point)
        )
    """)
end

# ============================================================================
# Config hashing — dispatched on AbstractAlgorithm
# ============================================================================

"""
    make_config_hash(alg::AbstractAlgorithm, prob_id::AbstractString,
                     eps::Float64, maxiter::Int) -> (hash::String, hash_input::String)

Content-addressable experiment ID. Pulls `name(typeof(alg))`, `version(typeof(alg))`,
and per-field values via `fieldnames(typeof(alg))` + `getfield`. Field names are
sorted before stringification so the hash is invariant to future struct-field
reorderings.

Each field value is stringified via `repr(value)` — round-trip-stable for
`Float64`, distinguishes `:foo` from `"foo"` for Symbol tags, and stable across
Julia versions (unlike `string(::Function)` which would leak the gensym counter
into the hash — algorithm structs are forbidden from holding `Function` fields
exactly to keep `repr` reproducible).

Hash input format:
  "METHOD|vMAJOR.MINOR.PATCH|field1=repr(val1)|...|prob=PROB_ID|eps=repr(EPS)|maxiter=MAXITER"

Note: `eps` is stringified via `repr` (round-trip-stable for `Float64`);
`maxiter::Int` is rendered via standard interpolation since integer
`string` and `repr` agree bit-for-bit.

Output: first 12 hex chars of SHA-256 (≈ 2.8 × 10¹⁴ states; collision risk
negligible at our run counts; `ensure_config!` adds a hash_input-mismatch
assertion as a silent-collision guard).
"""
function make_config_hash(alg::AbstractAlgorithm, prob_id::AbstractString,
                          eps::Float64, maxiter::Int)
    T = typeof(alg)

    # Canonical-order field stringification.
    fnames = sort(collect(fieldnames(T)))
    parts = String[]
    for f in fnames
        val = getfield(alg, f)
        val isa Function && throw(ArgumentError(
            "make_config_hash: algorithm $(name(T)) has Function-valued field :$f; " *
            "encode rule-selectors as Symbol tags instead so the hash is reproducible"))
        push!(parts, "$f=$(repr(val))")
    end

    input = string(
        name(T), "|",
        "v", version(T), "|",
        join(parts, "|"), "|",
        "prob=", prob_id, "|",
        "eps=", repr(eps), "|",
        "maxiter=", maxiter,
    )
    hash = bytes2hex(sha256(input))[1:12]
    return (hash, input)
end

# ============================================================================
# Config registry
# ============================================================================

"""
    ensure_config!(db, alg::AbstractAlgorithm, prob_id, eps, maxiter,
                   hash::String, hash_input::String)

Insert config into the `configs` table if it doesn't already exist.

**Silent-collision guard**: if `config_hash` already exists in the table,
asserts that the stored `hash_input` equals the new `hash_input`. If they
differ, raises an error with both strings (this would mean two distinct
configs computed the same 12-char hash — astronomically unlikely with
SHA-256 prefix, but worth catching loudly if it ever happens).

`params_json` stores a `JSON3`-encoded dict mapping each field name (as
`String`) to `repr(value)` (as `String`) — the same stringification used
in the hash input, so the DB record is deterministically reconstructible.
"""
function ensure_config!(db, alg::AbstractAlgorithm, prob_id::AbstractString,
                        eps::Float64, maxiter::Int,
                        hash::String, hash_input::String)
    existing = DBInterface.execute(db,
        "SELECT hash_input FROM configs WHERE config_hash = ?", (hash,)) |> DataFrame

    if nrow(existing) > 0
        stored = existing.hash_input[1]
        stored == hash_input || error(
            "config_hash collision detected for hash=$hash:\n" *
            "  stored hash_input: $stored\n" *
            "  new    hash_input: $hash_input\n" *
            "Two distinct configs hashed to the same 12-char SHA-256 prefix. " *
            "Consider widening the prefix in make_config_hash, or investigate " *
            "whether a stale row is in the DB.")
        return  # config already registered, identical
    end

    T = typeof(alg)
    # params_json is an AUDIT record (not used in any lookup), so the field
    # iteration order is left unsorted (matches `fieldnames(T)`'s declaration
    # order). The hash input in `make_config_hash` is independently sorted
    # for canonicality; the two are intentionally allowed to diverge.
    params_dict = Dict(string(f) => repr(getfield(alg, f)) for f in fieldnames(T))
    params_json = JSON3.write(params_dict)

    DBInterface.execute(db, """
        INSERT INTO configs (config_hash, method, version, params_json, eps, maxiter, hash_input, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (hash, name(T), string(version(T)), params_json, eps, maxiter, hash_input,
          Dates.format(now(), "yyyy-mm-dd HH:MM:SS")))
end

# ============================================================================
# Result CRUD
# ============================================================================

"""
    is_done(db, hash, problem, dim, init; script="s30") -> Bool

Check if a result row exists for this (script, config, problem, dim, init)
combination. The `script` keyword filters by the discriminator column (so
an s10 OAT run with the same config_hash does NOT count as a completed
s30 benchmark run, and vice versa).
"""
function is_done(db, hash::String, problem::AbstractString, dim::Int,
                 init::AbstractString; script::AbstractString="s30")
    r = DBInterface.execute(db,
        "SELECT 1 FROM results WHERE script=? AND config_hash=? AND problem=? AND dimension=? AND init_point=?",
        (script, hash, problem, dim, init)) |> DataFrame
    return nrow(r) > 0
end

"""
    insert_result!(db, hash, problem, dim, init, seed_idx, run_id, result;
                   script="s30", native_residual=NaN)

Insert (or replace) a result row. `result` must be a `SolverResult` (carries
`converged`, `iterations`, `f_evals`, `cpu_time`, `flag::Symbol`, `residual`,
`scaled_residual`). `native_residual` is computed externally via a
`NativeResRecorder` observer and passed as a kwarg.

The `flag::Symbol` is stored as a string; the recognised values are listed
in `RECOGNISED_FLAGS` (`:converged, :maxiter, :nan, :stagnation, :error`).
Non-recognised flags are stored as-is — no validation here, by design.
"""
function insert_result!(db, hash::String, problem::AbstractString, dim::Int,
                        init::AbstractString, seed_idx::Int, run_id::String,
                        result;
                        script::AbstractString="s30",
                        native_residual::Float64=NaN)
    DBInterface.execute(db, """
        INSERT OR REPLACE INTO results
            (script, config_hash, problem, dimension, init_point, seed_idx, run_id,
             converged, iterations, f_evals, cpu_time, flag,
             residual, scaled_residual, native_residual, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (script, hash, problem, dim, init, seed_idx, run_id,
          result.converged ? 1 : 0, result.iterations, result.f_evals,
          result.cpu_time, string(result.flag),
          result.residual, result.scaled_residual, native_residual,
          Dates.format(now(), "yyyy-mm-dd HH:MM:SS")))
end

"""
    insert_history!(db, hash, problem, dim, init, seed_idx, history; script="s30")

Bulk-insert per-iteration history records. Deletes existing history for this
(script, config, problem, dim, init) combination first (safe for re-runs
with `--force`).

`history` is a `Vector{IterRecord}`; each record carries `k, f_evals, elapsed,
residual, scaled_residual, step_size` — all six columns are written.

**Wrapped in an explicit SQL transaction.** SQLite auto-commits each
`INSERT` outside a transaction (one fsync per row); for the s30 sweep this
amounts to ~500k fsync calls and dominates wall-clock time. The transaction
defers fsync to a single commit at the end. Two-orders-of-magnitude
speedup expected over per-row auto-commit.
"""
function insert_history!(db, hash::String, problem::AbstractString, dim::Int,
                         init::AbstractString, seed_idx::Int,
                         history::Vector{IterRecord};
                         script::AbstractString="s30")
    isempty(history) && return

    DBInterface.execute(db, "BEGIN TRANSACTION")
    try
        DBInterface.execute(db, """
            DELETE FROM history
            WHERE script=? AND config_hash=? AND problem=? AND dimension=? AND init_point=?
        """, (script, hash, problem, dim, init))

        for h in history
            DBInterface.execute(db, """
                INSERT INTO history
                    (script, config_hash, problem, dimension, init_point, seed_idx,
                     k, f_evals, elapsed, residual, scaled_residual, step_size)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (script, hash, problem, dim, init, seed_idx,
                  h.k, h.f_evals, h.elapsed,
                  h.residual, h.scaled_residual, h.step_size))
        end

        DBInterface.execute(db, "COMMIT")
    catch e
        DBInterface.execute(db, "ROLLBACK")
        rethrow(e)
    end
end

# ============================================================================
# Export & summary
# ============================================================================

"""
    export_results_csv(db, path) -> Int

Export all results (joined with configs) to a CSV file. Includes all new
columns (script, seed_idx, scaled_residual, native_residual). Returns row count.
"""
function export_results_csv(db, path::String)
    df = DBInterface.execute(db, """
        SELECT r.script,
               c.method, c.version, c.params_json, c.eps, c.maxiter,
               r.problem, r.dimension, r.init_point, r.seed_idx,
               r.converged, r.iterations, r.f_evals, r.cpu_time, r.flag,
               r.residual, r.scaled_residual, r.native_residual,
               r.run_id, r.created_at
        FROM results r
        JOIN configs c ON r.config_hash = c.config_hash
        ORDER BY r.script, c.method, r.problem, r.dimension, r.seed_idx
    """) |> DataFrame
    CSV.write(path, df)
    return nrow(df)
end

"""
    print_summary(db, tee, method_order::Vector{String}; script="s30")

Print aggregate statistics per method to the `TeeIO` stream. `script` filters
to a single script's rows (default `"s30"` for the main benchmark). Pass
`script=""` to omit the filter and aggregate across ALL scripts — useful
for one-shot "how many runs total per method?" reports; usually not what
you want for paper-table generation (use the script-filtered default).
"""
function print_summary(db, tee, method_order::Vector{String};
                       script::AbstractString="s30")
    all_scripts = isempty(script)

    println(tee, "\n" * "=" ^ 70)
    println(tee, "  Summary  (script=" * (all_scripts ? "ALL" : repr(script)) * ")")
    println(tee, "=" ^ 70)

    @printf(tee, "\n  %-12s  %6s  %6s  %7s  %8s  %8s  %8s\n",
            "Method", "Total", "Conv", "Rate%", "Med_IT", "Med_FE", "Med_CPU")
    println(tee, "  " * "-" ^ 62)

    for m in method_order
        df = if all_scripts
            DBInterface.execute(db, """
                SELECT r.converged, r.iterations, r.f_evals, r.cpu_time
                FROM results r JOIN configs c ON r.config_hash = c.config_hash
                WHERE c.method = ?
            """, (m,)) |> DataFrame
        else
            DBInterface.execute(db, """
                SELECT r.converged, r.iterations, r.f_evals, r.cpu_time
                FROM results r JOIN configs c ON r.config_hash = c.config_hash
                WHERE c.method = ? AND r.script = ?
            """, (m, script)) |> DataFrame
        end

        nrow(df) == 0 && continue
        n_total = nrow(df)
        conv = filter(r -> r.converged == 1, df)
        n_conv = nrow(conv)
        rate = 100.0 * n_conv / n_total
        med_it  = n_conv > 0 ? median(conv.iterations) : NaN
        med_fe  = n_conv > 0 ? median(conv.f_evals)    : NaN
        med_cpu = n_conv > 0 ? median(conv.cpu_time)   : NaN

        @printf(tee, "  %-12s  %6d  %6d  %6.1f%%  %8.0f  %8.0f  %8.4f\n",
                m, n_total, n_conv, rate, med_it, med_fe, med_cpu)
    end

    println(tee, "  " * "-" ^ 62)
    println(tee, "  DB: $(DB_PATH)")
end
