# Monitoring-only diagnostics and saddle-point-specific SQLite tables.

struct GameNaturalRecord
    k::Int
    residual::Float64
end

mutable struct GameNaturalHistory{F} <: AbstractObserverCallback
    native_residual::F
    records::Vector{GameNaturalRecord}
end

GameNaturalHistory(native_residual::F) where {F} =
    GameNaturalHistory{F}(native_residual, GameNaturalRecord[])

function on_event!(cb::GameNaturalHistory, state::SolverState, event::Symbol)
    event === :iter || return nothing
    # This direct diagnostic call is not routed through algorithm_B!, hence it
    # is excluded from the reported algorithmic F-evaluation count.
    value = cb.native_residual(state.x, state.x_prev)
    push!(cb.records, GameNaturalRecord(state.k, value))
    return nothing
end

struct TauBranchRecord
    tau_index::Int
    tau::Float64
    ratio_branch::Bool
end

mutable struct AEFBFPTauObserver <: AbstractObserverCallback
    xi_exp::Float64
    tau_0::Float64
    records::Vector{TauBranchRecord}
end

AEFBFPTauObserver(xi_exp::Real, tau_0::Real) =
    AEFBFPTauObserver(Float64(xi_exp), Float64(tau_0), TauBranchRecord[])

function on_event!(cb::AEFBFPTauObserver, state::SolverState, event::Symbol)
    if event === :init
        push!(cb.records, TauBranchRecord(0, cb.tau_0, false))
    elseif event === :iter
        previous_tau = isempty(cb.records) ? cb.tau_0 : last(cb.records).tau
        # state.k is k+1 at the event, while xi_k=(k+1)^(-xi_exp).
        xi_k = state.k^(-cb.xi_exp)
        increment_candidate = previous_tau + xi_k
        scale = max(1.0, abs(state.step_size), abs(increment_candidate))
        tie_tolerance = 64 * eps(scale)
        ratio_branch = state.step_size < increment_candidate - tie_tolerance
        push!(cb.records, TauBranchRecord(state.k, state.step_size, ratio_branch))
    end
    return nothing
end

function min_branch_fraction(cb::AEFBFPTauObserver)
    iteration_records = filter(record -> record.tau_index > 0, cb.records)
    isempty(iteration_records) && return NaN
    return count(record -> record.ratio_branch, iteration_records) / length(iteration_records)
end

function anchor_diagnostics(prob::TestProblem, u0::Vector{Float64}, ubar::Vector{Float64})
    prob.metadata.instance_kind === :duplicated_identity ||
        throw(ArgumentError("Anchor diagnostics require the duplicated-identity game."))
    q = something(prob.metadata.q)
    projection = degenerate_solution_projection(u0, q)
    displacement = norm(u0 .- ubar)
    distance_to_solution = norm(u0 .- projection)
    return (
        distance_to_projection = norm(ubar .- projection),
        displacement = displacement,
        distance_to_solution = distance_to_solution,
        projection_distance_error = abs(displacement - distance_to_solution),
        certificate = degenerate_projection_certificate(u0, ubar, q),
        target_hash = array_sha256(projection),
    )
end

const GAME_METRICS_SQL = """
CREATE TABLE IF NOT EXISTS game_final_metrics (
    config_hash TEXT NOT NULL,
    problem TEXT NOT NULL,
    dimension INTEGER NOT NULL,
    init_point TEXT NOT NULL,
    seed_idx INTEGER NOT NULL,
    run_id TEXT NOT NULL,
    script TEXT NOT NULL,
    production INTEGER NOT NULL,
    natural_residual REAL NOT NULL,
    duality_gap REAL NOT NULL,
    min_branch_fraction REAL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (config_hash, problem, dimension, init_point, script, production)
)
"""

const GAME_HISTORY_SQL = """
CREATE TABLE IF NOT EXISTS game_history (
    config_hash TEXT NOT NULL,
    problem TEXT NOT NULL,
    dimension INTEGER NOT NULL,
    init_point TEXT NOT NULL,
    seed_idx INTEGER NOT NULL,
    script TEXT NOT NULL,
    production INTEGER NOT NULL,
    k INTEGER NOT NULL,
    natural_residual REAL NOT NULL,
    tau REAL,
    ratio_branch INTEGER,
    PRIMARY KEY (config_hash, problem, dimension, init_point, script, production, k)
)
"""

const GAME_ANCHOR_SQL = """
CREATE TABLE IF NOT EXISTS game_anchor_metrics (
    config_hash TEXT NOT NULL,
    problem TEXT NOT NULL,
    dimension INTEGER NOT NULL,
    init_point TEXT NOT NULL,
    seed_idx INTEGER NOT NULL,
    script TEXT NOT NULL,
    production INTEGER NOT NULL,
    distance_to_projection REAL NOT NULL,
    displacement REAL NOT NULL,
    distance_to_solution REAL NOT NULL,
    projection_distance_error REAL NOT NULL,
    certificate REAL NOT NULL,
    target_hash TEXT NOT NULL,
    PRIMARY KEY (config_hash, problem, dimension, init_point, script, production)
)
"""

function ensure_game_tables!(db)
    DBInterface.execute(db, GAME_METRICS_SQL)
    DBInterface.execute(db, GAME_HISTORY_SQL)
    DBInterface.execute(db, GAME_ANCHOR_SQL)
    return nothing
end

function insert_game_metrics!(db, hash, problem, dimension, init, run_id, result,
                              natural_residual, duality_gap, branch_fraction;
                              script, production)
    DBInterface.execute(db, """
        INSERT OR REPLACE INTO game_final_metrics
        (config_hash, problem, dimension, init_point, seed_idx, run_id, script,
         production, natural_residual, duality_gap, min_branch_fraction, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (hash, problem, dimension, init.label, init.seed_idx, run_id, script,
           production ? 1 : 0, natural_residual, duality_gap, branch_fraction,
           Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS")))
    return nothing
end


function insert_game_history!(db, hash, problem, dimension, init, natural_records,
                              tau_records; script, production)
    isempty(natural_records) && return nothing
    tau_by_k = Dict(record.tau_index => record for record in tau_records)
    DBInterface.execute(db, "BEGIN TRANSACTION")
    try
        DBInterface.execute(db, """
            DELETE FROM game_history
            WHERE config_hash=? AND problem=? AND dimension=? AND init_point=?
              AND script=? AND production=?
        """, (hash, problem, dimension, init.label, script, production ? 1 : 0))
        tau_zero = get(tau_by_k, 0, nothing)
        if tau_zero !== nothing
            DBInterface.execute(db, """
                INSERT INTO game_history
                (config_hash, problem, dimension, init_point, seed_idx, script,
                 production, k, natural_residual, tau, ratio_branch)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (hash, problem, dimension, init.label, init.seed_idx, script,
                   production ? 1 : 0, 0, -1.0, tau_zero.tau, 0))
        end
        for record in natural_records
            tau_record = get(tau_by_k, record.k, nothing)
            DBInterface.execute(db, """
                INSERT INTO game_history
                (config_hash, problem, dimension, init_point, seed_idx, script,
                 production, k, natural_residual, tau, ratio_branch)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (hash, problem, dimension, init.label, init.seed_idx, script,
                   production ? 1 : 0, record.k, record.residual,
                   tau_record === nothing ? missing : tau_record.tau,
                   tau_record === nothing ? missing : (tau_record.ratio_branch ? 1 : 0)))
        end
        DBInterface.execute(db, "COMMIT")
    catch
        DBInterface.execute(db, "ROLLBACK")
        rethrow()
    end
    return nothing
end

function insert_anchor_metrics!(db, hash, problem, dimension, init, metrics;
                                script, production)
    DBInterface.execute(db, """
        INSERT OR REPLACE INTO game_anchor_metrics
        (config_hash, problem, dimension, init_point, seed_idx, script,
         production, distance_to_projection, displacement, distance_to_solution,
         projection_distance_error, certificate, target_hash)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (hash, problem, dimension, init.label, init.seed_idx, script,
           production ? 1 : 0, metrics.distance_to_projection,
           metrics.displacement, metrics.distance_to_solution,
           metrics.projection_distance_error, metrics.certificate,
           metrics.target_hash))
    return nothing
end
