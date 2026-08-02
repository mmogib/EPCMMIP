# Replay one untimed deterministic solve per completed cell to compute the
# matrix-game gap at the feasible shadow P_C(u_k). Performance rows are not
# changed; only game_final_metrics.duality_gap is repaired/audited.

include(joinpath(@__DIR__, "s30_benchmark.jl"))

const GAME_GAP_AUDIT_SQL = """
CREATE TABLE IF NOT EXISTS game_gap_audit (
    config_hash TEXT NOT NULL,
    problem TEXT NOT NULL,
    dimension INTEGER NOT NULL,
    init_point TEXT NOT NULL,
    projected_gap REAL NOT NULL,
    final_iterate_hash TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (config_hash, problem, dimension, init_point)
)
"""

function gap_audit_done(db, hash, prob, init)
    rows = DBInterface.execute(db, """
        SELECT 1 FROM game_gap_audit
        WHERE config_hash=? AND problem=? AND dimension=? AND init_point=?
    """, (hash, prob.name, prob.dim, init.label)) |> DataFrame
    return nrow(rows) > 0
end

function stored_game_signature(db, hash, prob, init)
    rows = DBInterface.execute(db, """
        SELECT flag, iterations, f_evals FROM results
        WHERE config_hash=? AND problem=? AND dimension=? AND init_point=?
          AND script='s30' AND production=1
    """, (hash, prob.name, prob.dim, init.label)) |> DataFrame
    nrow(rows) == 1 || error("Expected one production result for $(prob.name)/$(init.label), found $(nrow(rows)).")
    row = first(eachrow(rows))
    return (Symbol(row.flag), Int(row.iterations), Int(row.f_evals))
end

function update_gap_audit!(db, hash, prob, init, gap, result)
    DBInterface.execute(db, """
        UPDATE game_final_metrics SET duality_gap=?
        WHERE config_hash=? AND problem=? AND dimension=? AND init_point=?
          AND script='s30' AND production=1
    """, (gap, hash, prob.name, prob.dim, init.label))
    DBInterface.execute(db, """
        INSERT OR REPLACE INTO game_gap_audit
        (config_hash, problem, dimension, init_point, projected_gap,
         final_iterate_hash, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (hash, prob.name, prob.dim, init.label, gap, array_sha256(result.x),
           Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS")))
    return nothing
end

function game_gap_replay_main(args = ARGS)
    runtime = configure_reproducible_runtime!()
    cfg = read_game_config(args)
    cfg.production || error("Gap replay is only for definitive production rows.")
    db = open_db(GAME_DB_PATH)
    ensure_game_tables!(db)
    DBInterface.execute(db, GAME_GAP_AUDIT_SQL)
    try
        println("Feasible-shadow game-gap replay on Julia $(runtime.julia_version)")
        for instance in cfg.instances
            prob = build_game_instance(instance; n_inits = cfg.initial_points)
            println("[$instance]")
            for method in cfg.methods
                alg = build_game_algorithm(method, prob)
                hash, _ = game_config_hash(alg, prob, cfg)
                for init in prob.initial_points
                    if gap_audit_done(db, hash, prob, init)
                        println("  $method $(init.label) skipped")
                        continue
                    end
                    expected = stored_game_signature(db, hash, prob, init)
                    result = solve(alg, prob, copy(init.x0);
                                   stopping = game_stopping(prob, cfg), observers = ())
                    actual = game_result_signature(result)
                    actual == expected || error("Gap replay signature mismatch for $(prob.name)/$method/$(init.label): stored=$expected replay=$actual")
                    gap = max(game_duality_gap(prob, result.x), 0.0)
                    update_gap_audit!(db, hash, prob, init, gap, result)
                    @printf("  %-8s %-6s gap=%9.3e\n", method, init.label, gap)
                end
            end
        end
    finally
        SQLite.close(db)
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    game_gap_replay_main()
end
