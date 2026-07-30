include(joinpath(@__DIR__, "s20_aefbfp_parameter_search.jl"))

const PARTIAL_WINNER_HASH = "7547a169b2a7"
const PARTIAL_WINNER_SCORE = (
    nconv = 40,
    med_iter = 2607.5,
    med_eval = 2608.5,
    med_cpu = 0.8309999704360962,
)

function main()
    db = open_db(DB_PATH)
    ensure_local_tables!(db)

    try
        row = DBInterface.execute(db, """
            SELECT config_hash, params_json
            FROM configs
            WHERE config_hash = ?
        """, (PARTIAL_WINNER_HASH,)) |> DataFrame

        nrow(row) == 1 || error("Expected exactly one config row for hash $(PARTIAL_WINNER_HASH), found $(nrow(row))")

        promote_winner!(db, PARTIAL_WINNER_HASH, PARTIAL_WINNER_SCORE)

        println("Promoted AEFBFP winner for compressed_sensing")
        println("  hash   = $(row.config_hash[1])")
        println("  params = $(row.params_json[1])")
        println("  score  = $(PARTIAL_WINNER_SCORE)")
    finally
        SQLite.close(db)
    end

    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
