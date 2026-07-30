# Build representative MSE-versus-iteration convergence plots for the isolated
# MSE-stopping compressed-sensing experiment.

include(joinpath(@__DIR__, "s30_benchmark.jl"))
gr()

function figure_config(args)
    opts, flags = parse_cli(args)
    cases = haskey(opts, "cases") ? parse_list(opts["cases"], CASE_BY_NAME) : [DEFAULT_CASES[1].problem, DEFAULT_CASES[end].problem]
    return (cases=cases, png=("png" in flags), production=!("quick" in flags))
end

function latest_hashes(db, case_id, production)
    df = DBInterface.execute(db, """
        SELECT c.method, r.config_hash, MAX(r.created_at) AS created_at
        FROM results r JOIN configs c ON c.config_hash=r.config_hash
        WHERE r.script=? AND r.problem=? AND r.production=? AND r.init_point='seed1'
        GROUP BY c.method, r.config_hash
    """, (SCRIPT_ID, case_id, production ? 1 : 0)) |> DataFrame
    out = Dict{String,String}()
    for method in unique(String.(df.method))
        rows = df[df.method .== method, :]; sort!(rows, :created_at)
        out[method] = String(rows.config_hash[end])
    end
    return out
end

function build_mse_figures(args=ARGS)
    cfg = figure_config(args)
    logpath, tee, _ = setup_logging("mse_s70_figures"; logdir=LOGDIR)
    db = open_db(DB_PATH)
    try
        for case_id in cfg.cases
            hashes = latest_hashes(db, case_id, cfg.production)
            isempty(hashes) && (println(tee, "$case_id: no benchmark history found; run s30 first."); continue)
            plt = plot(xlabel="Iteration", ylabel="MSE", yscale=:log10, legend=:topright,
                       title="MSE convergence: $(case_title(CASE_BY_NAME[case_id]))", grid=true)
            for method in sort(collect(keys(hashes)))
                df = DBInterface.execute(db, """
                    SELECT k, mse FROM mse_history
                    WHERE script=? AND config_hash=? AND problem=? AND init_point='seed1' AND production=?
                    ORDER BY k
                """, (SCRIPT_ID, hashes[method], case_id, cfg.production ? 1 : 0)) |> DataFrame
                nrow(df) > 0 && plot!(plt, df.k, df.mse; label=method, linewidth=2)
            end
            tol_df = DBInterface.execute(db, "SELECT eps FROM configs WHERE config_hash=?", (first(values(hashes)),)) |> DataFrame
            mse_tol = nrow(tol_df) == 1 ? Float64(tol_df.eps[1]) : MSE_TOL_REF
            hline!(plt, [mse_tol]; label="MSE tolerance", linestyle=:dash, color=:black)
            stem = joinpath(FIGDIR, "mse_vs_iteration_" * lowercase(case_id))
            savefig(plt, stem * ".pdf")
            cfg.png && savefig(plt, stem * ".png")
            println(tee, "Saved $(stem).pdf")
        end
    finally
        SQLite.close(db); teardown_logging(tee, logpath)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    build_mse_figures()
end
