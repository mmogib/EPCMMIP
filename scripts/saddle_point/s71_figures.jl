# MOHAMMED-ONLY plotting handoff. Codex must never execute this file.

include(joinpath(@__DIR__, "s30_benchmark.jl"))
using Plots
using LaTeXStrings

const GAME_FIGDIR = joinpath(GAME_RESULT_ROOT, "figures")

function load_random500_history(db)
    return DBInterface.execute(db, """
        SELECT c.method, h.k, h.natural_residual, h.tau, h.ratio_branch
        FROM game_history h JOIN configs c ON c.config_hash=h.config_hash
        WHERE h.problem='random_500' AND h.init_point='seed1'
          AND h.script='s30' AND h.production=1
          AND c.hash_input LIKE '%protocol=game_manuscript_v1%'
          AND c.hash_input LIKE '%reps=3%'
        ORDER BY c.method, h.k
    """) |> DataFrame
end

function game_single_panel_plot(; kwargs...)
    return plot(; size = (900, 620),
                dpi = 220,
                background_color = :white,
                background_color_inside = :white,
                foreground_color_subplot = :black,
                foreground_color_legend = :black,
                background_color_legend = :white,
                legend_font_pointsize = 11,
                guidefont = font(12),
                tickfont = font(10),
                left_margin = 10Plots.mm,
                right_margin = 6Plots.mm,
                bottom_margin = 8Plots.mm,
                top_margin = 5Plots.mm,
                gridalpha = 0.18,
                framestyle = :box,
                kwargs...)
end

function save_game_plot_files(plt, stem::String)
    savefig(deepcopy(plt), joinpath(GAME_FIGDIR, stem * ".pdf"))
    savefig(deepcopy(plt), joinpath(GAME_FIGDIR, stem * ".png"))
    return nothing
end

function saddle_figures_main()
    mkpath(GAME_FIGDIR)
    db = open_db(GAME_DB_PATH)
    try
        df = load_random500_history(db)
        nrow(df) > 0 || error("No definitive random_500 history rows found. Run s30_benchmark.jl first.")

        residual_plot = game_single_panel_plot(;
            title = "random-500, start 1",
            xlabel = "Iteration",
            ylabel = L"R_k",
            yscale = :log10,
            legend = :topright,
            linewidth = 2)
        for method in GAME_METHOD_NAMES
            rows = df[(df.method .== method) .& (df.k .>= 1), :]
            nrow(rows) == 0 && continue
            plot!(residual_plot, rows.k, rows.natural_residual; label = method)
        end
        save_game_plot_files(residual_plot, "game_random500_residual")

        tau_rows = df[(df.method .== "AEFBFP") .& .!ismissing.(df.tau), :]
        nrow(tau_rows) > 0 || error("No AEFBFP tau history found for random_500/seed1.")
        tau_values = Float64.(tau_rows.tau)
        tau_span = maximum(tau_values) - minimum(tau_values)
        tau_padding = max(0.06 * tau_span, 1.0e-3)
        tau_plot = game_single_panel_plot(;
            title = "random-500, start 1",
            xlabel = "Event index",
            ylabel = L"\tau_k",
            legend = :topright,
            ylims = (minimum(tau_values) - tau_padding,
                     maximum(tau_values) + tau_padding))
        plot!(tau_plot, tau_rows.k, tau_values;
              label = "AEFBFP",
              linewidth = 2)
        ratio_rows = tau_rows[coalesce.(tau_rows.ratio_branch .== 1, false), :]
        nrow(ratio_rows) == 0 || scatter!(tau_plot, ratio_rows.k, Float64.(ratio_rows.tau);
                                          label = "ratio branch",
                                          markersize = 1.6,
                                          markeralpha = 0.45,
                                          markerstrokewidth = 0.0)
        save_game_plot_files(tau_plot, "game_random500_aefbfp_tau")
        println("Wrote residual and tau figures to $GAME_FIGDIR")
    finally
        SQLite.close(db)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    saddle_figures_main()
end
