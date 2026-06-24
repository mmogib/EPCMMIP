# includes.jl — Single entry point for all source files.
# Usage: include("src/includes.jl") from scripts or the REPL at jcode/.
#
# ─── Include order (dependency-driven; DO NOT reorder lightly) ──────────────
#
#  1. deps.jl              ← package imports (LinearAlgebra, SQLite, ProgressMeter, …)
#  2. types.jl             ← SolverResult, IterRecord, make_result
#  3. io_utils.jl          ← TeeIO, setup_logging, teardown_logging
#  4. callbacks.jl         ← AbstractCallback hierarchy, SolverState, built-in
#                            stoppers + observers. Uses IterRecord (from types.jl).
#  5. resolvents.jl        ← clipping_box, soft_thresholding. Standalone.
#  6. problems.jl          ← TestProblem, InitialPoint, three problem builders.
#                            Uses clipping_box/soft_thresholding (from resolvents.jl).
#  7. algorithm_types.jl   ← AbstractAlgorithm + 5 concrete algorithm structs
#                            + per-algorithm PRESETS + name/version traits.
#                            Standalone (does NOT reference TestProblem here;
#                            solve method bodies in algorithm.jl).
#  8. algorithm.jl         ← solve(::T, prob::TestProblem, x0; stopping, observers)
#                            method bodies for all five solvers (EPCM, MTTM,
#                            IMTTM, SFRBM, IFRAB) + their rule dispatchers. Uses
#                            TestProblem, AbstractAlgorithm, SolverState,
#                            SolverResult, IterRecord.
#  9. benchmark.jl         ← DB layer. Uses AbstractAlgorithm, name, version
#                            (from algorithm_types.jl) and IterRecord (from
#                            types.jl). Does NOT call solve, so it loads fine
#                            even when algorithm.jl is absent — only scripts
#                            that actually run a solver need algorithm.jl.

# Project root — used by io_utils.jl (logs/), benchmark.jl (DB_PATH), and scripts.
const JCODE_ROOT = dirname(@__DIR__)

include("deps.jl")
include("types.jl")
include("io_utils.jl")
include("callbacks.jl")
include("resolvents.jl")
include("problems.jl")
include("algorithm_types.jl")
include(joinpath(JCODE_ROOT, "configs", "competitors_original_parameters_presets.jl"))  # baseline :paper/:Pk presets (hand-maintained)
include("algorithm.jl")
include("benchmark.jl")
