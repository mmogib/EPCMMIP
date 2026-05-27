# includes.jl — Single entry point for all source files
# Usage: include("src/includes.jl") from scripts or REPL at jcode/ directory
#
# Include order matters: each file may depend on files included before it.

# Project root (used by all scripts for results/, logs/, etc.)
const JCODE_ROOT = dirname(@__DIR__)

include("deps.jl")             # 1. Dependencies
include("types.jl")            # 2. SolverResult, IterRecord, make_result
include("io_utils.jl")         # 3. TeeIO, setup_logging, teardown_logging

# Problem domains — uncomment as files are created
# include("problems.jl")       # 4. Test problems (Tan2022a 5.1, Yao2024 4.2, Izuchukwu2023 6.2)

# Algorithm components — uncomment as files are created
# include("resolvents.jl")     # closed-form resolvents J^A_lambda per problem
# include("algorithm.jl")      # 5. EPCM (proposed) + benchmark algorithms (MTTM, IMTTM, IFRAB, SFRBM)

# Infrastructure
include("benchmark.jl")        # 6. DB layer (open_db, config hash, CRUD)
