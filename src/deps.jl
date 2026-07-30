# deps.jl — Central dependency management
# All shared dependencies go here. Scripts add only script-specific packages.
#
# Script-only packages (add via `using` in the script, not here):
#   BenchmarkProfiles — profile/attainment plotting scripts
#   Colors — custom RGB palettes

# Standard library
using LinearAlgebra
using SparseArrays
using Printf
using Random
using Statistics
using Dates

# Database (default backend — see benchmark.jl)
using SQLite
using SHA
using DBInterface
using JSON3

# Data I/O
using DataFrames
using CSV

# Figure support
using Plots
using LaTeXStrings

# Progress bars
using ProgressMeter

# Constraint sets (uncomment if using LazySets for projections)
# using LazySets
# import LazySets: σ, an_element
