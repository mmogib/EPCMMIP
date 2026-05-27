# io_utils.jl — TeeIO logging
#
# Dual-output logging: everything printed via the returned `tee` IO goes to
# both the console and a timestamped log file in jcode/results/logs/.

# --- TeeIO: simultaneous write to two IO streams ---

"""
    TeeIO(a::IO, b::IO)

IO wrapper that writes to both `a` and `b` simultaneously. Used for
console + logfile output without pipe-based redirection.
"""
struct TeeIO <: IO
    a::IO   # primary (console)
    b::IO   # secondary (log file)
end

function Base.unsafe_write(t::TeeIO, p::Ptr{UInt8}, n::UInt)
    Base.unsafe_write(t.a, p, n)
    Base.unsafe_write(t.b, p, n)
    return n
end

Base.flush(t::TeeIO) = (flush(t.a); flush(t.b))
Base.isopen(t::TeeIO) = isopen(t.a) && isopen(t.b)

# --- Logging ---

"""
    setup_logging(script_name::String; logdir=nothing)

Open a log file and return a `TeeIO` that writes to both console and log.
Returns `(logpath, tee, logfile)`.

Call `teardown_logging(tee, logpath)` when done.
"""
function setup_logging(script_name::String; logdir::Union{Nothing,String}=nothing)
    if logdir === nothing
        logdir = joinpath(JCODE_ROOT, "results", "logs")
    end
    mkpath(logdir)

    timestamp = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
    logpath = joinpath(logdir, "log_$(script_name)_$(timestamp).txt")
    logfile = open(logpath, "w")

    tee = TeeIO(stdout, logfile)

    println(tee, "Log file: $logpath")
    println(tee, "Timestamp: $timestamp")
    println(tee)

    return (logpath, tee, logfile)
end

"""
    teardown_logging(tee::TeeIO, logpath::String)

Flush and close the log file.
"""
function teardown_logging(tee::TeeIO, logpath::String)
    flush(tee)
    close(tee.b)
    println("Log saved to: $logpath")
end
