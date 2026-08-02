# reproducibility.jl — deterministic runtime setup and realized-artifact hashes.

const EXPECTED_JULIA_VERSION = v"1.12.6"

"Configure and validate the deterministic single-thread benchmark runtime."
function configure_reproducible_runtime!()
    VERSION == EXPECTED_JULIA_VERSION ||
        error("Expected Julia $(EXPECTED_JULIA_VERSION), got $(VERSION)")
    Threads.nthreads() == 1 ||
        error("JULIA_NUM_THREADS must be 1, got $(Threads.nthreads())")
    BLAS.set_num_threads(1)
    BLAS.get_num_threads() == 1 ||
        error("BLAS thread count must be 1, got $(BLAS.get_num_threads())")

    return (
        julia_version = string(VERSION),
        julia_threads = Threads.nthreads(),
        blas_threads = BLAS.get_num_threads(),
        blas_config = sprint(show, BLAS.get_config()),
        cpu_name = Sys.CPU_NAME,
        word_size = Sys.WORD_SIZE,
        machine = Sys.MACHINE,
    )
end

"Return a shape- and element-type-aware SHA-256 hash of an isbits array."
function array_sha256(A::AbstractArray{T}) where {T}
    isbitstype(T) || throw(ArgumentError("array_sha256 requires an isbits element type, got $T"))
    header = "eltype=$(T)|size=$(join(size(A), ','))|"
    payload = Vector{UInt8}(codeunits(header))
    contiguous = vec(copy(A))
    append!(payload, reinterpret(UInt8, contiguous))
    return bytes2hex(sha256(payload))
end

"Return the SHA-256 hash of a file, or `nothing` when it is absent."
function file_sha256(path::AbstractString)
    isfile(path) || return nothing
    return open(path, "r") do io
        bytes2hex(sha256(io))
    end
end

"Write a self-contained JSON run manifest and return its path."
function write_run_manifest(path::AbstractString; runtime, protocol, seeds, hashes,
                            parameters, project_manifest::AbstractString)
    mkpath(dirname(path))
    manifest = (
        created_at = Dates.format(Dates.now(), dateformat"yyyy-mm-ddTHH:MM:SS"),
        runtime = runtime,
        protocol = protocol,
        seeds = seeds,
        hashes = hashes,
        parameters = parameters,
        manifest_sha256 = file_sha256(project_manifest),
    )
    open(path, "w") do io
        JSON3.pretty(io, manifest)
        println(io)
    end
    return path
end
