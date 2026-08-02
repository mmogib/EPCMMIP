using Test

const TIMING_TEST_ROOT = normpath(joinpath(@__DIR__, ".."))
include(joinpath(TIMING_TEST_ROOT, "src", "includes.jl"))
include(joinpath(TIMING_TEST_ROOT, "scripts", "saddle_point",
                 "halpern_forward_backward.jl"))

function counted_linear_problem()
    b_calls = Ref(0)
    prox_calls = Ref(0)
    B = x -> begin
        b_calls[] += 1
        return 0.1 .* x
    end
    prox = (x, rho) -> begin
        prox_calls[] += 1
        return copy(x)
    end
    native = (x, x_prev) -> norm(x .- prox(x .- B(x), 1.0))
    x0 = [0.8, -0.4, 0.2]
    prob = TestProblem(9901, "timing_counter", length(x0), B, prox, native,
                       zeros(length(x0)),
                       [InitialPoint("seed1", 1, copy(x0))],
                       (L = 0.1,))
    return prob, x0, b_calls, prox_calls
end

function run_counted(alg; monitor_residual::Bool, record_elapsed::Bool)
    prob, x0, b_calls, prox_calls = counted_linear_problem()
    result = solve(alg, prob, copy(x0);
                   stopping = (MaxIterStopping(3), NanStopping()),
                   observers = (),
                   monitor_residual = monitor_residual,
                   record_elapsed = record_elapsed)
    return result, b_calls[], prox_calls[]
end

const TIMING_TEST_ALGORITHMS = (
    AEFBFP(:P3),
    VAFBS(:paper),
    MDITSM(:paper),
    RFBSM(:paper),
    IRFBSM(:paper),
    IFRAB(:paper),
    HalpernForwardBackward(0.5),
)

@testset "monitoring-free solver signatures" begin
    for alg in TIMING_TEST_ALGORITHMS
        regular, regular_B, regular_prox =
            run_counted(alg; monitor_residual = true, record_elapsed = true)
        lean, lean_B, lean_prox =
            run_counted(alg; monitor_residual = false, record_elapsed = false)

        @test (lean.flag, lean.iterations, lean.f_evals) ==
              (regular.flag, regular.iterations, regular.f_evals)
        @test lean.x == regular.x
        @test lean.residual == 0.0
        @test lean.scaled_residual == 0.0
        @test lean.cpu_time == 0.0
        @test lean_B <= regular_B
        @test lean_prox < regular_prox

        if alg isa AEFBFP || alg isa MDITSM || alg isa IRFBSM ||
           alg isa HalpernForwardBackward
            @test regular_B - lean_B == lean.iterations
        else
            @test regular_B == lean_B
        end
        @test regular_prox - lean_prox == lean.iterations
    end
end

const TIMING_V2_SOURCE = joinpath(TIMING_TEST_ROOT, "src", "timing_protocol_v2.jl")

@testset "external timing and retained repetitions" begin
    @test isfile(TIMING_V2_SOURCE)
    @test isdefined(Main, :run_timing_v2)
    @test isdefined(Main, :ensure_timing_v2_table!)

    if isdefined(Main, :run_timing_v2) &&
       isdefined(Main, :ensure_timing_v2_table!)
        calls = Ref(0)
        expected = (:maxiter, 2, 3)
        solve_once = () -> begin
            calls[] += 1
            sleep(0.001)
            return make_result(
                converged = false,
                iterations = 2,
                f_evals = 3,
                cpu_time = 0.0,
                x = [1.0, 2.0],
                flag = :maxiter,
                residual = 0.0,
                scaled_residual = 0.0,
            )
        end

        batches = run_timing_v2(solve_once, expected;
                                warmups = 2,
                                repetitions = 3,
                                min_batch_seconds = 0.004)
        @test length(batches) == 3
        @test calls[] >= 2 + sum(batch.batch_size for batch in batches)
        @test all(batch -> batch.batch_size >= 1, batches)
        @test all(batch -> batch.total_ns >= 4_000_000, batches)
        @test all(batch -> batch.ms_per_solve > 0.0, batches)
        @test all(batch -> batch.signature == expected, batches)
        @test_throws ErrorException run_timing_v2(solve_once, (:converged, 2, 3);
                                                   warmups = 1,
                                                   repetitions = 1,
                                                   min_batch_seconds = 0.001)

        mktempdir() do dir
            db = SQLite.DB(joinpath(dir, "timing.db"))
            try
                ensure_timing_v2_table!(db)
                protocol_hash = timing_v2_protocol_hash(
                    warmups = 2,
                    repetitions = 3,
                    min_batch_seconds = 0.1,
                )
                samples = (1.0, 2.0, 4.0)
                for (rep, ms) in enumerate(samples)
                    row = TimingV2Repetition(
                        protocol_hash = protocol_hash,
                        family = "test",
                        method = "AEFBFP",
                        source_config_hash = "abc123",
                        source_script = "s30",
                        problem = "P",
                        dimension = 2,
                        init_point = "seed1",
                        seed_idx = 1,
                        production = true,
                        repetition = rep,
                        batch_size = 2,
                        total_ns = round(Int, 2_000_000 * ms),
                        ms_per_solve = ms,
                        expected_flag = :maxiter,
                        expected_iterations = 2,
                        expected_f_evals = 3,
                        run_id = "test_run",
                    )
                    insert_timing_v2_repetition!(db, row)
                end
                # Duplicate insertion is resume-safe and does not add a row.
                duplicate = TimingV2Repetition(
                    protocol_hash = protocol_hash,
                    family = "test",
                    method = "AEFBFP",
                    source_config_hash = "abc123",
                    source_script = "s30",
                    problem = "P",
                    dimension = 2,
                    init_point = "seed1",
                    seed_idx = 1,
                    production = true,
                    repetition = 1,
                    batch_size = 2,
                    total_ns = 2_000_000,
                    ms_per_solve = 1.0,
                    expected_flag = :maxiter,
                    expected_iterations = 2,
                    expected_f_evals = 3,
                    run_id = "test_run",
                )
                insert_timing_v2_repetition!(db, duplicate)
                count_df = DBInterface.execute(
                    db,
                    "SELECT COUNT(*) FROM timing_v2_repetitions",
                ) |> DataFrame
                @test Int(count_df[1, 1]) == 3
                @test timing_v2_done(db, protocol_hash, "abc123", "P", 2,
                                     "seed1", true; repetitions = 3)

                summary = timing_v2_summary(db; protocol_hash = protocol_hash)
                @test nrow(summary) == 1
                @test summary.median_ms[1] == 2.0
                @test summary.q1_ms[1] == 1.5
                @test summary.q3_ms[1] == 3.0
                @test summary.min_ms[1] == 1.0
                @test summary.max_ms[1] == 4.0

                source_rows = DataFrame(
                    config_hash = ["abc123"], problem = ["P"],
                    dimension = [2], init_point = ["seed1"],
                    legacy_cpu_time = [99.0],
                )
                attached = attach_timing_v2(
                    db, source_rows; warmups = 2, repetitions = 3,
                    min_batch_seconds = 0.1, protocol_hash = protocol_hash,
                )
                @test nrow(attached) == 1
                @test attached.timing_ms[1] == 2.0
                @test attached.timing_q1_ms[1] == 1.5
                @test attached.timing_q3_ms[1] == 3.0
                @test attached.timing_repetitions[1] == 3
                @test attached.legacy_cpu_time[1] == 99.0
                bad_source = copy(source_rows)
                bad_source.config_hash[1] = "missing"
                @test_throws ErrorException attach_timing_v2(
                    db, bad_source; warmups = 2, repetitions = 3,
                    min_batch_seconds = 0.1, protocol_hash = protocol_hash,
                )
                @test_throws ErrorException attach_timing_v2(
                    db, source_rows; warmups = 2, repetitions = 3,
                    min_batch_seconds = 0.1, protocol_hash = "wrong",
                )
            finally
                SQLite.close(db)
            end
        end
    end
end
