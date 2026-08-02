# Game-local diagnostic method. Its cocoercivity hypothesis is intentionally
# violated by the skew game operator; the observed behavior is reported as-is.

struct HalpernForwardBackward <: AbstractAlgorithm
    lambda::Float64
    function HalpernForwardBackward(lambda::Real)
        lambda > 0 || throw(ArgumentError("HalpernForwardBackward requires lambda > 0"))
        new(Float64(lambda))
    end
end

name(::Type{HalpernForwardBackward}) = "HFB"
version(::Type{HalpernForwardBackward}) = v"1.0.0"

function solve(alg::HalpernForwardBackward, prob::TestProblem,
               x0::Vector{Float64}; stopping::Tuple, observers::Tuple = (),
               monitor_residual::Bool = true,
               record_elapsed::Bool = true)
    state = SolverState(:HFB, x0)
    state.step_size = alg.lambda
    anchor = copy(x0)
    t0 = time()
    Bu = algorithm_B!(state, prob.B, state.x)  # initialization evaluation

    for cb in observers
        on_event!(cb, state, :init)
    end

    while true
        state.x_prev = copy(state.x)
        alpha_k = 1.0 / (state.k + 2.0)
        forward_backward = prob.resolvent_A(state.x .- alg.lambda .* Bu, alg.lambda)
        next_x = alpha_k .* anchor .+ (1.0 - alpha_k) .* forward_backward
        next_Bu = algorithm_B!(state, prob.B, next_x)

        state.x = next_x
        record_elapsed && (state.elapsed = time() - t0)
        state.step_size = alg.lambda
        if monitor_residual
            # Monitoring B-calls inside native_residual are not algorithm calls.
            state.residual = prob.native_residual(state.x, state.x_prev)
            state.scaled_residual = state.residual
        else
            state.residual = 0.0
            state.scaled_residual = 0.0
        end
        state.k += 1

        for cb in observers
            on_event!(cb, state, :iter)
        end
        halted = false
        for cb in stopping
            should_stop, reason = check_stop(cb, state)
            if should_stop
                state.flag = reason
                halted = true
                break
            end
        end
        halted && break
        Bu = next_Bu
    end

    for cb in observers
        on_event!(cb, state, :terminate)
    end
    history = IterRecord[]
    for cb in observers
        if cb isa HistoryCallback
            history = cb.history
            break
        end
    end
    return make_result(
        converged = state.flag === :converged,
        iterations = state.k,
        f_evals = state.f_evals,
        cpu_time = state.elapsed,
        x = state.x,
        flag = state.flag,
        history = history,
        residual = state.residual,
        scaled_residual = state.scaled_residual,
    )
end
