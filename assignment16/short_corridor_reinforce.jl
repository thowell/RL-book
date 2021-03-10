using StatsBase, LinearAlgebra, Plots

include("initialize_transitions.jl")
include("value_iteration.jl")

struct State
    s
end
S = [State(i) for i = 1:4]
T = [State(4)]
N = setdiff(S, T)

# actions
A_set = [:LEFT, :RIGHT]
A = Dict()
for s in N
    _a = []
    for a in A_set
        push!(_a, a)
    end
    push!(A, s => _a)
end

function state_action_transition(s, a)
    if s != State(2)
        if a == :LEFT
            return State(max(1, s.s - 1))
        elseif a == :RIGHT
            return State(s.s + 1)
        end
    else
        if a == :LEFT
            return State(s.s + 1)
        elseif a == :RIGHT
            return State(s.s - 1)
        end
    end
end

# transitions
P = initialize_transitions(S, A, N = N)
for s in N
    for a in A[s]
        t = state_action_transition(s, a)
        P[(s, a, t)] = 1.0
    end
end

function sample_next_state(s, a, S)
    probs = Float64[]
    for t in S
        push!(probs, P[(s, a, t)])
    end
    sample(S, Weights(probs))
end

sample_next_state(State(3), :RIGHT, S)

function discounted_return(r, γ)
    G = 0.0
    for t = 0:length(r)-1
        G += r[t+1] * γ^t
    end
    return G
end

# rewards
# option 1
γ = 1
R = initialize_transitions(S, A, N = N, init = -1.0)

V, Π = value_iteration(S, A, P, R;
    V = Dict([s => 0.0 for s in S]),
    T = T,
    N = setdiff(S, T),
    γ = γ, iter = 100, verbose = false)

pretty_print2(V)
pretty_print2(Π)


num_features = 2

function features(s, a)
    if a == :LEFT
        return [0; 1]
    elseif a == :RIGHT
        return [1; 0]
    end
end

function policy(s, a, θ)
    x = [θ' * features(s, b) for b in A[s]]
    exp(θ' * features(s, a) - maximum(x)) / sum([exp(xi) for xi in x])
end

function ∇θlog_policy(s, a, θ)
    features(s, a) - sum([policy(s, b, θ) * features(s, b) for b in A[s]])
end

# features(State(1), :RIGHT)
# policy(State(1), :LEFT, rand(2))
# ∇θlog_policy(State(1), :RIGHT, rand(2))
#
# θ = rand(2)
# s = [State(1)]
# p = [policy(s[end], b, θ) for b in A[s[end]]]
#
function REINFORCE(; ep = 1000)
    g = []
    θ = [-1.47; 1.47] # epsilon-greedy left policy

    α = 1.0e-3
    for i = 1:ep
        println("iter: $i")

        # rollout
        s = [State(1)]
        a = []
        r = []

        while s[end] ∉ T
            push!(a, sample(A[s[end]], Weights([policy(s[end], b, θ) for b in A[s[end]]])))
            push!(s, sample_next_state(s[end], a[end], S))
            push!(r, R[(s[end-1], a[end], s[end])])
        end

        # parameter update

        for t = 1:length(s)-1
            G = discounted_return(r[t:end], γ)
            θ .+= α * γ^(t-1) * ∇θlog_policy(s[t], a[t], θ) * G # perfect baseline
            # θ .+= α * γ^(t-1) * ∇θlog_policy(s[t], a[t], θ) * (G - V[State(1)]) # perfect baseline
        end

        # evaluate
        G_eval = []
        for j = 1:100
            # rollout
            s = [State(1)]
            a = []
            r = []

            while s[end] ∉ T
                push!(a, sample(A[s[end]], Weights([policy(s[end], b, θ) for b in A[s[end]]])))
                push!(s, sample_next_state(s[end], a[end], S))
                push!(r, R[(s[end-1], a[end], s[end])])
            end

            push!(G_eval, discounted_return(r, γ))
        end

        push!(g, mean(G_eval))
    end

    return θ, g
end

θ, G_hist = REINFORCE(ep = 1000)
plot(G_hist,
    title = "short-corridor grid world (REINFORCE)",
    xlabel = "iteration",
    ylabel = "return (avg. over 100)",
    label = "")
