using Distributions, Plots

# Discrete-Time Asset Allocation
struct State
    W
    t
end

T = 4
μ = 0.13
σ = 0.2
N = Distributions.Normal(μ, σ)
a = 1.0
U(WT) = (1.0 - exp(-a * WT)) / a
r = 0.07
γ = 1.0

W_init = 1.0
W_var = 0.1

function dynamics(zt::State, xt)
    Yt = rand(N)
    State(xt * (Yt - r) + zt.W * (1.0 + r), zt.t + 1)
end

function optimal_policy(zt::State)
    (μ - r) / ((σ^2.0) * a * (1.0 + r)^(T - zt.t - 1))
end

function rollout(s0, policy, T)
    S = State[s0]
    A = Float64[]
    for t = 1:T-1
        push!(A, policy(S[end]))
        push!(S, dynamics(S[end], A[end]))
    end
    return S, A, U(s[end].W)
end

s0 = State(rand(Distributions.Normal(W_init, sqrt(W_var))), 0)
optimal_policy(s0)
dynamics(s0, optimal_policy(s0))
S, A, G = rollout(s0, optimal_policy, T)

plot([s.W for s in S])

function features(s)
    [1.0; s.W; s.t]
end

num_features = 3

function gaussian_policy(s, θ; σ = 0.01)
    rand(Distributions.Normal(features(s)' * θ, σ))
end

gaussian_policy(s0, rand(num_features))

function ∇θlogπ(s, a, θ; σ = 0.01)
    (a - features(s)' * θ) * features(s) / σ^2.0
end

∇θlogπ(s0, 1.0, rand(num_features))

function reinforce(s0; max_ep = 10000)
    θ = 0.01 * rand(num_features)
    α = 0.0001
    for i = 1:max_ep
        # rollout
        S = State[s0]
        A = Float64[]
        for t = 1:T-1
            push!(A, gaussian_policy(S[end], θ))
            push!(S, dynamics(S[end], A[end]))
        end
        G = U(s[end].W)

        # update
        for t = 1:T-1
            θ .+= α * γ^(t-1) * ∇θlogπ(S[t], A[t], θ) * G
        end
    end

    return θ
end

θ = reinforce(s0)
