using StatsBase

include("initialize_transitions.jl")
include("value_iteration.jl")

# random walk
B1 = 3
B2 = 3
maze = zeros(B1 + 1, B2 + 1)

# states
struct MazeState
    x::Int
    y::Int
end

S = []
for i = 0:B1
    for j = 0:B2
        push!(S, MazeState(i, j))
    end
end
length(S)

T = []
for j = 0:B2
    push!(T, MazeState(0, j))
    push!(T, MazeState(B1, j))
end

for i = 0:B1
    push!(T, MazeState(i, 0))
    push!(T, MazeState(i, B2))
end

N = setdiff(S, T)

# actions
function state_action_transition(s, a)
    if a == :LEFT
        return (s.x, s.y - 1)
    elseif a == :RIGHT
        return (s.x, s.y + 1)
    elseif a == :UP
        return (s.x - 1, s.y)
    elseif a == :DOWN
        return (s.x + 1, s.y)
    end
end
A_set = [:LEFT, :RIGHT, :UP, :DOWN]
A = Dict()
for s in N
    _a = []
    for a in A_set
        t = MazeState(state_action_transition(s, a)...)
        if t in S
            push!(_a, a)
        end
    end
    push!(A, s => _a)
end

# transitions
P = initialize_transitions(S, A, N = N)
for s in N
    for a in A[s]
        t = MazeState(state_action_transition(s, a)...)
        P[(s, a, t)] = 1.0
    end
end

# rewards
# option 1
γ = 0.9
R = initialize_transitions(S, A, N = N, init = 0.0)
for s in N
    for a in A[s]
        t = MazeState(state_action_transition(s, a)...)
        if t.x == B1
            R[(s, a, t)] = 1.0
        end
        if t.y == B2
            R[(s, a, t)] = 1.0
        end
    end
end

# value iteration
# rewards option 1
V, Π = value_iteration(S, A, P, R;
    V = Dict([s => 0.0 for s in S]),
    T = T,
    N = setdiff(S, T),
    γ = γ, iter = 1000, verbose = false)

Vmatrix = zeros(B1 + 1, B2 + 1)

for s in S
    Vmatrix[s.x + 1, s.y + 1] = V[s]
end
@show Vmatrix


# sample random state
function sample_next_state(s, a, S)
    probs = Float64[]
    for t in S
        push!(probs, P[(s, a, t)])
    end
    sample(S, Weights(probs))
end

function ϵ_greedy_policy(Π, s, ϵ)
    if rand(1)[1] <= 1.0 - ϵ
        return Π[s]
    else
        return rand(A[s])
    end
end

function rollout(Π; ϵ = 0.0, max_iter = 100)
    s = [rand(N)] # random initial non-terminal state
    a = []
    r = []

    iter = 1
    while s[end] ∉ T
        push!(a, ϵ_greedy_policy(Π, s[end], ϵ))
        push!(s, sample_next_state(s[end], a[end], S))
        push!(r, R[(s[end-1], a[end], s[end])])

        iter += 1
        if iter > max_iter
            @warn "episode not complete"
            break
        end
    end
    return s, a, r
end

function discounted_return(r, γ)
    G = 0.0
    for t = 0:length(r)-1
        G += r[t+1] * γ^t
    end
    return G
end

num_features = sum([length(A[s]) for s in N])

function features(s, a)
    x = zeros(num_features)
    id = 1
    for _s in N
        for _a in A[_s]
            if (s, a) == (_s, _a)
                x[id] = 1.0
                return x
            else
                id += 1
            end
        end
    end
    @warn "invalid state-action pair"
    return x
end

# Monte Carlo control
function monte_carlo_control(; iter = 100)
    w = zeros(num_features)
    Qmc = Dict([(s, a) => 0.0 for s in N for a in A[s]])
    Nmc = Dict([(s, a) => 0.0 for s in N for a in A[s]])
    Πmc = Dict([s => rand(A[s]) for s in N]) # random initial policy
    println("Monte Carlo Control")

    for k = 1:iter
        k % 100 == 0 && println("iter: $k")

        ϵ = 1.0 / k

        # rollout
        s, a, r = rollout(Πmc, ϵ = ϵ, max_iter = 100)

        # update action-value function
        for t = 1:length(s)-1
            Nmc[(s[t], a[t])] += 1
            G = discounted_return(r[t:end], γ)
            w += (1 / Nmc[(s[t], a[t])]) * (G - w' * features(s[t], a[t])) * features(s[t], a[t])
            # w += (1 / Nmc[(s[t], a[t])]) * (V[s[t]] - w' * features(s[t], a[t])) * features(s[t], a[t])
        end

        # policy update
        for s in N
            Πmc[s] = argmax(Dict([a => w' * features(s, a) for a in A[s]]))
        end
    end

    for s in N
        for a in A[s]
            Qmc[(s, a)] = w' * features(s, a)
        end
    end

    return Qmc, Πmc, w
end

Qmc, Πmc, wmc = monte_carlo_control(iter = 10000000)
Vmc = Dict([s => (s ∈ T ? 0.0 : Qmc[(s, Πmc[s])]) for s in S])

@show Πmc
Vmatrix_mc = zeros(B1 + 1, B2 + 1)

for s in S
    Vmatrix_mc[s.x + 1, s.y + 1] = Vmc[s]
end
@show Vmatrix_mc
