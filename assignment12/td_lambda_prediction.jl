using StatsBase

include("initialize_transitions.jl")
include("grid_maze.jl")
include("value_iteration.jl")

maze = zeros(8, 8)
for (k, v) in maze_grid
    if v == :SPACE || v == :GOAL
        maze[k[1] + 1, k[2] + 1] = 1
    end
end
@show maze

# states
struct MazeState
    x::Int
    y::Int
end

S = []
for (k, v) in maze_grid
    if v == :SPACE || v == :GOAL
        # println(v)
        push!(S, MazeState(k[1], k[2]))
    end
end
S_dim = length(S)
T = []
for (k, v) in maze_grid
    if v == :GOAL
        # println(v)
        push!(T, MazeState(k[1], k[2]))
    end
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
γ = 1.0
R = initialize_transitions(S, A, N = N, init = 0.0)
for s in N
    for a in A[s]
        t = MazeState(state_action_transition(s, a)...)
        R[(s, a, t)] = -1.0
    end
end

# value iteration
# rewards option 1
V, Π = value_iteration(S, A, P, R;
    V = Dict([s => 0.0 for s in S]),
    T = T,
    N = setdiff(S, T),
    γ = γ, iter = 100, verbose = false)

pretty_print2(V)
pretty_print2(Π)

Vmatrix = zeros(8, 8)

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

function rollout(; s1 = rand(N), max_iter = 100)
    s = [s1] # random initial non-terminal state
    a = []
    r = []

    iter = 1
    while s[end] ∉ T
        push!(a, Π[s[end]])
        push!(s, sample_next_state(s[end], a[end], S))
        push!(r, R[(s[end-1], a[end], s[end])])

        iter += 1
        iter > max_iter && break
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

# TD prediction
function decay_eligibility_trace!(E, γ, λ, N)
    for s in N
        E[s] = γ * λ * E[s]
    end
end

function TD_λ_prediction(λ; max_iter = 100)
    Vtd = Dict([s => 0.0 for s in S]) # value function approximation
    Ntd = Dict([s => 0 for s in S])   # counter

    α = 1.0 # learning rate

    for i = 1:max_iter
        s = rand(N) # random initial state
        Etd = Dict([s => 0.0 for s in S])
        while s ∉ T
            Ntd[s] += 1
            decay_eligibility_trace!(Etd, γ, λ, N)
            Etd[s] += 1.0

            a = Π[s]
            t = sample_next_state(s, a, S)
            r = R[(s, a, t)]

            # TD update
            δ = r + γ * Vtd[t] - Vtd[s] # TD error

            # update over all non-terminal states
            for ss in N
                Vtd[ss] = Vtd[ss] + α * δ * Etd[ss]
            end

            # update state
            s = t
        end
    end
    return Vtd
end

Vtd = TD_λ_prediction(1.0; max_iter = 1000)

Vmatrix_td = zeros(8, 8)

for s in S
    Vmatrix_td[s.x + 1, s.y + 1] = Vtd[s]
end
@show Vmatrix_td

# table features
function features(s)
    x = zeros(S_dim)

    for (i, si) in enumerate(S)
        if s == si
            x[i] = 1.0
        end
    end

    return x
end

# TD prediction
function TD_λ_linear_approx_prediction(λ; max_iter = 100)
    Vtd = Dict([s => 0.0 for s in S]) # value function approximation
    w = zeros(S_dim) # linear value function approximation

    α = 1.0e-1 # learning rate
    for i = 1:max_iter
        s = rand(N) # random initial state
        E = zeros(S_dim) # initialize eligibility trace
        while s ∉ T
            x = features(s)
            # @show Vw = w' * x
            E .= γ * λ * E + x

            a = Π[s]
            t = sample_next_state(s, a, S)
            r = R[(s, a, t)]

            # δ = r + γ * V[t] - w' * x
            δ = r + γ * w' * features(t) - w' * x
            w .+= α * δ * E
            s = t
        end
    end

    for s in S
        Vtd[s] = w' * features(s)
    end

    return Vtd, w
end

Vtd_approx, w = TD_λ_linear_approx_prediction(1.0, max_iter = 10000)

Vmatrix_td_approx = zeros(8, 8)

for s in S
    Vmatrix_td_approx[s.x + 1, s.y + 1] = Vtd_approx[s]
end

@show Vmatrix_td_approx
