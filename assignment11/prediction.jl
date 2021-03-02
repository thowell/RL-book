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
length(S)
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
γ = 1
R = initialize_transitions(S, A, N = N, init = 0.0)
for s in N
    for a in A[s]
        t = MazeState(state_action_transition(s, a)...)
        R[(s, a, t)] = -1.0
    end
end

# # option 2
# γ2 = 0.9
# R2 = initialize_transitions(S, A, N = N, init = 0.0)
#
# for t in T
#     s = MazeState(t.x, t.y + 1)
#     if s in S
#         if maze_grid[(s.x, s.y)] == :SPACE
#             a = :LEFT
#             R2[(s, a, t)] = 1.0
#         end
#     end
#
#     s = MazeState(t.x, t.y - 1)
#     if s in S
#         if maze_grid[(s.x, s.y)] == :SPACE
#             a = :RIGHT
#             R2[(s, a, t)] = 1.0
#         end
#     end
#
#     s = MazeState(t.x + 1, t.y)
#     if s in S
#         if maze_grid[(s.x, s.y)] == :SPACE
#             a = :UP
#             R2[(s, a, t)] = 1.0
#         end
#     end
#
#     s = MazeState(t.x - 1, t.y)
#     if s in S
#         if maze_grid[(s.x, s.y)] == :SPACE
#             a = :DOWN
#             R2[(s, a, t)] = 1.0
#         end
#     end
# end

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

function rollout(; max_iter = 100)
    s = [rand(N)] # random initial non-terminal state
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

discounted_return([1.0 for i = 1:5], 0.9)

function monte_carlo(; iter = 1000, mode = :every)
    Vmc = Dict([s => 0.0 for s in S]) # value function approximation
    Nmc = Dict([s => 0 for s in S])   # counter
    Smc = Dict([s => 0.0 for s in S]) # total sample return

    println("Monte Carlo Prediction ($(String(mode)) visit)")

    for i = 1:iter
        i % 100 == 0 && println("iter: $i")
        # rollout
        s, a, r = rollout()

        # update count and sample returns (every visit MC)
        s_visit = []

        for (t, st) in enumerate(s[1:end-1])
            if mode == :first
                if st ∉ s_visit
                    # println("t = $t")
                    Nmc[st] += 1
                    Smc[st] += discounted_return(r[t:end], γ)
                    push!(s_visit, st)
                end
            else
                Nmc[st] += 1
                Smc[st] += discounted_return(r[t:end], γ)
            end
        end
    end

    # compute value function approximation
    for s in N
        Vmc[s] = Smc[s] / Nmc[s]
    end

    return Vmc
end

Vmc = monte_carlo(iter = 100, mode = :every)
Vmatrix_mc = zeros(8, 8)

for s in S
    Vmatrix_mc[s.x + 1, s.y + 1] = Vmc[s]
end
@show Vmatrix_mc

# TD prediction
function TD_prediction(; max_iter = 100)
    Vtd = Dict([s => 0.0 for s in S]) # value function approximation
    Ntd = Dict([s => 0 for s in S])   # counter

    α = 1.0 # learning rate

    for i = 1:max_iter
        s = rand(N) # random initial state
        while s ∉ T
            Ntd[s] += 1
            a = Π[s]
            t = sample_next_state(s, a, S)
            r = R[(s, a, t)]

            # TD update
            Vtd[s] = Vtd[s] +  (α / (α + Ntd[s])) * (r + γ * Vtd[t] - Vtd[s])
            Vtd[s] = Vtd[s] +  α * (r + γ * Vtd[t] - Vtd[s])

            # update state
            s = t
        end
    end

    return Vtd
end

Vtd = TD_prediction(; max_iter = 10000)
Vmatrix_td = zeros(8, 8)

for s in S
    Vmatrix_td[s.x + 1, s.y + 1] = Vtd[s]
end
@show Vmatrix_td

1 / log(1 + 100)
