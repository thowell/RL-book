using StatsBase, LinearAlgebra
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

    status = true
    iter = 1
    while s[end] ∉ T
        push!(a, ϵ_greedy_policy(Π, s[end], ϵ))
        push!(s, sample_next_state(s[end], a[end], S))
        push!(r, R[(s[end-1], a[end], s[end])])

        iter += 1
        if iter > max_iter
            @warn "episode not terminated"
            status = false
            break
        end
    end
    return s, a, r, status
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
    return x
end


function lspi(; iter_batch = 1000, iter_policy = 1000)
    w = 0.1 * rand(num_features)
    Als = zeros(num_features, num_features)
    bls = zeros(num_features)
    println("Least-Squares Policy Iteration")

    Πlspi = Dict()
    for s in N
        Πlspi[s] = argmax(Dict([a => w' * features(s, a) for a in A[s]]))
    end

    for j = 1:iter_policy
        for k = 1:iter_batch
            k % 100 == 0 && println("iter: $k")
            s = rand(N)
            while s ∉ T
                a = ϵ_greedy_policy(Πlspi, s, 1 / j)
                t = sample_next_state(s, a, S)
                r = R[(s, a, t)]
                b = t ∉ T ? argmax(Dict([a => w' * features(t, a_) for a_ in A[t]])) : rand(A_set)
                Als .+= features(s, a) * (features(s, a) - γ * features(t, b))'
                bls .+= r * features(s, a)

                s = t
            end
        end

        @show rank(Als)
        @show rank(Als' * Als)

        # compute weights
        w = pinv(Als) * bls

        # update policy
        for s in N
            Πlspi[s] = argmax(Dict([a => w' * features(s, a) for a in A[s]]))
        end
    end

    Qlspi = Dict([(s, a) => 0.0 for s in S for a in A[s]])
    for s in S
        for a in A[s]
            Qlspi[(s, a)] = w' * features(s, a)
        end
    end

    return Qlspi, w
end

Qlspi, w = lspi(iter_batch = 1000, iter_policy = 10)
Vlspi = Dict([s => (s ∈ T ? 0.0 : Qlspi[(s, argmax(Dict([a => Qlspi[(s, a)] for a in A[s]])))]) for s in S])
Πlspi = Dict()
for s in N
    Πlspi[s] = argmax(Dict([a => Qlspi[(s, a)] for a in A[s]]))
end

Vmatrix_lspi = zeros(8, 8)

for s in S
    Vmatrix_lspi[s.x + 1, s.y + 1] = Vlspi[s]
end
@show Vmatrix_lspi
