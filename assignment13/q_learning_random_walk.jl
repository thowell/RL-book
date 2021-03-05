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
for s in S
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
Π
# sample random state
function sample_next_state(s, a, S)
    probs = Float64[]
    for t in S
        push!(probs, P[(s, a, t)])
    end
    sample(S, Weights(probs))
end

function ϵ_greedy_policy(Q, s, ϵ)
    if rand(1)[1] <= 1.0 - ϵ
        return argmax(Dict([a => Q[(s, a)] for a in A[s]]))
    else
        println("random action!")
        return rand(A[s])
    end
end

# Sarsa
function qlearning(; iter = 100)
    Qql = Dict([(s, a) => 0.0 for s in S for a in A[s]])
    # Nsa = Dict([(s, a) => 0.0 for s in N for a in A[s]])
    println("\nQ learning")

    α = 1.0

    for k = 1:iter
        k % (iter / 100) == 0 && println("iter: $k")

        ϵ = 1.0 / k
        s = rand(N)
        while s ∉ T
            # @show s
            # @show a
            a = ϵ_greedy_policy(Qql, s, ϵ)
            t = sample_next_state(s, a, S)
            r = R[(s, a, t)]

            # update Q function
            Qmax = Qql[(t, argmax(Dict([a => Qql[(t, a)] for a in A[t]])))]
            Qql[(s, a)] += (α / k) * (r + γ * Qmax - Qql[(s, a)])
            # Qsa[(s, a)] += (α / k) * (V[s] - Qsa[(s, a)])

            # step
            s = t
        end
    end

    return Qql
end

Qql = qlearning(iter = 1000000)
Vql = Dict([s => (s ∈ T ? 0.0 : Qql[(s, argmax(Dict([a => Qql[(s, a)] for a in A[s]])))]) for s in S])
Πql = Dict()
for s in N
    Πql[s] = argmax(Dict([a => Qql[(s, a)] for a in A[s]]))
end

Vmatrix_ql = zeros(B1 + 1, B2 + 1)

for s in S
    Vmatrix_ql[s.x + 1, s.y + 1] = Vql[s]
end
@show Vmatrix_ql
