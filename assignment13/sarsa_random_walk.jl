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
function sarsa(; iter = 100)
    Qsa = Dict([(s, a) => 0.0 for s in S for a in A[s]])
    # Nsa = Dict([(s, a) => 0.0 for s in N for a in A[s]])
    println("\nSarsa")

    α = 1.0

    for k = 1:iter
        k % (iter / 100) == 0 && println("iter: $k")

        ϵ = 1.0 / k
        s = rand(N)
        a = ϵ_greedy_policy(Qsa, s, ϵ)
        while s ∉ T
            # @show s
            # @show a

            t = sample_next_state(s, a, S)
            r = R[(s, a, t)]
            b = ϵ_greedy_policy(Qsa, t, ϵ)

            # update Q function
            Qsa[(s, a)] += (α / k) * (r + γ * Qsa[(t, b)] - Qsa[(s, a)])
            # Qsa[(s, a)] += (α / k) * (V[s] - Qsa[(s, a)])

            # step
            s = t
            a = b
        end
    end

    return Qsa
end

Qsa = sarsa(iter = 100000)
Vsa = Dict([s => (s ∈ T ? 0.0 : Qsa[(s, argmax(Dict([a => Qsa[(s, a)] for a in A[s]])))]) for s in S])
Πsa = Dict()
for s in N
    Πsa[s] = argmax(Dict([a => Qsa[(s, a)] for a in A[s]]))
end
Πsa

Vmatrix_sa = zeros(B1 + 1, B2 + 1)

for s in S
    Vmatrix_sa[s.x + 1, s.y + 1] = Vsa[s]
end
@show Vmatrix_sa
