using LinearAlgebra
include("initialize.jl")
include("value_iteration.jl")

# states
S = [:s1, :s2, :s3]
T = [:s3]
N = setdiff(S, T)

# actions
A = [:a1, :a2]

# state transition probability function
P = initialize_transitions(S, A)

P[(:s1, :a1, :s1)] = 0.2
P[(:s1, :a1, :s2)] = 0.6
P[(:s1, :a1, :s3)] = 0.2

P[(:s1, :a2, :s1)] = 0.1
P[(:s1, :a2, :s2)] = 0.2
P[(:s1, :a2, :s3)] = 0.7

P[(:s2, :a1, :s1)] = 0.3
P[(:s2, :a1, :s2)] = 0.3
P[(:s2, :a1, :s3)] = 0.4

P[(:s2, :a2, :s1)] = 0.5
P[(:s2, :a2, :s2)] = 0.3
P[(:s2, :a2, :s3)] = 0.2

# reward function
R = initialize_transitions(S, A)

for s in S
    R[(:s1, :a1, s)] = 8.0
    R[(:s1, :a2, s)] = 10.0

    R[(:s2, :a1, s)] = 1.0
    R[(:s2, :a2, s)] = -1.0
end

# discount factor
γ = 1.0

# value function
V = Dict()
for s in S
    if s in N
        push!(V, s => maximum([sum([P[(s, a, t)] * R[(s, a, t)] for t in S]) for a in A]))
    else # terminal state
        push!(V, s => 0.0)
    end
end

# value iteration
V_new, Π = value_iteration(S, A, P, R, T = T, V = V, γ = γ, iter = 3)
