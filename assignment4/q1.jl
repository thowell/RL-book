using LinearAlgebra

# states
S = [:s1, :s2, :s3]
T = [:s3]
N = setdiff(S, T)

# actions
A = [:a1, :a2]

# state transition probability function
P = Dict()

push!(P, (:s1, :a1, :s1) => 0.2)
push!(P, (:s1, :a1, :s2) => 0.6)
push!(P, (:s1, :a1, :s3) => 0.2)

push!(P, (:s1, :a2, :s1) => 0.1)
push!(P, (:s1, :a2, :s2) => 0.2)
push!(P, (:s1, :a2, :s3) => 0.7)

push!(P, (:s2, :a1, :s1) => 0.3)
push!(P, (:s2, :a1, :s2) => 0.3)
push!(P, (:s2, :a1, :s3) => 0.4)

push!(P, (:s2, :a2, :s1) => 0.5)
push!(P, (:s2, :a2, :s2) => 0.3)
push!(P, (:s2, :a2, :s3) => 0.2)

# reward function
R = Dict()

push!(R, (:s1, :a1) => 8.0)
push!(R, (:s1, :a2) => 10.0)

push!(R, (:s2, :a1) => 1.0)
push!(R, (:s2, :a2) => -1.0)

# discount factor
γ = 1.0

# value function
V = Dict()

for s in S
    if s in N
        push!(V, s => maximum([R[s, a] for a in A]))
    else # terminal state
        push!(V, s => 0.0)
    end
end

# value iteration
include("value_iteration.jl")
V_new, Π = value_iteration(S, N, A, R, P, V, γ = γ, iter = 10)
