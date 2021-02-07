# Frog escape

# number of lilypads
n = 6

# states
S = [Symbol("s$i") for i = 0:n]
T = [Symbol("s0"), Symbol("s$n")]
N = setdiff(S, T)

# actions
A = [:A, :B]

# state transition probability function
P = Dict()

for i = 0:n
    for a in A
        for j = 0:n
            push!(P, (Symbol("s$i"), a, Symbol("s$j")) => 0.0)
        end
    end
end

for i = 1:n-1
    P[(Symbol("s$i"), :A, Symbol("s$(i-1)"))] = i / n
    P[(Symbol("s$i"), :A, Symbol("s$(i+1)"))] = (n - i) / n

    for j = 0:n
        if j != i
            P[(Symbol("s$i"), :B, Symbol("s$j"))] = 1 / n
        end
    end
end

# reward function
R = Dict()
for i = 0:n
    for a in A
        R[(Symbol("s$i"), a)] = 1.0 * P[(Symbol("s$i"), a, Symbol("s$n"))]
    end
end

# value iteration
γ = 1.0
include("value_iteration.jl")
V_vi, Π_vi = value_iteration(S, A, P, R, T = T, γ = γ, iter = 100)

# policy iteration
include("policy_iteration.jl")

# initialize random policy
V = Dict([s => 0.0 for s in S])
Π = Dict(s => rand(A) for s in N)

V = policy_evaluation(S, Π, P, R, T = T, V = V, γ = γ, iter = 2)
Π = policy_improvement(S, A, P, R, V = V, T = T, γ = γ)

Π = policy_iteration(S, A, P, R, V = V, Π = Π, T = T, γ = γ,
    pi_iter = 10, pe_iter = 10, pe_tol = 1.0e-6)

# complexity
