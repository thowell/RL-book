# Frog escape
include("initialize.jl")
include("value_iteration.jl")
include("policy_iteration.jl")

# number of lilypads
n = 3

# states
S = [Symbol("s$i") for i = 0:n]
T = [Symbol("s0"), Symbol("s$n")]
N = setdiff(S, T)

# actions
A = [:A, :B]

# state transition probability function
P = initialize_transitions(S, A)

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
R = initialize_transitions(S, A)

for i = 0:n
    for a in A
        R[(Symbol("s$i"), a, Symbol("s$n"))] = 1.0 * P[(Symbol("s$i"), a, Symbol("s$n"))]
    end
end

# value iteration
γ = 1.0
V_vi, Π_vi = value_iteration(S, A, P, R, T = T, γ = γ, iter = 100)

# policy iteration

# initialize random policy
V = Dict([s => 0.0 for s in S])
Π = Dict(s => rand(A) for s in N)

# V = policy_evaluation(S, Π, P, R, T = T, V = V, γ = γ, iter = 2)
# Π = policy_improvement(S, A, P, R, V = V, T = T, γ = γ)

Π = policy_iteration(S, A, P, R, V = V, Π = Π, T = T, γ = γ,
    pi_iter = 100, pe_iter = 10, pe_tol = 1.0e-6)

# complexity
