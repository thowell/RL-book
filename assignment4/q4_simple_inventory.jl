# Inventory
include("initialize.jl")
include("value_iteration.jl")
include("policy_iteration.jl")

# Poisson
function f(i, λ)
    (exp(-λ) * λ^i) / factorial(i)
end

function F(i, λ)
    sum([f(j, λ) for j = 0:i])
end

struct InventoryState
    α::Int
    β::Int
end

available_actions(α, β, C) = [Symbol("a$i") for i = 0:max(C - (α + β), 0)]

# problem
C = 2  # capacity
λ = 1.0 # mean

# s1 = InventoryState(1, 1)
# s2 = InventoryState(1, 2)
# s3 = InventoryState(1, 1)
# s1 == s3

# states
S = []
for α = 0:C
    for β = 0:C
        if α + β <= C
            push!(S, InventoryState(α, β))
        end
    end
end

# actions
A = [Symbol("a$i") for i = 0:C]
get_num(a) = parse(Int64, String(a)[2])

# transitions
P = initialize_transitions(S, A)
for s in S
    for a in A
        for i = 0:(s.α + s.β - 1)
            P[(s, a, InventoryState(s.α + s.β - i, get_num(a)))] = get_num(a) > (C - (s.α + s.β)) ? 0.0 : f(i, λ)
        end
        P[(s, a, InventoryState(0, get_num(a)))] = 1.0 - F(s.α + s.β - 1, λ)
    end
end

# rewards
h = 100.0
p = 5.0
k = 1.0

R = initialize_transitions(S, A)

# RL book pg. 93
for s in S
    for a in A
        for i = 0:(s.α + s.β - 1)
            R[(s, a, InventoryState(s.α + s.β - i, get_num(a)))] = get_num(a) > (C - (s.α + s.β)) ? 0.0 : -h * s.α -k * get_num(a)
        end

        prob = 1.0 - F(s.α + s.β - 1, λ)
        R[(s, a, InventoryState(0, get_num(a)))] = (-h * s.α
            - p * λ * prob
            - (s.α + s.β) * prob
            -k * get_num(a))
    end
end

γ = 1.0
V_vi, Π_vi = value_iteration(S, A, P, R, γ = γ, iter = 10)

Π = Dict(s => rand(A) for s in S)
V = Dict([s => 0.0 for s in S])
Π = policy_iteration(S, A, P, R, V = V, Π = Π, γ = γ,
    pi_iter = 100, pe_iter = 10, pe_tol = 1.0e-6)
