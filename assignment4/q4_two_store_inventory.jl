# Two-store Inventory
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

struct TwoStoreState
    α1::Int
    β1::Int
    α2::Int
    β2::Int
end

# problem
C1 = 1  # capacity
C2 = 1
λ1 = 1.0 # mean
λ2 = 1.0

# s1 = InventoryState(1, 1)
# s2 = InventoryState(1, 2)
# s3 = InventoryState(1, 1)
# s1 == s3

# states
S = []
for α1 = 0:C1
    for β1 = 0:C1
        for α2 = 0:C2
            for β2 = 0:C2
                if α1 + β1 <= C1 && α2 + β2 <= C2
                    push!(S, TwoStoreState(α1, β1, α2, β2))
                end
            end
        end
    end
end

struct TwoStoreAction
    a1::Int
    a2::Int
    d::Int
end

# actions
A = [TwoStoreAction(i, j, k) for i = 0:C1 for j = 0:C2 for k = min(C1, C2)]
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
