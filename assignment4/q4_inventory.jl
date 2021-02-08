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

available_actions(α, β, C) = [a for a = 0:max(C - (α + β), 0)]

# problem
C = 2   # capacity
λ = 1.0 # mean

s1 = InventoryState(1, 1)
s2 = InventoryState(1, 2)
s3 = InventoryState(1, 1)
s1 == s3

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
A = [a for a = 0:C]

# transitions
P = Dict()

for s in S
    for a in available_actions(s.α, s.β, C)
        for i = 0:(s.α + s.β - 1)
            P[(s, a, InventoryState(s.α + s.β - i, a))] = f(i, λ)
        end
        P[(s, a, InventoryState(0, a))] = 1.0 - F(s.α + s.β - 1, λ)
    end
end

# rewards
h = 1.0
p = 1.0
R = Dict()

for s in S
    for a in available_actions(s.α, s.β, C)
        for i = 0:(s.α + s.β - 1)
            R[(s, a, InventoryState(s.α + s.β - i, a))] = -h * s.α
        end
        prob = 1.0 - F(s.α + s.β - 1, λ)
        R[(s, a, InventoryState(0, a))] = (-h * s.α
            -p * (prob * (λ - (s.α + s.β)) + (s.α + s.β) * f()
    end
end
