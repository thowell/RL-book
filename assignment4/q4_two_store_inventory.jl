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
λ1 = 0.5 # mean
λ2 = 0.5

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
    d1::Int
    d2::Int
end

# actions
A = [TwoStoreAction(i, j, k, l) for i = 0:C1 for j = 0:C2 for k = 0:min(C1, C2) for l = 0:min(C1, C2)]
get_num(a) = parse(Int64, String(a)[2])

# transitions
P = initialize_transitions(S, A)
for s in S
    for a in A
        for i = 0:(s.α1 + s.β1)
            for j = 0:(s.α2 + s.β2)
                # check valid actions
                a.d1 > s.α1 + s.β1 - i && continue
                a.d2 > s.α2 + s.β2 - j && continue
                a.a1 > C1 - (s.α1 + s.β1 - i - a.d1) && continue
                a.a2 > C2 - (s.α2 + s.β2 - j - a.d2) && continue

                α1 = s.α1 + s.β1 - i - a.d1
                α2 = s.α2 + s.β2 - j - a.d2
                p1 = α1 > 0 ? f(i, λ1) : (1.0 - F(s.α1 + s.β1 - 1, λ1))
                p2 = α2 > 0 ? f(i, λ2) : (1.0 - F(s.α2 + s.β2 - 1, λ2))

                # next state
                t = TwoStoreState(α1, a.a1, α2, a.a2)
                @assert t in S
                P[(s, a, t)] = p1 * p2
            end
        end
    end
end

# rewards
h1 = 100.0
h2 = 200.0
p1 = 5.0
p2 = 1.0
k1 = 0.1
k2 = 1.0

R = initialize_transitions(S, A)
for s in S
    for a in A
        for i = 0:(s.α1 + s.β1)
            for j = 0:(s.α2 + s.β2)
                # check valid actions
                a.d1 > s.α1 + s.β1 - i && continue
                a.d2 > s.α2 + s.β2 - j && continue
                a.a1 > C1 - (s.α1 + s.β1 - i - a.d1) && continue
                a.a2 > C2 - (s.α2 + s.β2 - j - a.d2) && continue

                α1 = s.α1 + s.β1 - i - a.d1
                α2 = s.α2 + s.β2 - j - a.d2

                r1 = -h1 * s.α1 - k1 * a.a1 - k2 * a.d1 + (α1 > 0 ? 0.0 : (-p1 * λ1 + s.α1 + s.β1) * (1.0 - F(s.α1 + s.β1 - 1, λ1)))
                r2 = -h2 * s.α2 - k2 * a.a2 - k2 * a.d2 + (α2 > 0 ? 0.0 : (-p2 * λ2 + s.α2 + s.β2) * (1.0 - F(s.α2 + s.β2 - 1, λ2)))

                # next state
                t = TwoStoreState(α1, a.a1, α2, a.a2)
                @assert t in S

                R[(s, a, t)] = r1 + r2
            end
        end
    end
end

γ = 1.0

# solve with value iteration
V_vi, Π_vi = value_iteration(S, A, P, R, γ = γ, iter = 10)
@show V_vi

# Π = Dict(s => rand(A) for s in S)
# V = Dict([s => 0.0 for s in S])
# Π = policy_iteration(S, A, P, R, V = V, Π = Π, γ = γ,
#     pi_iter = 100, pe_iter = 10, pe_tol = 1.0e-6)

# printing method below stolen from stackexchange...
function pretty_print2(d::Dict, pre=1)
    todo = Vector{Tuple}()
    for (k,v) in d
        if typeof(v) <: Dict
            push!(todo, (k,v))
        else
            println(join(fill(" ", pre)) * "$(repr(k)) => $(repr(v))")
        end
    end

    for (k,d) in todo
        s = "$(repr(k)) => "
        println(join(fill(" ", pre)) * s)
        pretty_print2(d, pre+1+length(s))
    end
    nothing
end

pretty_print2(V_vi)
