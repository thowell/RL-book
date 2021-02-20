include("initialize_transitions.jl")
include("value_iteration.jl")
using Distributions
prob = Poisson(1.0)

# problem formulation
H = 3
W = 10
α = 0.08
β = 0.82
γ = 0.95

# State
struct CareerState
    w_employer::Int
    w_offer::Int
end
S = [CareerState(i, j) for i = 1:W for j = 1:W]
T = []

# Actions
struct CareerAction
    l::Int
    s::Int
end
A_set = [CareerAction(i, j) for i = 0:H for j = 0:H if i + j <= H]
A = Dict([s => A_set for s in S])

# Transitions
P = initialize_transitions(S, A)

for s in S
    wjob = max(s.w_employer, s.w_offer)

    for a in A[s]
        prob = 0.0
        prob_offer = wjob < W ? β * a.s / H : 0.0
        # prob_offer = true ? β * a.s / H : 0.0

        # same job
        for i = 0:(W - wjob - 1)
            t = CareerState(wjob + i, wjob)
            prob_wage = pdf(Poisson(α * a.l), i)

            P[(s, a, t)] = prob_wage * (1.0 - prob_offer)
            prob += prob_wage * (1.0 - prob_offer)
        end
        t = CareerState(W, wjob)
        prob_wage = 1 - cdf(Poisson(α * a.l), W - wjob - 1)
        P[(s, a, t)] = prob_wage * (1.0 - prob_offer)
        prob += prob_wage * (1.0 - prob_offer)

        # new job
        for i = 0:(W - wjob - 1)
            t = CareerState(wjob + i, wjob + 1)
            prob_wage = pdf(Poisson(α * a.l), i)
            P[(s, a, t)] = prob_wage * prob_offer
            prob += prob_wage * prob_offer
        end
        t = CareerState(W, wjob + 1)
        prob_wage = 1 - cdf(Poisson(α * a.l), W - wjob - 1)
        P[(s, a, t)] = prob_wage * prob_offer
        prob += prob_wage * prob_offer

        println("total")
        println(prob)
    end
end

# for s in S
#     for a in A[s]
#         prob = 0
#         prob_offer = β * a.s / H
#         P[(s, a, CareerState(min(s.w + 1, W)))] = prob_offer
#         P[(s, a, CareerState(s.w))] = (1.0 - prob_offer)
#     end
# end

# Rewards
R = initialize_transitions(S, A)
for s in S
    for a in A[s]
        for t in S
            R[(s, a, t)] = max(s.w_employer, s.w_offer) * (H - a.l - a.s)
        end
    end
end

# Value iteration# value iteration
V, Π = value_iteration(S, A, P, R;
    V = Dict([s => 1.0 for s in S]),
    T = [],
    N = S,
    γ = γ, iter = 1000, verbose = false)

println("value function")
pretty_print2(V)
println()
println("optimal policy")
pretty_print2(Π)
