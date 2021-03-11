using LinearAlgebra

# experience
D = [[(:A, 2.), (:A, 6.), (:B, 1.), (:B, 2.), (:C, nothing)],
     [(:A, 3.), (:B, 2.), (:A, 4.), (:B, 2.), (:B, 0.), (:C, nothing)],
     [(:B, 3.), (:B, 6.), (:A, 1.), (:B, 1.), (:C, nothing)],
     [(:A, 0.), (:B, 2.), (:A, 4.), (:B, 4.), (:B, 2.), (:B, 3.), (:C, nothing)],
     [(:B, 8.), (:B, 2.), (:C, nothing)]]

S = [:A, :B, :C]
T = [:C]
N = setdiff(S, T)

A = Dict([s => [:nothing] for s in S])

# compute empirical P, R functions
include("initialize_transitions.jl")
Nc = Dict([(s, t) => 0 for s in S for t in S])
Nr = Dict([(s, t) => 0.0 for s in S for t in S])
Ntol = 0
for d in D
    for (t, e) in enumerate(d)
        if e[1] ∉ T
            Nc[(e[1], d[t+1][1])] += 1
            Nr[(e[1], d[t+1][1])] += e[2]
            Ntol += 1
        end
    end
end

P = Dict([(s, a, t) => 0.0 for s in S for a in A[s] for t in S])
R = Dict([(s, a, t) => 0.0 for s in S for a in A[s] for t in S])
for s in N
    for a in A[s]
        @show P_norm = sum([Nc[(s, t)] for t in S])
        for t in S
            P[(s, a, t)] = Nc[(s, t)] / P_norm
            R[(s, a, t)] = Nr[(s, t)] / max(1, Nc[(s, t)])
        end
    end
end

# MRP value iteration
γ = 1.0
include("value_iteration.jl")
V, Π = value_iteration(S, A, P, R, iter = 1000, γ = γ)
pretty_print2(V)
pretty_print2(P)
pretty_print2(R)

# Monte Carlo
function discounted_return(r, γ)
    G = 0.0
    for t = 0:length(r)-1
        G += r[t+1] * γ^t
    end
    return G
end

function monte_carlo(D; max_iter = 1)
    Vmc = Dict([s => 0.0 for s in S]) # value function approximation
    Nmc = Dict([s => 0 for s in S])   # counter
    Smc = Dict([s => 0.0 for s in S]) # total sample return

    println("\nMonte Carlo Prediction")

    for i = 1:max_iter
        i % (max_iter / 10) == 0 && println("iter: $i")
        for d in D
            for (t, e) in enumerate(d)
                Nmc[e[1]] += 1
                Smc[e[1]] += discounted_return([f[2] for f in d[t:end-1]], γ)
            end
        end
    end

    # compute value function approximation
    for s in N
        Vmc[s] = Smc[s] / Nmc[s]
    end

    return Vmc, Nmc, Smc
end

Vmc, Nmc, Smc = monte_carlo(D, max_iter = 1000000)

# TD(0)
function TD_prediction(D; max_iter = 1)
    Vtd = Dict([s => 0.0 for s in S]) # value function approximation
    Ntd = Dict([s => 0 for s in S])   # counter

    println("\nTD(0)")
    α = 1.0e-3 # learning rate
    for i = 1:max_iter
        i % (max_iter / 10) == 0 && println("iter: $i")
        for d in D
            for (t, e) in enumerate(d)
                s = e[1]
                s ∈ T && continue
                r = e[2]
                t = d[t+1][1]
                Ntd[s] += 1

                # TD update
                # Vtd[s] = Vtd[s] + (α / (α + Ntd[s])) * (r + γ * Vtd[t] - Vtd[s])
                ᾱ = α * (Ntd[s] / 30 + 1) ^ (-0.5)
                Vtd[s] = Vtd[s] +  ᾱ * (r + γ * Vtd[t] - Vtd[s])
            end
        end
    end

    return Vtd, Ntd
end

Vtd, Ntd = TD_prediction(D, max_iter = 1000000)

# least-squares TD
num_features = length(N)

function features(s)
    x = zeros(num_features)
    id = 1
    for _s in N
        if s == _s
            x[id] = 1.0
            return x
        else
            id += 1
        end
    end
    # @warn "invalid state pair"
    return x
end

function lstd(D)
    w = zeros(num_features)
    Als = zeros(num_features, num_features)
    bls = zeros(num_features)

    println("Least-Squares Temporal-Difference Prediction")

    # for k = 1:max_iter
    #     k % (max_iter / 10) == 0 && println("iter: $k")
    for d in D
        for (t, e) in enumerate(d)
            s = e[1]
            s ∈ T && continue
            r = e[2]
            t = d[t+1][1]
            Als .+= features(s) * (features(s) - γ * features(t))'
            bls .+= r * features(s)
        end
    end
    # end

    w = pinv(Als) * bls

    Vlstd = Dict([s => 0.0 for s in S])
    for s in S
        Vlstd[s] = w' * features(s)
    end

    return Vlstd, w
end

Vlstd, w = lstd(D, max_iter = 100)
pretty_print2(Vlstd)
