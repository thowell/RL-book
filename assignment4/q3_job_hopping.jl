# Job hopping
n = 5

# states
S = [[Symbol("e$i") for i = 1:n]..., [Symbol("u$i") for i = 1:n]...]
T = []
N = setdiff(S, T)

# actions
A = [:accept, :reject]

# state transition probability function
α = 0.9
p = rand(n)
p ./= sum(p) # normalize
# sum(p)
P = Dict()

for s in S
    for a in A
        for t in S
            push!(P, (s, a, t) => 0.0)
        end
    end
end

for i = 1:n
    # same job
    P[(Symbol("e$i"), :accept, Symbol("e$i"))] = 1.0 - α
    P[(Symbol("e$i"), :reject, Symbol("e$i"))] = 1.0 - α

    # lose job
    for j = 1:n
        P[(Symbol("e$i"), :accept, Symbol("u$j"))] = α * p[j]
        P[(Symbol("e$i"), :reject, Symbol("u$j"))] = α * p[j]
    end

    # accept job
    P[(Symbol("u$i"), :accept, Symbol("e$i"))] = 1.0

    # reject job
    for j = 1:n
        P[(Symbol("u$i"), :reject, Symbol("u$j"))] = p[j]
    end
end

# reward function
w0 = 1.0 * rand(n)[1] # unemployed reward
w = rand(n)     # employed rewards
R = Dict()
for i = 1:n
    for a in A
        R[(Symbol("e$i"), a)] = w[i]
        R[(Symbol("u$i"), a)] = w0
    end
end

# custom value iteration
γ = 1.0
V_vi, Π_vi = value_iteration_custom(S, A, P, R, T = T, γ = γ, iter = 100)

function value_iteration_custom(S, A, P, R;
    V = Dict([s => 0.0 for s in S]),
    T = [],
    N = setdiff(S, T),
    γ = 1.0, iter = 10)

    Q = Dict([s => Dict([a => 0.0 for a in A]) for s in N])
    V_new = Dict()
    Π = Dict()
    for i = 1:iter
        println("iter: $i")
        for i = 1:n
            # employed
            V_new[Symbol("e$i")] = w[i] + γ * (α * sum([p[j] * V[Symbol("u$j")] for j = 1:n]) + (1.0 - α) * V[Symbol("e$i")])

            # unemployed
            Q[Symbol("u$i")][:accept] = w0 + γ * V[Symbol("e$i")]
            Q[Symbol("u$i")][:reject] = w0 + γ * sum([p[j] * V[Symbol("u$j")] for j = 1:n])

            # @show s
            @show Q
            Π[Symbol("u$i")] = argmax(Q[Symbol("u$i")])
            V_new[Symbol("u$i")] = Q[Symbol("u$i")][Π[Symbol("u$i")]]
        end
        V = V_new
        @show V
        @show Π
    end
    return V, Π
end

v
