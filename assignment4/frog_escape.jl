# Frog escape

# number of lilypads
n = 6

# states
S = [i for i = 0:n]
T = [0, n]
N = setdiff(S, T)

# actions
A = [:A, :B]

# state transition probability function
P = Dict()

for i = 0:n
    for a in A
        for j = 0:n
            push!(P, (i, a, j) => 0.0)
        end
    end
end

for i = 1:n-1
    P[(i, :A, i)] = 0.0
    P[(i, :A, i - 1)] = i / n
    P[(i, :A, i + 1)] = (n - i) / n

    for j = 0:n
        if j != i
            P[(i, :B, j)] = 1 / n
        end
    end
end

# reward function
R = Dict()
for i = 0:n
    for a in A
        R[(i, a)] = 0.0
    end
end

γ = 1.0

# value function
V = Dict()

for s in S
    if s in N
        push!(V, s => maximum([R[s, a] for a in A]))
    elseif s == n # terminal state
        push!(V, s => 1.0)
    else
        push!(V, s => 0.0)
    end
end
V
# value iteration
include("value_iteration.jl")
V_new, Π = value_iteration(S, N, A, R, P, V, γ = γ, iter = 1000)
V_new
