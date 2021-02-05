function value_iteration(S, A, P, R;
    V = Dict([s => 0.0 for s in S]),
    T = [],
    N = setdiff(S, T),
    γ = 1.0, iter = 10)

    Q = Dict([s => Dict([a => 0.0 for a in A]) for s in N])
    V_new = Dict()
    Π = Dict()
    for i = 1:iter
        println("iter: $i")
        for s in S
            if s in N
                for a in A
                    Q[s][a] = R[(s, a)] + γ * sum([P[(s, a, t)] * V[t] for t in S])
                end
                @show s
                @show Q
                Π[s] = argmax(Q[s])
                V_new[s] = Q[s][Π[s]]
            else
                V_new[s] = V[s]
            end
        end
        V = V_new
        @show V
        @show Π
    end
    return V, Π
end
