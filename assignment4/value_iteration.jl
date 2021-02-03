function value_iteration(S, N, A, R, P, V; γ = 1.0, iter = 2)
    V_new = Dict()
    Π = Dict()
    for i = 1:iter
        println("iter: $i")
        for s in S
            if s in N
                Q = [(R[(s, a)]
                        + γ * sum([P[(s, a, t)] * V[t] for t in S])) for (i, a) in enumerate(A)]
                @show s
                @show Q
                V_new[s] = maximum(Q)
                Π[s] = argmax(Q)
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
