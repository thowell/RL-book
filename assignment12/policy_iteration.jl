function policy_evaluation(S, Π, P, R;
    T = [], N = setdiff(S, T),
    V = Dict([s => 0.0 for s in S]),
    γ = 1.0, iter = 10, tol = 1.0e-6)

    V_new = Dict()
    for i = 1:iter
        println("iter: $i")
        for s in S
            if s in setdiff(S, T)
                a = Π[s]
                V_new[s] = sum([P[(s, a, t)] * (R[(s, a, t)] + γ * V[t]) for t in S])
            else
                V_new[s] = 0.0
            end
        end

        # compute value function difference
        err = 0.0
        for k in keys(V)
            err += abs(V[k] - V_new[k])
        end
        println("V difference: $err")
        err < tol ? (return V_new) : (V = V_new)
    end
    return V
end

function policy_improvement(S, A, P, R;
    V = Dict([s => 0.0 for s in S]),
    T = [], N = setdiff(S, T),
    γ = 1.0)

    Q = Dict([s => Dict([a => 0.0 for a in A]) for s in N])
    Π = Dict()

    for s in N
        for a in A
            Q[s][a] = sum([P[(s, a, t)] * (R[(s, a, t)] + γ * V[t]) for t in S])
        end
        @show s
        @show Q
        Π[s] = argmax(Q[s])
    end

    return Π
end

function policy_iteration(S, A, P, R;
    V = Dict([s => 0.0 for s in S]),
    Π = Dict(s => rand(A) for s in N),
    T = [],
    γ = 1.0, pi_iter = 10, pe_iter = 10, pe_tol = 1.0e-6)

    println("policy iteration")
    for i = 1:pi_iter
        println("iter: $i")

        println("   policy_evaluation")
        V = policy_evaluation(S, Π, P, R;
                T = T, V = V,
                γ = γ, iter = pe_iter, tol = pe_tol)

        println("   policy improvement")
        Π = policy_improvement(S, A, P, R;
            V = V,
            T = T, γ = 1.0)
    end

    @show V
    @show Π
    return V, Π
end
