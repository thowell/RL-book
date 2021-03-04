function value_iteration(S, A, P, R;
    V = Dict([s => 0.0 for s in S]),
    T = [],
    N = setdiff(S, T),
    γ = 1.0, iter = 10, verbose = false, tol = 1.0e-12)

    Q = Dict([s => Dict([a => 0.0 for a in A[s]]) for s in N])
    V_new = Dict()
    Π = Dict()
    for i = 1:iter
        println("iter: $i")
        for s in S
            if s in N
                for a in A[s]
                    Q[s][a] = sum([P[(s, a, t)] * (R[(s, a, t)] + γ * V[t]) for t in S])
                end
                if verbose
                    @show Q[s]
                end
                Π[s] = argmax(Q[s])
                V_new[s] = Q[s][Π[s]]
            else
                V_new[s] = V[s]
            end
        end

        err = 0.0
        for s in N
            # @show V[s] - V_new[s]
            err += (V[s] - V_new[s])^2.0
        end

        V = copy(V_new)
        V_new = Dict()

        if err < tol
            verbose && println("solve in $i iterations (tol = $tol)")
            return V, Π
        else
            verbose && println("error: $err")
        end
    end
    return V, Π
end

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
