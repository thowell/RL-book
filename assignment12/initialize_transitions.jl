function initialize_transitions(S, A; N = S, init = 0.0)
    D = Dict()
    for s in N
        for a in A[s]
            for t in S
                push!(D, (s, a, t) => init)
            end
        end
    end
    return D
end
