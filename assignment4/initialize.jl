function initialize_transitions(S, A)
    D = Dict()
    for s in S
        for a in A
            for t in S
                push!(D, (s, a, t) => 0.0)
            end
        end
    end
    return D
end
