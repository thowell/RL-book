include("initialize_transitions.jl")
include("grid_maze.jl")
include("value_iteration.jl")

maze = zeros(8, 8)
for (k, v) in maze_grid
    if v == :SPACE || v == :GOAL
        maze[k[1] + 1, k[2] + 1] = 1
    end
end
@show maze

# states
struct MazeState
    x::Int
    y::Int
end

S = []
for (k, v) in maze_grid
    if v == :SPACE || v == :GOAL
        # println(v)
        push!(S, MazeState(k[1], k[2]))
    end
end
length(S)
T = []
for (k, v) in maze_grid
    if v == :GOAL
        # println(v)
        push!(T, MazeState(k[1], k[2]))
    end
end

N = setdiff(S, T)

# actions
function state_action_transition(s, a)
    if a == :LEFT
        return (s.x, s.y - 1)
    elseif a == :RIGHT
        return (s.x, s.y + 1)
    elseif a == :UP
        return (s.x - 1, s.y)
    elseif a == :DOWN
        return (s.x + 1, s.y)
    end
end
A_set = [:LEFT, :RIGHT, :UP, :DOWN]
A = Dict()
for s in N
    _a = []
    for a in A_set
        t = MazeState(state_action_transition(s, a)...)
        if t in S
            push!(_a, a)
        end
    end
    push!(A, s => _a)
end

# transitions
P = initialize_transitions(S, A, N = N)
for s in N
    for a in A[s]
        t = MazeState(state_action_transition(s, a)...)
        P[(s, a, t)] = 1.0
    end
end

# rewards
# option 1
γ1 = 1
R1 = initialize_transitions(S, A, N = N, init = 0.0)
for s in N
    for a in A[s]
        t = MazeState(state_action_transition(s, a)...)
        R1[(s, a, t)] = -1.0
    end
end

# option 2
γ2 = 0.9
R2 = initialize_transitions(S, A, N = N, init = 0.0)

for t in T
    s = MazeState(t.x, t.y + 1)
    if s in S
        if maze_grid[(s.x, s.y)] == :SPACE
            a = :LEFT
            R2[(s, a, t)] = 1.0
        end
    end

    s = MazeState(t.x, t.y - 1)
    if s in S
        if maze_grid[(s.x, s.y)] == :SPACE
            a = :RIGHT
            R2[(s, a, t)] = 1.0
        end
    end

    s = MazeState(t.x + 1, t.y)
    if s in S
        if maze_grid[(s.x, s.y)] == :SPACE
            a = :UP
            R2[(s, a, t)] = 1.0
        end
    end

    s = MazeState(t.x - 1, t.y)
    if s in S
        if maze_grid[(s.x, s.y)] == :SPACE
            a = :DOWN
            R2[(s, a, t)] = 1.0
        end
    end
end

# value iteration
# rewards option 1
V1, Π1 = value_iteration(S, A, P, R1;
    V = Dict([s => 0.0 for s in S]),
    T = T,
    N = setdiff(S, T),
    γ = γ1, iter = 100, verbose = false)

pretty_print2(V1)
pretty_print2(Π1)

Vmatrix = zeros(8, 8)

for s in S
    Vmatrix[s.x + 1, s.y + 1] = V1[s]
end
@show Vmatrix

# value iteration
# rewards option 2
V2, Π2 = value_iteration(S, A, P, R2;
    V = Dict([s => 0.0 for s in S]),
    T = T,
    N = setdiff(S, T),
    γ = γ2, iter = 100, verbose = false)

pretty_print2(V2)
pretty_print2(Π2)
Vmatrix = zeros(8, 8)

for s in S
    Vmatrix[s.x + 1, s.y + 1] = V2[s]
end
@show Vmatrix
