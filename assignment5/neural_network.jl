using LinearAlgebra, Distributions, Plots, ForwardDiff

# linear regression
N = 100
t = range(0.0, stop = 2 * 2 * pi, length = N)
y = cos.(t)
plot(t, y, color = :blue, label = "original data")

# additive Gaussian noise
ŷ = y .+ 0.1 * randn(N)
scatter!(t, ŷ, color = :black, label = "noisy data")

z = vcat([[tt; 1.0]' for tt in t]...)
θ = 1.0 * (z' * z) \ (z' * ŷ)

# y_linear = [θ[1] * tt + θ[2] for tt in t]
# plot!(t, y_linear, color = :red, label = "linear regression",
#     legend = :topleft)

# neural network
function nn(θ, x)
    w1 = θ[1:5]
    b1 = θ[6:10]
    w2 = reshape(θ[11:35], 5, 5)
    b2 = θ[36:40]
    w3 = θ[41:45]
    b3 = θ[46]

    z1 = tanh.(w1 * x + b1) # relu
    z2 = tanh.(w2 * z1 + b2)
    z3 = w3' * z2 + b3
    return z3
end

# mean squared error
function obj(θ, x_data, y_data)
    J = 0.0
    for (i, x) in enumerate(x_data)
        J += (nn(θ, x) - y_data[i])^2.0
    end
    return J / length(x_data)
end

θ = randn(46)
nn(θ, t[1])
obj(θ, t, ŷ)

function train(x_data, y_data)
    k = 46
    iter = 10000
    θ = 0.1 * randn(k)

    # Adam
    α = 0.01
    β1 = 0.9
    β2 = 0.999
    ϵ = 10.0^(-8.0)
    m = zeros(k)
    v = zeros(k)

    for i = 1:iter
        J = obj(θ, x_data, y_data)
        println("iter $i")
        println("cost $J")

        _obj(z) = obj(z, x_data, y_data)
        g = ForwardDiff.gradient(_obj, θ)
		m = β1 .* m + (1.0 - β1) .* g
		v = β2 .* v + (1.0 - β2) .* (g.^2.0)
		m̂ = m ./ (1.0 - β1^i)
		v̂ = v ./ (1.0 - β2^i)
		θ .-= α * m̂ ./ (sqrt.(v̂) .+ ϵ)
    end

    return θ
end

θ = train(t, ŷ)

y_nn = zeros(N)
for (i, tt) in enumerate(t)
    y_nn[i] = nn(θ, tt)
end

plot!(t, y_nn, color = :green, label = "neural network",
    legend = :topleft)
