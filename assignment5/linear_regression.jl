using LinearAlgebra, Distributions, Plots

# linear regression
function f(x)
    a = 1.0
    b = 1.0

    a * x + b
end

N = 100
t = range(-1.0, stop = 1.0, length = N)
y = f.(t)
plot(t, y, color = :blue, label = "original data")

# additive Gaussian noise
ŷ = y .+ 0.5 * randn(N)
scatter!(t, ŷ, color = :black, label = "noisy data")

z = vcat([[tt; 1.0]' for tt in t]...)
θ = 1.0 * (z' * z) \ (z' * ŷ)

y_linear = [θ[1] * tt + θ[2] for tt in t]
plot!(t, y_linear, color = :red, label = "linear regression",
    legend = :topleft, title = "linear regression")

#
