using Distributions

export DataLongGenerator

"""
$(TYPEDEF)

 Simulate longitudinal data according to the semi-parametric stochastic mixed-effects model given by: 

```math
Y_i(t)=f(X_i(t))+Z_i(t)\\beta_i + \\omega_i(t)+\\epsilon_i
```

with ``Y_i(t)`` the output at time ``t`` for the ``i``th individual; ``X_i(t)`` the input predictors (fixed effects) at time ``t`` for the ``i``th individual;
 ``Z_i(t)`` are the random effects at time ``t`` for the ``i``th individual; ``\\omega_i(t)`` is a Brownian motion with volatility ``\\gamma^2=0.8`` at time ``t`` for the ``i``th individual; ``\\epsilon_i`` is the residual error with
 variance ``\\sigma^2=0.5``.
 The data are simulated according to the simulations in low dimensional in the low dimensional scheme of the paper [Capitaine2021](@cite)

 `n` : Number of individuals. The default value is n=50.
 `p` : Number of predictors. The default value is p=6.
 `G` : Number of groups of predictors with temporal behavior, generates p-G input variables with no temporal behavior.

return a list of the following elements: 

 - `Y`: vector of the output trajectories.
 - `X`: matrix of the fixed-effects predictors.
 - `Z`: matrix of the random-effects predictors.
 - `id`:  vector of the identifiers for each individual.
 - `time`: vector the the time measurements for each individual.

"""
struct DataLongGenerator

    Y :: Vector{Float64}
    X :: Matrix{Float64}
    Z :: Matrix{Float64}
    id :: Vector{Int}
    time :: Vector{Float64}


function DataLongGenerator(; n = 50, p = 6, G = 6)

    mes = floor.(Int, 4 * rand(n) .+ 8)
    time = Float64[]
    id = Int64[]
    nb2 = 1:n
    for i = 1:n
        push!(time, 1:mes[i]...)
        push!(id, fill(nb2[i], mes[i])...)
    end

    bruit = floor(0 * p)
    bruit += (p - bruit) % G
    nices = Int[]
    for i = 1:G
        push!(nices, fill(i, (p - bruit) ÷ G)...)
    end

    comportements = zeros(length(time), G)
    @. comportements[:, 1] = 2.44 + 0.04 * (time - ((time - 6)^2) / (time / 3))
    @. comportements[:, 2] = 0.5 * time - 0.1 * (time - 5)^2
    @. comportements[:, 3] = 0.25 * time - 0.05 * (time - 6)^2
    @. comportements[:, 4] = cos((time - 1) / 3)
    @. comportements[:, 5] = 0.1 * time + sin(0.6 * time + 1.3)
    @. comportements[:, 6] = -0.1 * time^2


    X = zeros(length(time), p)
    for i = 1:(p-bruit)
        X[:, i] .= comportements[:, nices[i]] .+ randn(length(time)) .* 0.2
    end

    for j = 1:n
        w = findall(id .== j)
        X[w, 1:(p-bruit)] .= X[w, 1:(p-bruit)] .+ randn(1) * 0.1
    end

    for i = (p-bruit):p
        X[:, i] .= randn(length(time)) .* 3.0
    end

    f = 1.3 .* X[:, 1] .^ 2 .+ 2 * sqrt.(abs.(X[:, findfirst(n -> n == 2, nices)]))

    sigma = [0.5 0.6; 0.6 3]
    Btilde = zeros(length(unique(id)), 2)
    d = MvNormal([0.0, 0.0], sigma)
    for i in axes(Btilde, 1)
        Btilde[i, :] .= rand(d)
    end

    Z = stack([ones(length(f)), 2 * rand(length(f))])

    effets = Float64[]
    for i = 1:length(unique(id))
        w = findall(id .== unique(id)[i])
        push!(effets, (Z[w, :] * Btilde[i, :])...)
    end

    ##### simulation de mouvemments brownien

    gam = 0.8
    BM = Float64[]
    m = length(unique(id))
    for i = 1:m
        w = findall(id .== unique(id)[i])
        W = zeros(length(w))
        t = time[w]
        for j = 2:length(w)
            W[j] = W[j-1] + sqrt(gam * (t[j] - t[j-1])) * randn()
        end
        push!(BM, W...)
    end

    sigma2 = 0.5
    Y = f .+ effets .+ randn(length(f)) .* sigma2 .+ BM

    new( Y, X, Z, id, time )

end

end
