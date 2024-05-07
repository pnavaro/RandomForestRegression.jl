# ---
# jupyter:
#   jupytext:
#     formats: ipynb,jl:light
#     text_representation:
#       extension: .jl
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Julia 1.10.3
#     language: julia
#     name: julia-1.10
# ---

# # (S)MERF algorithm
#
# (S)MERF is an adaptation of the random forest regression method to longitudinal data introduced by Hajjem et. al. (2014) <doi:10.1080/00949655.2012.741599>.
# The model has been improved by Capitaine et. al. (2020) <doi:10.1177/0962280220946080> with the addition of a stochastic process.
# The algorithm will estimate the parameters of the following semi-parametric stochastic mixed-effects model: 
# $$
# Y_i(t)=f(X_i(t))+Z_i(t)\beta_i + \omega_i(t)+\epsilon_i
# $$
#
# with $Y_i(t)$ the output at time $t$ for the $i$th individual; $X_i(t)$ the input predictors (fixed effects) at time $t$ for the $i$th individual;
# \eqn{Z_i(t)} are the random effects at time \eqn{t} for the \eqn{i}th individual; \eqn{\omega_i(t)} is the stochastic process at time \eqn{t} for the \eqn{i}th individual
#  which model the serial correlations of the output measurements; \eqn{\epsilon_i} is the residual error.
#
# - `X` : A $N \times p$ matrix containing the `p` predictors of the fixed effects, column codes for a predictor.
# - `Y` : A vector containing the output trajectories.
# - `id` : Is the vector of the identifiers for the different trajectories.
# - `Z` : A $N \times q$ matrix containing the `q` predictor of the random effects.
# - `iter` : Maximal number of iterations of the algorithm. The default is set to \code{iter=100}
# - `mtry` : Number of variables randomly sampled as candidates at each split. The default value is \code{p/3}.
# - `ntree` : Number of trees to grow. This should not be set to too small a number, to ensure that every input row gets predicted at least a few times. The default value is  `ntree=500`.
# - `time` : Is the vector of the measurement times associated with the trajectories in `Y`,`Z` and `X`.
# - `sto` : Defines the covariance function of the stochastic process, can be either "none" for no stochastic process, `"BM"` for Brownian motion, `OrnUhl` for standard Ornstein-Uhlenbeck process, `BBridge` for Brownian Bridge, `fbm` for Fractional Brownian motion; can also be a function defined by the user.
# - `delta`: The algorithm stops when the difference in log likelihood between two iterations is smaller than `delta`. The default value is set to O.O01

# - `forest` Random forest obtained at the last iteration.
# - `random_effect` : Predictions of random effects for different trajectories.
# - `id_btilde`: Identifiers of individuals associated with the predictions `random_effects`.
# - `var_random_effects`: Estimation of the variance covariance matrix of random effects.
# - `sigma_sto`: Estimation of the volatility parameter of the stochastic process.
# - `sigma`: Estimation of the residual variance parameter.
# - `time`: The vector of the measurement times associated with the trajectories in `Y`,`Z` and `X`.
# - `sto`: Stochastic process used in the model.
# - `Vraisemblance`: Log-likelihood of the different iterations.
# - `id`: Vector of the identifiers for the different trajectories.
# - `OOB`: out of bag error of the fitted random forest at each iteration.

import Pkg; Pkg.add("DecisionTree")

# +
using LinearAlgebra
using RandomForestRegression
using DecisionTree

data = DataLongGenerator(n=20) # Generate the data composed by n=20 individuals.

# +
X = data.X
Y = data.Y
Z = data.Z
id = data.id
time = data.time
mtry = 2
ntree = 500

iter = 2
mtry = 2
delta = 0.1
# -

q = size(Z,2)
nind = length(unique(id))
btilde = zeros(nind, q)
sigmahat = 1
Btilde = diagm(ones(q))
epsilonhat = zero(Y)
id_btilde = unique(id)
Tiime = sort(unique(time))
omega = zero(Y)
sigma2 = 1
#Vrai <- NULL
inc = 1
#OOB <- NULL

# +
ystar = zero(Y)
    
for i in 1:iter

  fill!(ystar, 0)
  
  for k in 1:nind 
    indiv = findall(id .== unique(id)[k])
    ystar[indiv] .= Y[indiv] - Z[indiv, :] * btilde[k,:]
  end
  # using 2 random features, 10 trees
  forest = build_forest(ystar, X, mtry, ntree)

pred_ys = apply_forest(model, cv_feature_matrix)
#    randomForest(X,
#                 ystar,
#                 mtry = mtry,
#                 ntree = ntree,
#                 importance = TRUE)
#  fhat <- predict(forest)
#  print("fhat=")
#  print(fhat)
#  OOB[i] <- forest$mse[ntree]
#  for (k in 1:nind) {
#    indiv <- which(id == unique(id)[k])
#    V <-
#      Z[indiv, , drop = FALSE] %*% Btilde %*% t(Z[indiv, , drop = FALSE]) + diag(as.numeric(sigmahat), length(indiv), length(indiv))
#    btilde[k,] <-
#      Btilde %*% t(Z[indiv, , drop = FALSE]) %*% solve(V) %*% (Y[indiv] - fhat[indiv])
#    epsilonhat[indiv] <-
#      Y[indiv] - fhat[indiv] - Z[indiv, , drop = FALSE] %*% btilde[k,]
#  }
#
#  sigm <- sigmahat
#  sigmahat <-
#    sig(
#      sigma = sigmahat,
#      id = id,
#      Z = Z,
#      epsilon = epsilonhat,
#      Btilde = Btilde
#    )
#  Btilde  <-
#    bay(
#      bhat = btilde,
#      Bhat = Btilde,
#      Z = Z,
#      id = id,
#      sigmahat = sigm
#    )
#  Vrai <-
#    c(Vrai, logV(Y, fhat, Z, time, id, Btilde, 0, sigmahat, sto))
#  print(paste0("Vrai = ", Vrai[i]))
#
#  if (i > 1) {
#
#    inc <- abs((Vrai[i - 1] - Vrai[i]) / Vrai[i - 1])
#
#    if (inc < delta) {
#      print(paste0("stopped after ", i, " iterations."))
#      sortie <-
#        list(
#          forest = forest,
#          random_effects = btilde,
#          var_random_effects = Btilde,
#          sigma = sigmahat,
#          id_btilde = unique(id),
#          sto = sto,
#          vraisemblance = Vrai,
#          id = id,
#          time = time,
#          OOB = OOB
#        )
#      class(sortie) <- "longituRF"
#      print(sortie)
#      break
#    }
#  }
end
# -




