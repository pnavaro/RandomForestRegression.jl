# ---
# jupyter:
#   jupytext:
#     formats: ipynb,jl:light,md
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

using LaTeXStrings
using Plots
using RandomForestRegression

?DataLongGenerator

data = DataLongGenerator(n=17, p=6, G=6)

w = findall(data.id .== 1)
p = plot(data.time[w], data.Y[w], c="grey")
for i in unique(data.id)
  w = findall(data.id .== i)
  plot!(p, data.time[w], data.Y[w], c="grey", label = :none)
end
p

# Let's see the fixed effects predictors:

p = plot(layout=(2,3))
for i in axes(data.X,2)
    w = findall(data.id .== 1)
    plot!(p[i], data.time[w],data.X[w,i], label = :none)
    title!(p[i], L"$X^{%$i}$")
    for k in unique(data.id)
        w = findall(data.id .== k)
        plot!(p[i], data.time[w], data.X[w,i], label = :none)
    end
end
p


