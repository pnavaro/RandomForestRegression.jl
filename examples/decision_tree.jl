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

# +
import CSV
using DecisionTree
using DataFrames
import Downloads

source = Downloads.download("http://perso.ens-lyon.fr/lise.vaudor/Rdata/Arbres_decisionnels/magasin.csv")

data = CSV.read(source, DataFrame)
# -

transform!(data, :Prix => ByRow(x -> parse(Float64, replace(x, "," => "."))) => :Euros)

features = Matrix{Float64}(data[!, [:Euros, :Flashitude, :Branchitude, :Qualite] ])
labels = vec(data[!, :Achat])

# +
tree = build_tree(labels, features)
model = prune_tree(tree, 0.9)

print_tree(model, 5)
# -



function calcul_impurete(x)
  probas = [sum( x .== i) for i in unique(x)] ./ length(x)
  return sum(probas .* ( 1 .- probas))
end

impurete = calcul_impurete(Achat)

prA = 1
IA = calcul_impurete(Achat)
println(IA)

# +
n = length(Achat)
Branchitude = data.Branchitude
indL = findall(Branchitude .< 85)
indR = findall(Branchitude .>= 85)
prL = length(indL)/n
prR = length(indR)/n
IL = calcul_impurete(Achat[indL])
IR = calcul_impurete(Achat[indR])

perte_impurete = prA * IA - prL * IL - prR * IR
println(perte_impurete)
# -

indL = findall(Branchitude .< 65)
indR = findall(Branchitude .>= 65)
prL = length(indL)/n
prR = length(indR)/n
IL = calcul_impurete(Achat[indL])
IR = calcul_impurete(Achat[indR])
perte_impurete = prA * IA - prL * IL - prR * IR
println(perte_impurete)

# +
Flashitude = data.Flashitude
prA=1
indL=findall(Flashitude .< 65)
indR=findall(Flashitude .>= 65)

n=length(Achat)
prL=length(indL)/n
prR=length(indR)/n
IA=calcul_impurete(Achat)
IL=calcul_impurete(Achat[indL])
IR=calcul_impurete(Achat[indR])

perte_impurete=prA*IA-prL*IL-prR*IR
println(perte_impurete)

# -


