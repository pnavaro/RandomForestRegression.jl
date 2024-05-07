import Pkg; Pkg.add("CSV")
import Pkg; Pkg.add("DataFrames")
import Pkg; Pkg.add("DecisionTree")

using CSV
using DataFrames
using DecisionTree
using LinearAlgebra
using RandomForestRegression
using Statistics

T = Float32

msd(a, b) = mean(abs2.(a .- b))
rmsd(a, b) = sqrt(msd(a, b))

file = joinpath(@__DIR__, "data", "train.csv")
df = DataFrame(CSV.File(file, missingstring="NA"))

feature_cols = [:LotFrontage, :LotArea, :OverallQual, :OverallCond, :YearBuilt, :TotalBsmtSF, Symbol("1stFlrSF"), 
                Symbol("2ndFlrSF"), :BedroomAbvGr, :KitchenAbvGr, :TotRmsAbvGrd, :GarageCars, :PoolArea, :MiscVal, :GrLivArea,
                :LowQualFinSF, :GarageYrBlt, :GarageArea
                ]

pred_col = :SalePrice
select!(df,vcat(feature_cols,pred_col))

dropmissing!(df)

feature_matrix = transpose(Matrix{T}(df[!,feature_cols])) |> collect

train_ys = Vector{T}(df[!,pred_col])

nof_real_train = floor(Int,0.8*length(train_ys))
println("Train on $(nof_real_train) of  $(length(train_ys))")

real_train_ys = @view train_ys[1:nof_real_train]
train_feature_matrix = @view feature_matrix[:,1:nof_real_train]
cv_ys = @view train_ys[nof_real_train+1:end]
cv_feature_matrix = @view feature_matrix[:, nof_real_train+1:end]

forest = RandomForestRegression.create_random_forest(train_feature_matrix, real_train_ys, 100)

pred_ys = RandomForestRegression.predict_forest(forest, cv_feature_matrix; default=mean(real_train_ys))

mae = sum(abs.(pred_ys.-cv_ys))/length(pred_ys)
println("MAE: ", mae)
println("RMSD: ", rmsd(log.(pred_ys), log.(cv_ys)))

println("")
println("What if we just take the mean?")
pred_ys = mean(real_train_ys)*ones(length(cv_ys))
mae = sum(abs.(pred_ys.-cv_ys))/length(pred_ys)
println("MAE: ", mae)
println("RMSD: ", rmsd(log.(pred_ys), log.(cv_ys)))


test_file = joinpath(@__DIR__, "data", "test.csv")
test_df = DataFrame(CSV.File(test_file, missingstring="NA"))

select!(test_df, vcat(:Id,feature_cols))
dropmissing!(test_df)

test_feature_matrix = transpose(Matrix{T}(test_df[!,feature_cols])) |> collect

pred_test_ys = RandomForestRegression.predict_forest(forest, test_feature_matrix; default=mean(real_train_ys))

submission_df = DataFrame(Id=test_df[!,:Id], SalePrice=pred_test_ys)
CSV.write(joinpath(@__DIR__, "data","submission_more_fts.csv"), submission_df)


# # julia package

feature_matrix = Matrix{T}(df[!,feature_cols])
nof_real_train = floor(Int, 0.8*length(train_ys))
println("Train on $(nof_real_train)  of  $(length(train_ys))")
real_train_ys = @view train_ys[1:nof_real_train]
train_feature_matrix = @view feature_matrix[1:nof_real_train,:]
cv_ys = @view train_ys[nof_real_train+1:end]
cv_feature_matrix = @view feature_matrix[nof_real_train+1:end,:]

model = build_forest(real_train_ys, train_feature_matrix, 8, 100)

pred_ys = apply_forest(model, cv_feature_matrix)

mae = sum(abs.(pred_ys.-cv_ys))/length(pred_ys)
println("With the RandomForestsRegression julia package: ")
println("MAE: ", mae)
println("RMSD: ", rmsd(log.(pred_ys), log.(cv_ys)))

test_feature_matrix = Matrix{T}(test_df[!,feature_cols])

pred_test_ys = apply_forest(model, test_feature_matrix)

submission_df = DataFrame(Id=test_df[!,:Id], SalePrice=pred_test_ys)
CSV.write(joinpath(@__DIR__, "data", "submission_julia_pkg.csv"), submission_df)
