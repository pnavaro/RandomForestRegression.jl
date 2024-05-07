# +
import DataFrames
using MLDatasets: BostonHousing
using RandomForestRegression
using Statistics

dataset = BostonHousing(; as_df = false)
# -

T = Float64

feature_matrix = T.(dataset.features)

train_ys = vec(T.(dataset.targets))

nof_real_train = floor(Int,0.8*length(train_ys))
println("Train on $(nof_real_train) of  $(length(train_ys))")

real_train_ys = @view train_ys[1:nof_real_train]
train_feature_matrix = @view feature_matrix[:,1:nof_real_train]
cv_ys = @view train_ys[nof_real_train+1:end]
cv_feature_matrix = @view feature_matrix[:, nof_real_train+1:end]

# +
forest = RandomForestRegression.create_random_forest(train_feature_matrix, real_train_ys, 100)

pred_ys = RandomForestRegression.predict_forest(forest, cv_feature_matrix; default=mean(real_train_ys))

msd(a, b) = mean(abs2.(a .- b))
rmsd(a, b) = sqrt(msd(a, b))
        
mae = sum(abs.(pred_ys.-cv_ys))/length(pred_ys)
println("MAE: ", mae)
println("RMSD: ", rmsd(log.(pred_ys), log.(cv_ys)))
println("")
println("What if we just take the mean?")
pred_ys = mean(real_train_ys)*ones(length(cv_ys))
mae = sum(abs.(pred_ys.-cv_ys))/length(pred_ys)
println("MAE: ", mae)
println("RMSD: ", rmsd(log.(pred_ys), log.(cv_ys)))
# -

# # Julia package DecisionTree.jl

# +
using DecisionTree

feature_matrix = Matrix{T}(dataset.features')
nof_real_train = floor(Int, 0.8*length(train_ys))
println("Train on $(nof_real_train)  of  $(length(train_ys))")
real_train_ys = @view train_ys[1:nof_real_train]
train_feature_matrix = @view feature_matrix[1:nof_real_train,:]
cv_ys = @view train_ys[nof_real_train+1:end]
cv_feature_matrix = @view feature_matrix[nof_real_train+1:end,:]

model = build_forest(real_train_ys, train_feature_matrix, 8, 100)

pred_ys = apply_forest(model, cv_feature_matrix)

mae = sum(abs.(pred_ys.-cv_ys))/length(pred_ys)
println("With the DecisionTree julia package: ")
println("MAE: ", mae)
println("RMSD: ", rmsd(log.(pred_ys), log.(cv_ys)))
# -


