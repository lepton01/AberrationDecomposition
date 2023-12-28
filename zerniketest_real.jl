# Aberration decomposition
# Angel M. Ortiz-Ochoa, foton.angel@gmail.com
# 20/04/2023
import Plots, ImageTransformations as IT, BSON, Serialization
using LinearAlgebra, Statistics
using Flux, MAT, CUDA
using Flux: DataLoader, setup, withgradient, update!, mae, mse

"zernike_eval.jl" |> include
const J::Vector{Int} = [aa for aa ∈ 1:20]# order of the function, OSA indexing (Zⱼ)
const res::Int = 2^7# resolution of the image
const num::Int = 5_000# number of generated data
const num_test::Int = 1_000# number of generated data
const iter::Int = 50# number of epochs to train
const x = range(-1.0f0, 1.0f0, res)# to evaluate the ZernikePolynomials
model_name::String = "conv_real_$(res)_004_$(length(J))order"# name of the model to use
"model_real_$(res).jl" |> include
#"train.jl" |> include
"fun_real.jl" |> include

## MODEL CREATION AND DATA GENERATION
@time modelcreateconv128_real(J |> length, model_name)# create new model

## TRAINING DATA GENERATION AND EXPORT
@time TRAIN = datagen_real(res, num, J);# create training data
#@time TRAIN2 = datagen_real(res, num, J, true);

Serialization.serialize("$(res)_real_train_$(num)_$(length(J))_order", TRAIN);

## TESTING DATA GENERATION AND EXPORT
@time TEST = datagen_real(res, num_test, J);# create testing data
#@time TEST = datagen_real(res, num_test, J, true);
Serialization.serialize("$(res)_real_test_$(num_test)_$(length(J))order", TEST);

## TRAINING
function train(n, model::String)
    TRAIN = Serialization.deserialize("$(res)_real_train_$(num)_$(length(J))_order")
    n_r = rand(1:num, n)
    TRAIN = TRAIN[1][:, :, :, n_r], TRAIN[2][:, n_r]
    return coefftrain!(TRAIN, model, ep=iter, bs=32)
end
train(5_000, model_name)
coefftrain!(TRAIN, model_name, ep=iter, bs=32)

## TESTING
function testing(num, model, mode=:CPU)
    data = Serialization.deserialize("$(res)_real_test_$(num_test)_$(length(J))order")
    n_r = rand(1:num_test, num)
    data = data[1][:, :, 1, n_r], data[2][:, n_r]
    return validation_real(data, model, mode)
end
function __init__()
    testing(1, model_name)
    testing(1, model_name, :GPU)
end
@time __init__();
@time acc, error_cuad_medio, coefs = testing(1_000, model_name)
@time acc, error_cuad_medio, coefs = testing(1_000, model_name, :GPU)
aa = vec((coefs[1] .- coefs[2]) ./ coefs[2]);
Plots.histogram(aa, legend=false)

## TESTING 2
data_test = Serialization.deserialize("$(res)_real_test_$(num_test)_$(length(J))order");
data_test = data_test[1][:, :, 1, :], data_test[2][:, :];

n_r = rand(1:num_test, 1_000);
data_test = data_test[1][:, :, 1, n_r], data_test[2][:, n_r];

@time acc, error_cuad_medio, coefs = validation_real(data_test, model_name)
@time acc, error_cuad_medio, coefs = validation_real(data_test, model_name, :GPU)
aa = vec((coefs[1] .- coefs[2]) ./ coefs[2]);
Plots.histogram(aa, legend=false)

## IMPORT SIM DATA
const ss1::String = "z_ojo_Normal.mat"
const ss2::String = "z_tilt_pi32.mat"
file1 = matopen(ss1)
file2 = matopen(ss2)
#@show keys(file |> read)
@read file1 zq
Plots.heatmap(zq, aspect_ratio=1, title="Original", ylabel="px", xlabel="px")
Plots.surface(zq)
zq = zq .- zq[end÷2, end÷2]
zq1 = g.(view(zq, :, :))
@read file2 zq
Plots.heatmap(zq, aspect_ratio=1)
zq = zq .- zq[end÷2, end÷2]
zq2 = g.(view(zq, :, :))

#Plots.heatmap(zq1)
#Plots.heatmap(zq2)

#@read file xq
#@read file yq
#m = maximum(xq)
#X_r = IT.imresize(xq, 128, 128) ./ m
#Y_r = IT.imresize(yq, 128, 128) ./ m
Z_comp1 = IT.imresize(zq1, res, res)
Z_comp2 = IT.imresize(zq2, res, res)

## DATA IN
input1, heat1, (X1, Y1) = lnr(ss1)
input2, heat2, (X2, Y2) = lnr(ss2)
Plots.heatmap(heat1, ascpect_ratio=1, xlabel="x", ylabel="y", title="Preprocessed for input")
#Plots.plot(heat1, heat2, layout=(1, 2), aspect_ratio=1)
#Plots.heatmap(x, x, evaluateZernike(128, [2], [1.0]), aspect_ratio=1)

## VALIDATION
@time coef, B, P = sample_real(res, J)
@time C, F0, p0 = test_real(B, J, model_name);
P
Plots.plot(p0, aspect_ratio=1, ylabel="px", xlabel="px")
Plots.plot([coef C], label=["input" "output"], linewidth=2, xlabel="Zernike OSA index (1-$(J |> length))", ylabel="value")

## REAL DATA SET
@time CC1, F1, p1 = test(input1, model_name);
@time CC2, F2, p2 = test(input2, model_name);
Plots.plot(p1[1], p1[2], layout=(1, 2), aspect_ratio=1)
F11 = F1[1]
F11 = F11 .+ zq[end÷2, end÷2]
Plots.surface(F11)
Plots.plot(p2[1], p2[2], layout=(1, 2), aspect_ratio=1, ylabel="px", xlabel="px")
Plots.plot(CC1, linewidth=2, xlabel="Zernike OSA index (1-$(J |> length))", ylabel="value", title="Obtained coefficients", legend=false)
F_normal = F11 .+ (NaN .* (X1 .^ 2 .+ Y1 .^ 2 .≥ 1.0))
#F_anormal = F2[1] .+ (NaN .* (X1 .^ 2 .+ Y1 .^ 2 .≥ 1.0))
#heatmap(F2[1])
#heatmap(F2[1] .- F1[1], aspect_ratio=1)
#heatmap(F1[1] .- Z_comp1)
#heatmap(F2[1] .- Z_comp2)
#surface(-F_normal)
Plots.heatmap(-F_normal, title="Postprocessed", aspect_ratio=1, ylabel="px", xlabel="px")
