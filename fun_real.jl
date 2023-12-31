"""
    datagen_real(resol::Int, n::Int, orders; pc::Bool=false)

Function to generate ``n`` images of ``resol`` resolution using ``orders`` of Zernike polynomials.\\
``pc`` is a Bool to define wether multithreading is used or not.
"""
function datagen_real(resol::Int, n::Int, orders=J; smt::Bool=false)
    coef = randn32(orders |> length, n)
    ϕ = Array{Float32}(undef, (resol, resol, 1, n))# preallocation of the array
    if smt == true
        Threads.@threads for ii ∈ 1:n# evaluate with data
            ϕ[:, :, 1, ii] = evaluateZernike(resol, orders, coef[:, ii])
        end
    elseif smt == false
        for ii ∈ 1:n# evaluate with data
            ϕ[:, :, 1, ii] = evaluateZernike(resol, orders, coef[:, ii])
        end
    end
    DATA_train = ϕ, coef# create the training dataset tuple
    return DATA_train
end
datagen_real(2, 1, [0]);
#=
function ajuste(A, n)
    A[n+1:end-n, n+1:end-n]
end
g(x) = isnan(x) ? zero(x) : x
f(r) = abs(r) ≤ one(r) ? one(r) : zero(r)
function lnr(s::String)
    file = matopen(s)
    keys(file |> read)
    @read file zq
    zq = g.(zq)
    @read file xq
    @read file yq
    Z_c = IT.imresize(zq, (np, np))
    m = maximum(xq)
    X_r = IT.imresize(xq, (np, np)) ./ m
    Y_r = IT.imresize(yq, (np, np)) ./ m
    mask = [f(sqrt(ii^2 + jj^2)) for ii ∈ X_r[1, :], jj ∈ Y_r[:, 1]]
    A = Z_c .* mask
    ext_A = maximum(abs, extrema(A))
    A = A ./ ext_A
    A = reshape(A, (np, np, 1))
    A = cat(A, zeros((np, np)), dims=3)
    h = Plots.heatmap(X_r[1, :], Y_r[:, 1], A[:, :, 1, 1], aspect_ratio=1)
    return A, h, (X_r, Y_r)
end
=#
"""
    sample_real(np::Int, orders)

Generate a single sample ``resol`` resolution image using ``orders`` orders of Zernike polynomials.
"""
function sample_real(resol::Int, orders=J)
    coef = randn32(orders |> length)
    ϕ = evaluateZernike(resol, orders, coef)
    return coef, ϕ
end
sample_real(2, [0]);
"""
    test_real(input, orders, name::String)

Test multiple images to decompose into Zernike coefficients.
"""
function test_real(input::Array{<:AbstractFloat,4}, model_name::String)
    BSON.@load model_name * ".bson" model
    m = model |> gpu
    out = input |> gpu |> m |> cpu
    CUDA.reclaim()
    #ϕ::Array{Float32,2} = evaluateZernike(res, orders, out)
    return out#, ϕ
end
function test_real(input::Array{<:AbstractFloat,2}, model_name::String)
    return test_real(reshape(input, (res, res, 1, 1)), model_name)
end
"""
    test_real_CPU(input::Array{<:AbstractFloat,2}, orders, name::String)

Test a single image to decompose into Zernike coefficients.
"""
function test_real_CPU(input::Array{<:AbstractFloat,4}, model_name::String)
    BSON.@load model_name * ".bson" model
    #input = reshape(input, (res, res, 1, 1))
    out = input |> model
    #ϕ::Array{Float32,2} = evaluateZernike(res, orders, out)
    return out#, ϕ
end
function test_real_CPU(input::Array{<:AbstractFloat,2}, model_name::String)
    return test_real_CPU(reshape(input, (res, res, 1, 1)), model_name)
end
function validation_real(data, model_name::String; mode::Symbol=:CPU)
    ϕs, coef = data
    if mode == :CPU
        out = test_real_CPU(ϕs, model_name)
    elseif mode == :GPU
        if CUDA.has_cuda_gpu() == false
            error("No CUDA GPU found... exiting function, please set mode=:CPU.")
        end
        out = test_real(ϕs, model_name)
    end
    return mean(isapprox.(out, coef, rtol=0.1)), (out .- coef) .^ 2 |> mean |> sqrt, (out, coef)
end
