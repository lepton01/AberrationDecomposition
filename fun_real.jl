function datagen_real(resol::Int, n::Int, J; pc::Bool=false)
    c_train = randn32(J |> length, n)
    ϕ_train = Array{Float32}(undef, (resol, resol, 1, n))# preallocation of the array
    if pc == true
        Threads.@threads for ii ∈ 1:n# evaluate with data
            ϕ_train[:, :, 1, ii] = evaluateZernike(resol, J, c_train[:, ii])
        end
    elseif pc == false
        for ii ∈ 1:n# evaluate with data
            ϕ_train[:, :, 1, ii] = evaluateZernike(resol, J, c_train[:, ii])
        end
    end
    DATA_train = ϕ_train, c_train# create the training dataset tuple
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
function test_real_single(input::Array{<:AbstractFloat,2}, J, name::String)
    BSON.@load name * ".bson" model
    model = model |> gpu
    input = reshape(input, (res, res, 1, 1))
    out::Vector{Float64} = input |> gpu |> model |> cpu |> vec
    ϕ::Array{Float32,2} = evaluateZernike(res, J, out)
    return out, ϕ
end
function test_real(input::Array{<:AbstractFloat,4}, J, name::String)
    BSON.@load name * ".bson" model
    model = model |> gpu
    out::Vector{Float64} = input |> gpu |> model |> cpu |> vec
    ϕ::Array{Float32,2} = evaluateZernike(res, J, out)
    return out, ϕ
end
function test_real(input::Array{<:AbstractFloat,2}, J, name::String)
    return test_real(reshape(input, (res, res, 1, 1)), J, name)
end
function test_real_CPU(input::Array{<:AbstractFloat,2}, J, name::String)
    BSON.@load name * ".bson" model
    input = reshape(input, (res, res, 1, 1))
    out::Vector{Float64} = input |> model |> vec
    ϕ::Array{Float32,2} = evaluateZernike(res, J, out)
    return out, ϕ
end
function sample_real(np::Int, J)
    C = randn32(J |> length)
    ϕ = evaluateZernike(np, J, C)
    return C, ϕ
end
function validation_real(data, model::String, mode::Symbol=:CPU; pc::Bool=false)
    ϕs, C = data
    out = C |> similar
    if mode == :CPU
        if pc == false
            for ii ∈ axes(ϕs, 3)
                out[:, ii], _ = test_real_CPU(ϕs[:, :, ii], J, model)
            end
        elseif pc == true
            Threads.@threads for ii ∈ axes(ϕs, 3)
                out[:, ii], _ = test_real_CPU(ϕs[:, :, ii], J, model)
            end
        end
    elseif mode == :GPU
        for ii ∈ axes(ϕs, 3)
            out[:, ii], _ = test_real_single(ϕs[:, :, ii], J, model)
        end
    end
    return mean(isapprox.(out, C, rtol=0.1)), (out .- C) .|> (x -> x^2) |> mean |> sqrt, out, C
end
