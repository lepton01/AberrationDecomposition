function coefftrain!(DATA, name; ep::Int=100, bs::Int=2)
    SET = DataLoader(DATA, batchsize=bs, parallel=true, shuffle=true) |> gpu
    BSON.@load name * ".bson" model
    model = model |> gpu
    opt = setup(Flux.Adam(), model)
    loss_log = Float32[]
    CUDA.@sync for ii ∈ 1:ep
        l1 = Float32[]
        for (input, label) ∈ SET
            loss, grads = withgradient(model) do m
                result = m(input)
                mae(result, label)
            end
            push!(l1, loss)
            update!(opt, model, grads[1])
        end
        l2 = sum(l1)
        push!(loss_log, l2)
        if ep >= 2_000
            if ii % 500 == 0
                println("Epoch = $ii. Training loss = $l2")
            end
        elseif ii % 10 == 0
            println("Epoch = $ii. Training loss = $l2")
        end
    end
    model = model |> cpu
    BSON.@save name * ".bson" model
    CUDA.reclaim()
    return loss_log
end
