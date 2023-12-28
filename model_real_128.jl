function modelcreateconv128_real(n_out::Int, name::String)# Modelo número 1
    n9 = 2^9
    model::Chain = Chain(
        Conv((3, 3), 1 => 32, selu, pad=SamePad()),
        Dropout(0.25),
        Conv((3, 3), 32 => 64, selu, pad=SamePad(), stride=2),
        Dropout(0.25),
        Conv((3, 3), 64 => 64, selu, pad=SamePad()),
        Dropout(0.25),
        Conv((3, 3), 64 => 128, selu, pad=SamePad(), stride=2),
        Dropout(0.25),
        Conv((3, 3), 128 => 128, selu, pad=SamePad()),
        Dropout(0.25),
        Conv((3, 3), 128 => 256, selu, pad=SamePad(), stride=2),
        Dropout(0.25),
        Conv((3, 3), 256 => 256, selu, pad=SamePad()),
        Dropout(0.25),
        Conv((3, 3), 256 => n9, selu, pad=SamePad(), stride=2),
        Dropout(0.25),
        Conv((3, 3), n9 => n9, selu, pad=SamePad()),
        Dropout(0.25),
        Conv((3, 3), n9 => n9, selu, pad=SamePad(), stride=2),
        Conv((4, 4), n9 => n9, selu),
        BatchNorm(n9),
        Flux.flatten,
        Dense(n9 => 256, selu),
        BatchNorm(256),
        Dropout(0.25),
        Dense(256 => n_out),
    )
    BSON.@save name * ".bson" model
    return
end

function modelcreateconv128_real_2(n_out::Int, name::String)# modelo número 2
    n9 = 2^9
    model::Chain = Chain(
        Conv((3, 3), 1 => 64, selu),# 128 -> 126
        Dropout(0.25),
        MaxPool((2, 2)),
        Conv((3, 3), 64 => 128, selu),# 63 -> 61
        Dropout(0.25),
        MaxPool((2, 2)),
        Conv((3, 3), 128 => 256, selu),# 30 -> 28
        Dropout(0.25),
        MaxPool((2, 2)),
        Conv((3, 3), 256 => n9, selu),# 14 -> 12
        Dropout(0.25),
        MaxPool((2, 2)),
        Conv((3, 3), n9 => n9, selu), # 6 -> 4
        Dropout(0.25),
        Conv((4, 4), n9 => n9, selu),
        BatchNorm(n9),
        Flux.flatten,
        Dense(n9 => n9, selu),
        BatchNorm(n9),
        Dropout(0.25),
        Dense(n9 => n_out),
    )
    BSON.@save name * ".bson" model
    return
end
