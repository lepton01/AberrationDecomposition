
"""
    mn2OSA(m::Int,n::Int)

Convert the integer pair (n,m) that defines the Zernike polynomial Z_n^m(ρ,θ) to the sequential OSA/ANSI standard index number.

Throws an `ArgumentError` for invalid integer pairs.

See also: [`mn2Noll`], [`OSA2mn`]

Source: "Standards for Reporting the Optical Aberrations of Eyes", Journal of Refractive Surgery Volume 18 September/October 2002

# Example:
```julia-repl
julia> mn2OSA(2,2)
5

julia> mn2OSA(1,1)
2

julia> mn2OSA(1,2)
ERROR: ArgumentError: Invalid combination of (m,n)=(1,2) in OSA/ANSI indexing.
Stacktrace:
 [1] mn2OSA(m::Int64, n::Int64)
   @ ZernikePolynomials ~/.julia/dev/ZernikePolynomials.jl/src/ZernikePolynomials.jl:29
 [2] top-level scope
   @ REPL[62]:1
```
"""
function mn2OSA(m::Int, n::Int)
    if n < abs(m) || isodd(n - m)
        throw(ArgumentError("Invalid combination of (m,n)=($m,$n) in OSA/ANSI indexing."))
    else
        return Int((1 // 2) * (n * (n + 2) + m))
    end
end

"""
    mn2Noll(m::Int,n::Int)

Convert the integer pair (n,m) that defines the Zernike polynomial Z_n^m(ρ,θ) to the sequential Noll index number.

Throws an `ArgumentError` for invalid integer pairs.

See also: [`mn2OSA`], [`Noll2mn`]

Source: (https://en.wikipedia.org/wiki/Zernike_polynomials)

# Example:
```julia-repl
julia> mn2Noll(1,1)
2

julia> mn2Noll(1,2)
ERROR: ArgumentError: Invalid combination of (m,n)=(1,2) in Noll indexing.
Stacktrace:
 [1] mn2Noll(m::Int64, n::Int64)
   @ ZernikePolynomials ~/.julia/dev/ZernikePolynomials.jl/src/ZernikePolynomials.jl:53
 [2] top-level scope
   @ REPL[20]:1
```
"""
function mn2Noll(m::Int, n::Int)
    if n < abs(m) || isodd(n - m)
        throw(ArgumentError("Invalid combination of (m,n)=($m,$n) in Noll indexing."))
    else
        if m > 0 && (mod(n, 4) ∈ (0, 1))
            p = 0
        elseif m < 0 && (mod(n, 4) ∈ (2, 3))
            p = 0
        elseif m ≥ 0 && (mod(n, 4) ∈ (2, 3))
            p = 1
        elseif m ≤ 0 && (mod(n, 4) ∈ (0, 1))
            p = 1
        else
            throw(ArgumentError("Invalid combination of (m,n)=($m,$n) in Noll indexing."))
        end
        return Int((1 // 2) * n * (n + 1) + abs(m) + p)
    end
end

"""
    OSA2mn(j::Int)

Convert the sequential OSA/ANSI stardard index number j to the integer pair (n,m) that defines the Zernike polynomial Z_n^m(ρ,θ).

See also: [`Noll2mn`], [`mn2OSA`], [`OSA2Noll`]

Source: "Standards for Reporting the Optical Aberrations of Eyes", Journal of Refractive Surgery Volume 18 September/October 2002

# Example:
```julia-repl
julia> [OSA2mn(j) for j in 0:10]
11-element Vector{Tuple{Int64, Int64}}:
 (0, 0)
 (-1, 1)
 (1, 1)
 (-2, 2)
 (0, 2)
 (2, 2)
 (-3, 3)
 (-1, 3)
 (1, 3)
 (3, 3)
 (-4, 4)
```
"""
function OSA2mn(j::Int)
    n = ceil(Int, (-3 + sqrt(9 + 8j)) / 2)
    m = 2j - n * (n + 2) |> Int
    return m, n
end

"""
    Noll2mn(j::Int)

Convert the Noll index number j to the integer pair (n,m) that defines the Zernike polynomial Z_n^m(ρ,θ).

See also: [`OSA2mn`], [`mn2Noll`], [`Noll2OSA`]

Source: (https://en.wikipedia.org/wiki/Zernike_polynomials)

# Example:
```julia-repl
julia> [OSA2mn(j) for j in 0:10]
```
"""
function Noll2mn(j::Int)
    n = ceil(Int, (-3 + sqrt(1 + 8j)) / 2)
    jr = j - Int(n * (n + 1) / 2)
    if mod(n, 4) ∈ (0, 1)
        m1 = jr
        m2 = -(jr - 1)
        if iseven(n - m1)
            m = m1
        else
            m = m2
        end
    else # mod(n,4) ∈ (2,3)
        m1 = jr - 1
        m2 = -(jr)
        if iseven(n - m1)
            m = m1
        else
            m = m2
        end
    end
    return m, n
end

"""
    Noll2OSA(j::Int)

Convert the Noll index number j to the OSA/ANSI stardard index number for Zernike polynomials.

See also: [`OSA2Noll`], [`Noll2mn`]

# Example:
```julia-repl
julia> [Noll2OSA(OSA2Noll(j)) for j = 0:10]
```
"""
function Noll2OSA(j::Int)
    return mn2OSA(Noll2mn(j)...)
end

"""
    OSA2Noll(j::Int)

Convert OSA/ANSI stardard index number to the Noll index number for Zernike polynomials.

See also: [`OSA2Noll`], [`Noll2mn`]

# Example:
```julia-repl
julia> [Noll2OSA(OSA2Noll(j)...) for j = 0:10]
```
"""
function OSA2Noll(j::Int)
    return mn2Noll(OSA2mn(j)...)
end


"""
    R([T=Float64], m::Int,n::Int)

Obtain the function ρ -> R_n^|m|(ρ), where R is the radial polynomial in Zernike polynomials

# Example:
```julia-repl
julia> R(1,1)
```
"""
function R(::Type{T}, m::Int, n::Int) where {T}
    p(s) = ((-1)^s * factorial(n - s)) / T(factorial(s) * factorial(Int(0.5 * (n + abs(m)) - s)) * factorial(Int(0.5 * (n - abs(m)) - s)))
    # round brackets to be a generator instead of a Vector
    f(x) = sum(p(s) * x .^ (n - 2s) for s in 0:Int((n - abs(m)) / 2))
    return f
end

function R(m::Int, n::Int) # radial polynomial
    R(Float64, m, n)
end

"""
    normalization([T=Float64], m::Int, n::Int)

Normalization constant of the zernike polynomial type `T`.
"""
function normalization(::Type{T}, m::Int, n::Int) where {T}
    δ(x, y) = x == y ? one(T) : zero(T)
    c = sqrt(T(2) * (n + 1) / (one(T) + δ(m, 0)))
    return c
end

function normalization(m::Int, n::Int) # normalization constant of the zernike polynomial
    return normalization(Float64, m, n)
end

"""
    Zernike(m::Int,n::Int;coord=:polar)

Obtain the function (ρ,θ) -> Z_n^m(ρ,θ), where Z is the Zernike polynomial with coefficients n and m. ρ is the radius and θ the angle.

If coord=:cartesian the function is (x,y) -> Z_n^m(x,y) in Cartesian coordinates.

# Example:
```julia-repl
julia> Zernike(1,1)
julia> Zernike(1,1,coord=:polar)
julia> Z = Zernike(1,1,:cartesian)
julia> Z(0.5,0.2)
```
"""
function Zernike(m::Int, n::Int; coord=:polar)
    δ(ρ) = ifelse(abs(ρ) ≤ 1, one(ρ), zero(ρ))
    # use let block to prevent this bug https://github.com/JuliaLang/julia/issues/15276
    # further, we pass the types to normalization
    Z = let
        if m ≥ 0
            (ρ, θ) -> (normalization(typeof(ρ), m, n) * R(typeof(ρ), m, n)(ρ) * cos(m * θ) * δ(ρ))
        else
            (ρ, θ) -> (-normalization(typeof(ρ), m, n) * R(typeof(ρ), m, n)(ρ) * sin(m * θ) * δ(ρ))
        end
    end
    if coord == :cartesian
        g(x, y) = (sqrt(x .^ 2 + y .^ 2), atan(y, x))
        return (x, y) -> Z(g(x, y)...)
    elseif coord == :polar
        return Z
    else
        throw(ArgumentError("Unrecognized coordinate system $coord."))
    end
end

"""
    Zernike(j::Int;index=:OSA,coord=:polar)

Obtain the function (ρ,θ) -> Z_j(ρ,θ), where Z is the Zernike polynomial with sequential index j, according to the indexing by :OSA or :Noll.

If coord=:cartesian the function is (x,y) -> Z_j(x,y) in Cartesian coordinates.

# Example:
```julia-repl
julia> Zernike(1)
julia> Zernike(1,coord=:polar)
julia> Z = Zernike(5,index=:Noll,coord=:cartesian)
julia> Z(0.5,0.2)
```
"""
function Zernike(j::Int; index=:OSA, coord=:polar)
    if index == :OSA
        return Zernike(OSA2mn(j)..., coord=coord)
    elseif index == :Noll
        return Zernike(Noll2mn(j)..., coord=coord)
    else
        error("Unknown Zernike sequential index.")
    end
end
"""
    evaluateZernike(N::Int, J::Vector{Int}, coefficients::Vector{Float64}; index=:OSA)

Evaluate the Zernike polynomials on an N-by-N grid as specified by the Zernike coefficients of the polynomials J

# Example:
```julia-repl
julia> W = evaluateZernike(64,[5, 6],[0.3, 4.1])
```
"""
function evaluateZernike(N::Int, J::Vector{Int}, coefficients::AbstractArray{T,1}; index=:OSA) where {T}
    X = range(-one(T), one(T), length=N)
    Y = range(-one(T), one(T), length=N) |> transpose
    out_arr = zeros(T, N, N)
    for jj ∈ 1:length(J)
        Z = Zernike(J[jj], coord=:cartesian, index=index)
        out_arr .+= coefficients[jj] .* Z.(X, Y)
    end
    return out_arr
end

"""
    evaluateZernike(x::AbstractArray{<: AbstractFloat,1}, J::Vector{Int}, coefficients::Vector{Float64}; index=:OSA)

Evaluate the Zernike polynomials on a grid as specified by the Zernike coefficients of the polynomials J, on a grid with points in x

# Example:
```julia-repl
julia> W = evaluateZernike(64,[5, 6],[0.3, 4.1])
```
"""
function evaluateZernike(X::AbstractArray{<:AbstractFloat,1}, J::Vector{Int},
    coefficients::Vector{T}; index=:OSA) where {T}
    N = length(X)
    out_arr = zeros(T, N, N)
    for j ∈ 1:length(J)
        Z = Zernike(J[j], coord=:cartesian, index=index)
        out_arr .+= coefficients[j] .* Z.(X, X')
    end
    return out_arr
end

function evaluateZernike(n::Int, J::Int, coefficients::T; index=:OSA) where {T}
    return evaluateZernike(n, [J], [coefficients]; index=index)
end

function evaluateZernike(X::AbstractArray{<:AbstractFloat,1}, J::Int, coefficients::T; index=:OSA) where {T}
    return evaluateZernike(X, [J], [coefficients]; index=index)
end
