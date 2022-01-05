using Statistics

#########################
### Pooling Functions ###
#########################

bag_mean(x) = mean(x, dims=2)
bag_maximum(x) = maximum(x, dims=2)

"""
    mean_max(x)

Concatenates mean and maximum.
"""
function mean_max(x)
    m1 = mean(x, dims=2)
    m2 = maximum(x, dims=2)
    return vcat(m1,m2)
end

"""
    sum_stat(x)

Calculates a summary vector as a concatenation of mean, maximum, minimum, and var pooling.
"""
function sum_stat(x)
    m1 = mean(x, dims=2)
    m2 = maximum(x, dims=2)
    m3 = minimum(x, dims=2)
    m4 = var(x, dims=2)
    if any(isnan.(m4))
        m4 = zeros(length(m1))
    end
    return vcat(m1,m2,m3,m4)
end

# Calculates the difference vector.
diffn(x, n) = hcat(map(i -> x[:, i+n] .- x[:, i], 1:size(x,2)-n)...)

"""
    pool_source(_x; u = 0)

This function takes the original input and returns summary statistics as
well as other characteristics of the input matrix. Only calculates the data
for breathe-in or breathe-out (controlled with u = 0 or 1).
"""
function pool_source(_x; u = 0, rc = true)
    b = _x[3,:]
    if rc
        R, C = _x[4,1], _x[5,1]
    end
    x = _x[1:2, b .== u]

    # simple statistics
    M = sum_stat(x)

    # gaps
    g0 = vcat(
        mean(diffn(x,1)),
        maximum(diffn(x,1)),
        minimum(diffn(x,1))
    )
    g1 = vcat(
        mean(diffn(x,2)),
        maximum(diffn(x,2)),
        minimum(diffn(x,2))
    )
    g2 = vcat(
        mean(diffn(x,3)),
        maximum(diffn(x,3)),
        minimum(diffn(x,3))
    )

    # cardinality
    card = sum(b .== u)

    if rc
        return vcat(M,g0,g1,g2,R,C,card)
    else
        return vcat(M,g0,g1,g2,card)
    end
end

"""
    pool_transformed(x)

Returns the summary statistics of the transformed input matrix.
"""
function pool_transformed(x)
    # simple statistics
    sum_stat(x)
end

function pool(x_source, x_transformed)
    p1 = pool_source(x_source)
    p2 = pool_transformed(x_transformed)
    vcat(p1,p2)
end

# Adds padding to a vector or matrix.
function padding(x, l = 80)
    n = l - size(x, 2)
    return hcat(x, zeros(eltype(x), size(x,1), n))
end

"""
    pool_feature_add(_x; u = 0, rc = false)

Feature engineering function. Calculates summary statistics and
difference vector based on whether the variable `u_out` is 0 or 1.
"""
function pool_feature_add(_x; u = 0, rc = false)
    b = _x[3,:]
    R, C = _x[4,1], _x[5,1]
    x = _x[1:2, b .== u]
    s = size(x, 2)

    # simple statistics
    M = sum_stat(x)

    # gaps
    d1 = diffn(x,1)
    d2 = diffn(x,2)
    d3 = diffn(x,3)

    g0 = vcat(
        mean(d1),
        maximum(d1),
        minimum(d1)
    )
    g1 = vcat(
        mean(d2),
        maximum(d2),
        minimum(d2)
    )
    g2 = vcat(
        mean(d3),
        maximum(d3),
        minimum(d3)
    )

    # cardinality
    c = sum(b .== u)
    if rc
        one_feature = vcat(M,g0,g1,g2,R,C)
    else
        one_feature = vcat(M,g0,g1,g2)
    end
    one_matrix = repeat(one_feature, 1, 80)

    return (
        one_matrix,
        vcat(x, padding(d1, c), padding(d2, c), padding(d3, c))
    )
end
