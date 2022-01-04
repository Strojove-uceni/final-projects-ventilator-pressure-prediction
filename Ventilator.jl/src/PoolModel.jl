using Flux
using ConditionalDists, Distributions
using StatsBase
using Random

"""
PoolModel has 5 components:
- prepool_net
- postpool_net
- poolf
- generator
- decoder

Pre-pool and post-pool nets are neural networks. The former is responsible for the
transformation of all instances, the latter only transforms the one-vector summary.

PoolModel can be extended to be used for various tasks based on the pooling fuction,
nature of data.
"""
struct PoolModel{pre <: Chain, post <: Chain, fun <: Function, g <: ConditionalMvNormal, d <: Chain}
    prepool_net::pre
    postpool_net::post
    poolf::fun
    generator::g
    decoder::d
end

Flux.@functor PoolModel

function Flux.trainable(m::PoolModel)
    (prepool_net = m.prepool_net, postpool_net = m.postpool_net, generator = m.generator, decoder = m.decoder)
end

function Base.show(io::IO, pm::PoolModel)
    nm = "PoolModel($(pm.poolf))"
	print(io, nm)
end

"""
pm_constructor(;idim, odim, hdim=32, predim=16, postdim=16, edim=16,
                            activation="swish", nlayers=3, var="scalar",
                            poolf="pool", init_seed=nothing, kwargs...)

Constructs a PoolModel. Some input dimensions are automatically calculated based on the chosen
pooling function.

Dimensions:
- idim: input dimension
- odim: output dimension of the decoder
- hdim: hidden dimension in all networks
- predim: the input dimension of pooling function
- postdim: the output dimension of post-pool network and input dimension of encoder and generator
- edim: output dimension of encoder and generator, input dimension to decoder
"""
function pm_constructor(xsample;idim, odim, hdim=32, predim=16, postdim=16, edim=16,
                            activation="swish", cell="Dense", nlayers=3, var="scalar",
                            poolf="pool", init_seed=nothing, kwargs...)

    fun = eval(:($(Symbol(poolf))))

    # if seed is given, set it
	(init_seed != nothing) ? Random.seed!(init_seed) : nothing

    # pre-pool network
    pre = Chain(
        build_mlp(idim,hdim,hdim,nlayers-1,activation=activation,cell=cell)...,
        Dense(hdim,predim)
    )
    # dimension after pooling
    xt = randn(5,10)
    xt[3,:] = sample([0,1],10)
    pooldim = length(fun(xt,randn(predim)))
    # post-pool network
    post = Chain(
        build_mlp(pooldim,hdim,hdim,nlayers-1,activation=activation)...,
        Dense(hdim,postdim)
    )
    
    if var == "scalar"
        gen = Chain(
            build_mlp(predim+postdim,hdim,hdim,nlayers-1,activation=activation)...,
            SplitLayer(hdim,[edim,1])
        )
        gen_dist = ConditionalMvNormal(gen)
    else
        gen = Chain(
            build_mlp(predim+postdim,hdim,hdim,nlayers-1,activation=activation)...,
            SplitLayer(hdim,[edim,edim])
        )
        gen_dist = ConditionalMvNormal(gen)
    end

    dec = Chain(
        build_mlp(edim+idim,hdim,hdim,nlayers-1,activation=activation,cell=cell)...,
        Dense(hdim,odim)
    )

    pm = PoolModel(pre, post, fun, gen_dist, dec)
    return pm
end

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
        mean(diffn(x,2)),
        maximum(diffn(x,2)),
        minimum(diffn(x,2))
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

"""
    pm_loss(m::PoolModel, x)

Loss function for the PoolModel. Based on Chamfer distance.
"""
function pm_loss(m::PoolModel, _x, _y; lossf::Function=Flux.mse)
    # get only u_out = 0 values
    b = _x[3,:]
    x = _x[1:2, b .== 0]

    # pre-pool network transformation of x
    # output size = predim
    v = m.prepool_net(x)
    
    # pooling
    # _x is the source, v is transformed
    p = m.poolf(_x, v)
    # output size = postdim
    p_post = m.postpool_net(p)

    # create context matrix and add to transformed matrix v
    c = reshape(repeat(p_post, size(v,2)),size(p_post,1),size(v,2))
    h = vcat(v, c)
    
    # generate the latent variables
    z = rand(m.generator, h)

    # add the pre-pool tranfrormated matrix to the decoder and decode
    zt = vcat(z, x)
    dz = m.decoder(zt)

    Flux.reset!(m.prepool_net)
    Flux.reset!(m.decoder)

    y = _y[:, b .== 0]
    return lossf(dz, y)
end
function pm_loss_full(m::PoolModel, _x, _y; lossf::Function=Flux.mse)
    # get only u_out = 0 values
    b = _x[3,:]
    x_full = _x[1:2,:]
    x = _x[1:2, b .== 0]

    # pre-pool network transformation of x
    # output size = predim
    # use all information in x
    v = m.prepool_net(x_full)

    # pooling
    # _x is the source, v is transformed
    p = m.poolf(_x, v)
    # output size = postdim
    p_post = m.postpool_net(p)

    # create context matrix and add to transformed matrix v
    len = sum(b .== 0)
    c = reshape(repeat(p_post, len),size(p_post,1),len)
    h = vcat(v[:, b .== 0], c)
    
    # generate the latent variables
    z = rand(m.generator, h)

    # add the pre-pool tranfrormated matrix to the decoder and decode
    zt = vcat(z, x)
    dz = m.decoder(zt)

    y = _y[:, b .== 0]
    return lossf(dz, y)
end

######################################
### Score functions and evaluation ###
######################################

"""
    reconstruct(m::PoolModel, x)

Reconstructs the input matrix. Adds random numbers for 
u_out = 1.
"""
function reconstruct(m::PoolModel, _x)
    b = _x[3,:]
    x = _x[1:2, b .== 0]
    v = m.prepool_net(x)
    p = m.poolf(_x, v)
    p_post = m.postpool_net(p)
    c = reshape(repeat(p_post, size(v,2)),size(p_post,1),size(v,2))
    h = vcat(v, c)
    z = mean(m.generator, h)
    zt = vcat(z, x)
    dz = m.decoder(zt)

    Flux.reset!(m.prepool_net)
    Flux.reset!(m.decoder)

    hcat(dz, rand(1,sum(b .== 1)))
end

"""
    predict(m::PoolModel, x)

Reconstructs the input matrix only at values u_out = 0.
"""
function predict(m::PoolModel, _x)
    b = _x[3,:]
    x = _x[1:2, b .== 0]
    v = m.prepool_net(x)
    p = m.poolf(_x, v)
    p_post = m.postpool_net(p)
    c = reshape(repeat(p_post, size(v,2)),size(p_post,1),size(v,2))
    h = vcat(v, c)
    z = mean(m.generator, h)
    zt = vcat(z, x)
    prediction = m.decoder(zt)

    Flux.reset!(m.prepool_net)
    Flux.reset!(m.decoder)

    return prediction
end

"""
    pool_encoding(m::PoolModel, x; post=true)

Returns the one-vector summary encoding for a bag.
If `post=true`, takes the bag through pre-pool network,
pooling function and post-pool network. If `post=false`,
skips the post-pool network transformation.
"""
function pool_encoding(m::PoolModel, _x; post=true)
    # get only u_out = 0 values
    b = _x[3,:]
    x = _x[1:2, b .== 0]

    v = m.prepool_net(x)
    p = m.poolf(v)
    if post
        return m.postpool_net(p)
    else
        return p
    end
end

#########################
### Scoring functions ###
#########################

"""
    score(model, x, p)

Scores the data for the PoolModel.
"""
function score(model::PoolModel, x::Matrix, p::Matrix)
    b = x[3,:]
    pred = predict(model, x)
    p_true = p[1, b .== 0]'
    Flux.mae(p_true, pred)
end
score(X::Vector{Matrix{Float32}}, P::Vector{Matrix{Float32}}) = mean(map((x,p) -> score(x,p), X, P))