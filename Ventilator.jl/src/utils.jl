"""
    RandomBatch(xdata,ydata;batchsize::Int=64)

Samples random batch of `batchsize` size from data X, Y.
"""
function RandomBatch(xdata,ydata;batchsize::Int=64)
    l = length(xdata)
	if batchsize > l
		return (xdata,ydata)
	end
    idx = sample(1:l,batchsize)
    return xdata[idx],ydata[idx]
end

"""
    RandomBatch(xdata,ydata,bdata;batchsize::Int=64)

Samples random batch of `batchsize` size from data X, Y, B,
where B is a vector of boolean values indicating value of `u_out`.
"""
function RandomBatch(xdata,ydata,bdata;batchsize::Int=64)
    l = length(xdata)
	if batchsize > l
		return (xdata,ydata,bdata)
	end
    idx = sample(1:l,batchsize)
    return xdata[idx],ydata[idx],bdata[idx]
end

"""
    RandomBatch(xdata,ydata,bdata;batchsize::Int=64)

Samples random batch of `batchsize` size from data X, Y, B,
where B is a vector of boolean values indicating value of `u_out`.
"""
function RandomBatch(Xd::Vector{Vector{Vector{Float32}}}, Yd::Vector{Vector{Vector{Float32}}}; batchsize=64)
    l = length(Xd)
    idx = sample(1:l, batchsize, replace=false)
    X = Xd[idx]
    Y = Yd[idx]
    x = [hcat([X[i][j] for i in 1:batchsize]...) for j in 1:80]
    y = [hcat([Y[i][j] for i in 1:batchsize]...) for j in 1:80]
    (x, y)
end

####################
### Constructors ###
####################

using Flux

"""
	function build_mlp(idim::Int, hdim::Int, odim::Int, nlayers::Int; activation::String="relu", lastlayer::String="")

Creates a chain with `nlayers` layers of `hdim` neurons with transfer function `activation`.
input and output dimension is `idim` / `odim`
If lastlayer is no specified, all layers use the same function.
If lastlayer is "linear", then the last layer is forced to be Dense.
It is also possible to specify dimensions in a vector.
"""
build_mlp(ks::Vector{Int}, fs::Vector) = Flux.Chain(map(i -> Dense(i[2],i[3],i[1]), zip(fs,ks[1:end-1],ks[2:end]))...)
build_mlp(ks::Vector{Int}, fs::Vector, Cell) = Flux.Chain(map(i -> Cell(i[2],i[3]), zip(fs,ks[1:end-1],ks[2:end]))...)
build_mlp(idim::Int, hdim::Int, odim::Int, nlayers::Int; kwargs...) =
	build_mlp(vcat(idim, fill(hdim, nlayers-1)..., odim); kwargs...)
function build_mlp(ks::Vector{Int}; activation::String = "relu", lastlayer::String = "", cell::String = "Dense")
	activation = (activation == "linear") ? "identity" : activation
	fs = Array{Any}(fill(eval(:($(Symbol(activation)))), length(ks) - 1))
	if !isempty(lastlayer)
		fs[end] = (lastlayer == "linear") ? identity : eval(:($(Symbol(lastlayer))))
	end
    if cell == "Dense"
	    return build_mlp(ks, fs)
    else
        Cell = eval(:($(Symbol(cell))))
        return build_mlp(ks, fs, Cell)
    end
end

function rnn_constructor(;idim::Int, hdim::Int, odim::Int, rnn_layers::Int=2, dense_layers::Int=2,
                            activation::String="relu", rnn_cell::String="LSTM")

    rnn_chain = build_mlp(idim, hdim, hdim, rnn_layers, cell=rnn_cell)
    if dense_layers > 1
        dense_chain = build_mlp(hdim, hdim, hdim, dense_layers-1, activation=activation)
        model = Chain(
            rnn_chain...,
            dense_chain...,
            Dense(hdim, odim)
        )
        return model
    else
        model = Chain(
            rnn_chain...,
            Dense(hdim, odim)
        )
        return model
    end
end