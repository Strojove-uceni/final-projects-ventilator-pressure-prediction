using Flux
using Base.Iterators: repeated
using Flux: throttle, @epochs
using StatsBase
using Ventilator

data = load_data(;seed=1)
X_train, P_train = data[1]
X_val, P_val = data[2]

function loss(_X, P)
    X = _X[1:2,:]
    Y = instance_net(X)
    R, C = _X[4:5,1]
    h = vcat(R, C)
    Z = hcat(map(y -> vcat(y, h), eachcol(Y))...)
    W = post_net(Z)
    Flux.mae(W, P)
end

function predict(_X)
    X = _X[1:2,:]
    Y = instance_net(X)
    R, C = _X[4:5,1]
    h = vcat(R, C)
    Z = hcat(map(y -> vcat(y, h), eachcol(Y))...)
    post_net(Z)
end
function predict(_X, best_model)
    X = _X[1:2,:]
    Y = best_model[:instance_net](X)
    R, C = _X[4:5,1]
    h = vcat(R, C)
    Z = hcat(map(y -> vcat(y, h), eachcol(Y))...)
    best_model[:post_net](Z)
end

function score(_X, P)
    W = predict(_X)
    Flux.mae(W[:,X[3,:] .== 0], P[:,X[3,:] .== 0])
end
function score(_X, P, best_model)
    W = predict(_X, best_model)
    Flux.mae(W[:,X[3,:] .== 0], P[:,X[3,:] .== 0])
end

instance_net = Chain(Dense(2, 64, relu), Dense(64, 64, relu), Dense(64, 32), softmax)
post_net = Chain(Dense(34, 64, relu), Dense(64, 64, relu), Dense(64, 1))
opt = ADAM()

@epochs 1000 begin
    batch = RandomBatch(X_train, P_train, batchsize=64)
    Flux.train!(loss, Flux.params(instance_net, post_net), zip(batch...), opt)
    println(mean(score.(X_val,P_val)))
end

i = rand(1:1000)
score(X_train[i],P_train[i])