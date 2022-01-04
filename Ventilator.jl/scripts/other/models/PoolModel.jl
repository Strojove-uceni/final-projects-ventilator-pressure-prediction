# packages
using DrWatson
@quickactivate
using Ventilator
using Flux
using Base.Iterators: repeated
using Flux: throttle, @epochs, mae, mse
using StatsBase
using Ventilator
include(srcdir("PoolModel.jl"))

using Plots
# ENV["GKSwstype"] = "100"

# load data and unpack it
# seed = rand(1:1000)
seed = 1
data = load_data_bags(;seed=seed);
# data = load_data_bags_onehot(;seed=seed);
X_train, P_train = data[1];
X_val, P_val = data[2];
X_test, P_test = data[3];
B_train, B_val, B_test = data[4];
x, p, b = X_train[1], P_train[1], B_train[1];
@info "Data loaded."


# parameters
idim = size(x, 1)
odim = 1
hdim = 256
predim = 32
postdim = 32
edim = 32
nlayers = 3
cell = "Dense"
cell = "LSTM"
activation = "swish"

# construct model
model = pm_constructor(;idim=idim, odim=odim, hdim=hdim, predim=predim, postdim=postdim, edim=edim,cell=cell)

# loss function and optimizer
loss(x, p) = pm_loss(model, x, p; lossf=Flux.mae)
score(x::Matrix, p::Matrix) = score(model, x, p)
loss(x, p)
score(x, p)
# tahle ztráta spíš nefunguje
# loss(x, p) = pm_loss_full(model, x, p; lossf=Flux.mae)
opt = ADAM()

best_val_score = Inf
best_model = deepcopy(model)
max_train_time = 60*60
k = 1

start_time = time()
while true
    # create a minibatch and train the model on a minibatch
    batch = RandomBatch(X_train, P_train, batchsize=64)
    Flux.train!(loss, Flux.params(model), zip(batch...),opt)

    # only save a best model which has the best score on validation data
    l = mean(score.(X_val, P_val))
    if l < best_val_score
        global best_model = deepcopy(model)
        global best_val_score = l
        @info "Epoch $k: score = $(round(l, digits=3))"
    end
    global k += 1

    # stop when training time is exceeded
    if (time() - start_time > max_train_time)
        @info "Stopped training, time limit exceeded."
        break
    end
end

# plot some of the results
function plot_lungs(X_val, P_val, model)
    p = Plots.Plot{Plots.GRBackend}[]
    for k in 1:4
        i = rand(1:7500)
        X = X_val[i]
        P = P_val[i]
        sc = score(X, P)
        W = predict(model, X)

        plot(P[1,X[3,:] .== 0], marker=:circle, markersize=2,label="", title="$(round(sc, digits=3)), $(round(Flux.mse(P[1,X[3,:] .== 0]', W), digits=3))");
        px = plot!(W', marker=:square, markersize=2, label="")
        push!(p, px)
    end
    plot(p...,layout=(2,2))
end
plot_lungs(X_val, P_val, best_model)

"""
Here is the problem: the values of pressure are not very similar,
some are fairly small, some are larger. This means that the model
does not treat the datapoint similarly when training.

We could - and probably should add some regularization or
standartization to make training better.
We can divide all point in the set with its maximum, save the max
value and then use it to "decode" the true values again. It would
also make it possible to use other activation functions since all
values would be between [0,1].

Without the regulatization, the model is not pushed to actually fit
the data accordingly to minimize and learn the SHAPES of the curves.

We can also add LSTM layers to the model - not sure if it can help
anyhow, but it is worth a try.
"""