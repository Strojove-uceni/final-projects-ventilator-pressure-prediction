# packages
using DrWatson
@quickactivate
using Ventilator
using Flux
using Base.Iterators: repeated
using Flux: throttle, @epochs, mae, mse
using StatsBase
using Ventilator

using Plots
ENV["GKSwstype"] = "100"

# load data and unpack it
# seed = rand(1:1000)
seed = 1
data = load_data_bags(;seed=seed);
X_train, P_train = data[1];
X_val, P_val = data[2];
X_test, P_test = data[3];
B_train, B_val, B_test = data[4]
@info "Data loaded."

"""
    sample_params()

Samples parameters - hdim, activation function, MAE vs MSE,
divided or not, scaling term etc.
"""
function sample_params()
    hdim = StatsBase.sample(2 .^ [4,5,6,7])
    err = StatsBase.sample(["mae", "mse"])
    return hdim, err
end

# get parameters
idim = size(X_train[1],1)
odim = size(P_train[1],1)
hdim, err = sample_params()
err_fun = eval(:($(Symbol(err))))
# activation = eval(:($(Symbol(act))))

# create the model
# model = Chain(Dense(idim, hdim, activation),Dense(hdim,hdim,activation),Dense(hdim,odim))
model = Chain(LSTM(idim, hdim),LSTM(hdim,hdim), Dense(hdim,odim))

# loss functions
"""
    lossfun(model, X, P; err::Function=mae)

Calculates simple MAE or MSE loss between the predicted values and true values.
"""
function lossfun(model, X, P; err::Function=mae)
    err(model(X), P)
end

# define the loss itself
loss(X, P) = lossfun(model, X, P; err=err_fun)

# definition of score functions (results are score with MAE)
score(X, P) = Flux.mae(model(X), P)
score(X, P, B) = Flux.mae(model(X)[B .== 0], P[B .== 0])
best_score(model, X, P, B) = Flux.mae(model(X)[B .== 0], P[B .== 0])

# initialize optimizer and other values
opt = ADAM()
best_val_score = Inf
best_model = deepcopy(model)
max_train_time = 60*120
k = 1

# start training
start_time = time()
@info "Started training model with $hdim number of hidden neurons and $err loss."
while true
    # create a minibatch and train the model on a minibatch
    batch = RandomBatch(X_train, P_train, batchsize=64)
    Flux.train!(loss, Flux.params(model), zip(batch...),opt)

    # only save a best model which has the best score on validation data
    l = mean(score.(X_val, P_val, B_val))
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

# calculate the score given the best model after training is finished
train_sc = mean(map((x, p, b) -> best_score(best_model, x, p, b), X_train, P_train, B_train))
val_sc = mean(map((x, p, b) -> best_score(best_model, x, p, b), X_val, P_val, B_val))
test_sc = mean(map((x, p, b) -> best_score(best_model, x, p, b), X_test, P_test, B_test))

# save the best model and the parameters
using BSON
d = Dict(
    :model => best_model,
    :seed => seed,
    :hdim => hdim,
    :loss => err,
    :train_score => train_sc,
    :val_score => val_sc,
    :test_score => test_sc
)
name = savename("model", d, "bson")
safesave(datadir("models", name), d)

# plot some of the results
function plot_lungs(X_val, P_val)
    p = Plots.Plot{Plots.GRBackend}[]
    for k in 1:4
        i = rand(1:7500)
        X = X_val[i]
        P = P_val[i]
        W = model(X)

        plot(P, marker=:circle, markersize=2,label="");
        px = plot!(W, marker=:square, markersize=2, label="")
        push!(p, px)
    end
    plot(p...,layout=(2,2))
end
p = plot_lungs(X_val, P_val)
safesave(plotsdir("predictions.png"),p)