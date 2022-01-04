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
seed = rand(1:1000)
data = load_data_vectors(;seed=seed);
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
    hdim = sample(2 .^ [6,7,8,9])
    act = "relu" # sample(["swish", "relu", "tanh", "sigmoid"])
    err = sample(["mae", "mse"])
    div = true # sample([true, false])
    β = sample(10:10:100)
    return hdim, act, err, div, β
end

# get parameters
idim = length(X_train[1])
odim = length(P_train[1])
hdim, act, err, div, β = sample_params()
err_fun = eval(:($(Symbol(err))))
activation = eval(:($(Symbol(act))))

# create the model
model = Chain(Dense(idim, hdim, activation),Dense(hdim,hdim,activation),Dense(hdim,odim))

# loss functions
"""
    lossfun(X, P; err::Function=mae)

Calculates simple MAE or MSE loss between the predicted values and true values.
"""
lossfun(X, P; err::Function=mae) = err(model(X), P)

"""
    lossfun_div(X, P, b; err::Function=mae,β=10)

Calculates MAE or MSE loss between the predicted values and true values
based on whether the breath goes in or out. Gives more importance to 
breathe-in values given β.
"""
function lossfun_div(X, P, b; err::Function=mae,β=10)
    W = model(X)
    Win, Wout = W[b .== 0], W[b .== 1]
    Pin, Pout = P[b .== 0], P[b .== 1]
    β*err(Win, Pin) + err(Wout,Pout)
end

# if div=true, uses the loss with more importance to breathe-in
if div
    loss(X, P, b) = lossfun_div(X, P, b; err=err_fun, β=β)
else
    loss(X, P, b) = lossfun(X, P; err=err_fun)
end

# definition of score functions (results are score with MAE)
score(X, P) = Flux.mae(model(X), P)
score(X, P, b) = Flux.mae(model(X)[b .== 0], P[b .== 0])
best_score(X, P, b) = Flux.mae(best_model(X)[b .== 0], P[b .== 0])

# initialize optimizer and other values
opt = ADAM()
best_val_score = Inf
best_model = deepcopy(model)
start_time = time()
if isempty(ARGS)
    max_train_time = 60*60*11.7
else
    max_train_time = parse(Float64, ARGS[1])
end
k = 1

# start training
@info "Started training model with $hdim number of hidden neurons, $act activation function, $err loss (divided=$div)."
while true
    # create a minibatch and train the model on a minibatch
    batch = RandomBatch(X_train, P_train, B_train, batchsize=64)
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
train_sc = mean(best_score.(X_train, P_train, B_train))
val_sc = mean(best_score.(X_val, P_val, B_val))
test_sc = mean(best_score.(X_test, P_test, B_test))

# save the best model and the parameters
using BSON
d = Dict(
    :model => best_model,
    :seed => seed,
    :hdim => hdim,
    :activation => act,
    :loss => err,
    :divide => div,
    :beta => β,
    :train_score => train_sc,
    :val_score => val_sc,
    :test_score => test_sc
)
name = savename("model", d, "bson")
safesave(datadir("models", name), d)