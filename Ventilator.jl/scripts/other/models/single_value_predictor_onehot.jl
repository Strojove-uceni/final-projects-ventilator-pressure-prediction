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
# ENV["GKSwstype"] = "100"

# load data and unpack it
seed = 1 #rand(1:1000)
data = load_data_single(;seed=seed);
X_train, P_train = data[1];
X_val, P_val = data[2];
X_test, P_test = data[3];
@info "Data loaded."

# load bag data
dt = load_data_bags(;seed=seed)
Xv, Pv = dt[2]

# model
idim = 5
hdim = 64
odim = 1
activation = relu
model = Chain(Dense(idim, hdim, activation),Dense(hdim,hdim,activation),Dense(hdim,odim, relu))

# loss function
loss(x, y) = Flux.mse(model(x), y)
loss(x, y)

# training
# initialize optimizer and other values
opt = ADAM()
best_val_score = Inf
best_model = deepcopy(model)
max_train_time = 60*60
k = 1

# start training
start_time = time()
@info "Started training model with $hdim number of hidden neurons and $activation activation function."
while true
    # create a minibatch and train the model on a minibatch
    batch = RandomBatch(X_train, P_train, batchsize=64)
    Flux.train!(loss, Flux.params(model), zip(batch...), opt)

    # only save a best model which has the best score on validation data
    # l = mean(loss.(X_val, P_val))
    # validation score
    sc = evaluate(Xv, Pv, model)
    if sc < best_val_score
        global best_model = deepcopy(model)
        global best_val_score = sc
        @info "Epoch $k: validation score = $(round(sc, digits=3))"
    end
    global k += 1

    # stop when training time is exceeded
    if (time() - start_time > max_train_time)
        @info "Stopped training, time limit exceeded."
        break
    end
end

# evaluate results on test dataset
evaluate(best_model, "test")

function score(X, P, best_model)
    prediction = best_model(X)
    b = X[3,:]
    Flux.mae(prediction[1,b .== 0], P[1,b .== 0])
end
score(X, P) = score(X, P, model)
function evaluate(best_model, set::String="test"; seed=nothing)
    if set == "test"
        ix = 3
    else
        ix = 2
    end
    # load bag data
    data = load_data_bags(;seed=seed);
    Xd, Pd = data[ix];
    @info "Bag data loaded."

    # calculate score
    sc = mean(map((x,p) -> score(x, p, best_model), Xd, Pd))
    @info "Score on validation dataset is: $sc."
    return sc
end
function evaluate(Xv, Pv, best_model; seed=nothing)
    # calculate score
    sc = mean(map((x,p) -> score(x, p, best_model), Xv, Pv))
    return sc
end