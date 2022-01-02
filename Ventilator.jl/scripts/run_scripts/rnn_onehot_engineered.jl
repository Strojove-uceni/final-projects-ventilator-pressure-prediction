# packages
using DrWatson
@quickactivate
using Ventilator
using Flux
using Base.Iterators: repeated
using Flux: throttle, @epochs, mae, mse
using StatsBase

# load data and unpack it
seed = 1
data = load_data_bags_engineered(;seed=seed, rnn=true);
X_train, P_train = data[1];
X_val, P_val = data[2];
X_test, P_test = data[3];
B_train, B_val, B_test = data[4];
x, p = X_train[1], P_train[1]
@info "Data loaded."

######################################
### Initialize necessary functions ###
######################################

function sample_params()
    par_vec = (2 .^ [5,6,7,8,9], 1:4, 1:4, ["swish", "relu", "elu"], ["LSTM", "GRU"])
    argnames = (:hdim, :rnn_layers, :dense_layers, :activation, :rnn_cell)
    parameters = (;zip(argnames, map(x->sample(x, 1)[1], par_vec))...)
    return parameters
end

function loss(x, y)
    Flux.reset!(model)
    mean(Flux.mae(model(xi), yi) for (xi, yi) in zip(x, y))
end

function predict(model, x)
    Flux.reset!(model)
    vcat([model(xi) for xi in x]...)
end

function score_model(x, p)
    b = hcat(x...)[3,:]
    Flux.mae(predict(model, x)[b .== 0], vcat(p...)[b .== 0])
end

function best_score(x, p)
    b = hcat(x...)[3,:]
    Flux.mae(predict(best_model, x)[b .== 0], vcat(p...)[b .== 0])
end

##################
### Model init ###
##################

# sample parameters and create model
idim = size(X_train[1][1],1)
pvec = sample_params()
model = rnn_constructor(;idim = idim, odim = 1, pvec...)

# initialize optimizer, saving variables
opt = ADAM()
best_val_score = Inf
best_model = deepcopy(model)
if isempty(ARGS)
    max_train_time = 60*60*47
else
    max_train_time = parse(Float64, ARGS[1])
end
k = 1

################
### Training ###
################

start_time = time()
while true
    # create a minibatch and train the model on a minibatch
    batch = RandomBatch(X_train, P_train, batchsize=128)
    Flux.train!(loss, Flux.params(model), repeated((batch[1], batch[2]), 5), opt)

    # only save a best model which has the best score on validation data
    sc = mean(score_model.(X_val, P_val))
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

###################################
### Evaluation and results save ###
###################################

# calculate the score given the best model after training is finished
train_sc = mean(best_score.(X_train, P_train))
val_sc = mean(best_score.(X_val, P_val))
test_sc = mean(best_score.(X_test, P_test))

# save the best model and the parameters
using BSON
d = Dict(
    :model => best_model,
    :seed => seed,
    :hdim => pvec.hdim,
    :activation => pvec.activation,
    :rnn_layers => pvec.rnn_layers,
    :dense_layers => pvec.dense_layers,
    :cell => pvec.rnn_cell,
    :train_score => train_sc,
    :val_score => val_sc,
    :test_score => test_sc
)
name = savename("model", d, "bson")
safesave(datadir("RNN_onehot_engineered", name), d)