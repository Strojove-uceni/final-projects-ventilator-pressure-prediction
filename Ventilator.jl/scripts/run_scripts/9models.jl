################
### Packages ###
################

using DrWatson
@quickactivate
using Ventilator
using Flux
using Base.Iterators: repeated
using Flux: throttle, @epochs, mae, mse
using StatsBase

using CSV, DataFrames, Random

############
### Data ###
############

"""
    load_data_RC(;seed=nothing, vars::Vector{Symbol}=[:time_step, :u_in, :u_out, :R, :C], rnn = true)

Loads the `train.csv` dataset and groups the data based on the combination of R, C.
Creates a dataset as a dictionary with keys being `Symbol("R=R, C=C)`. Each key
value is another dictionary with keys being `:train, :val, :test, B`. Each key has
data as value, train, val and test have Tuple of (X, Y) and B has a Tuple of u_out
binary vectors for train, val and test data.
"""
function load_data_RC(;seed=nothing, vars::Vector{Symbol}=[:time_step, :u_in, :u_out, :R, :C], rnn = true)
    data = CSV.read(datadir("train.csv"), DataFrame)
    RC_groups = groupby(data, [:R, :C])

    dataset = Dict()
    for g in RC_groups
        df = g |> DataFrame
        train, val, test, B = train_val_test_split(df; seed = seed, vars = vars, rnn = rnn)
        k = Symbol("R=$(g[1,:R]), C=$(g[1,:C])")
        d = Dict(
            :train => train,
            :val => val,
            :test => test,
            :B => B
        )
        push!(dataset, k => d)
    end

    return dataset
end


"""
    train_val_test_split(df::DataFrame; seed=nothing, vars::Vector{Symbol}=[:time_step, :u_in, :u_out, :R, :C], rnn=false)

Divides the data to train/validation/test splits of 0.8:0.1:0.1 ratios.
"""
function train_val_test_split(df::DataFrame; seed=nothing, vars::Vector{Symbol}=[:time_step, :u_in, :u_out, :R, :C], rnn=false)
    # group by breath_id
    groups = groupby(df, :breath_id)

    # allocate vectors
    if rnn
        X_data, P_data, B_data = Vector{Vector{Float32}}[], Vector{Vector{Float32}}[], Array{Float32,2}[]
    else
        X_data, P_data, B_data = Array{Float32,2}[], Array{Float32,2}[], Array{Float32,2}[]
    end

    for g in groups
        # the feature engineering part
        _x = collect(g[!, vars] |> Array |> transpose)
        xf0, x0 = pool_feature_add(_x; u = 0)
        xf1, x1 = pool_feature_add(_x; u = 1)
        xfull = vcat(hcat(x0,x1), xf0, xf1)

        # pressure values
        p = collect(g[:, :pressure]')

        # push to allocated vectors
        if rnn
            push!(X_data, [xi for xi in eachcol(xfull)])
            push!(P_data, [pi for pi in eachcol(p)])
        else
            push!(X_data, xfull)
            push!(P_data, p)
        end

        b = collect(g[:, :u_out]')
        push!(B_data, b)
    end

    # get lengths of train/val/test data
    n = length(groups)
    n1 = round(Int, n*0.8)
    n2 = (n - n1) ÷ 2

    # set seed
    (seed == nothing) ? nothing : Random.seed!(seed)

    # randomly sample indexes
    idx = sample(1:n, n, replace=false)
    tix, val_ix, test_ix = idx[1:n1], idx[n1+1:n1+n2], idx[n1+n2+1:end]

    # split data
    X_train, P_train, B_train = X_data[tix], P_data[tix], B_data[tix]
    X_val, P_val, B_val = X_data[val_ix], P_data[val_ix], B_data[val_ix]
    X_test, P_test, B_test = X_data[test_ix], P_data[test_ix], B_data[test_ix]

    # reset seed
	(seed !== nothing) ? Random.seed!() : nothing

    return (X_train, P_train), (X_val, P_val), (X_test, P_test), (B_train, B_val, B_test)
end

# load dataset
seed = 1
dataset = load_data_RC(;seed = seed)
@info "Data loaded."

#################
### Functions ###
#################

# function to sample parameters
function sample_params()
    par_vec = (2 .^ [5,6,7,8,9], 1:4, 1:4, ["swish", "relu", "elu"], ["LSTM", "GRU"])
    argnames = (:hdim, :rnn_layers, :dense_layers, :activation, :rnn_cell)
    parameters = (;zip(argnames, map(x->sample(x, 1)[1], par_vec))...)
    return parameters
end

# loss function
function lossf(model, x, y)
    Flux.reset!(model)
    mean(Flux.mae(model(xi), yi) for (xi, yi) in zip(x, y))
end

# function to predict pressure
function predict(model, x)
    Flux.reset!(model)
    vcat([model(xi) for xi in x]...)
end

# function to score model
function score_model(model, x, p, b)
    b = reshape(b, 80)
    Flux.mae(predict(model, x)[b .== 0], vcat(p...)[b .== 0])
end

##################
### Model init ###
##################

# sample parameters and create model
d = dataset[Symbol("R=5, C=20")][:train][1];
idim = size(d[1][1],1)
pvec = sample_params()
# create one model
model = rnn_constructor(;idim = idim, odim = 1, pvec...)
# deepcopy it 9 times (number of unique [R, C])
models = Dict([(k, deepcopy(model)) for (i,k) in enumerate(keys(dataset))])

# get keys to have them enumerated
ks = Dict([(i,k) for (i,k) in enumerate(keys(dataset))])

# now we need 9 optimizers (one for each model)
opts = Dict([(k, ADAM()) for k in keys(dataset)])

################
### Training ###
################

# prerequisities for training
best_val_scores = Dict([(k, Inf) for k in keys(dataset)])
best_models = deepcopy(models)
if isempty(ARGS)
    max_train_time = 60*60*24 # 24 training hours
else
    max_train_time = parse(Float64, ARGS[1])
end
k = 1

# and now to the training
@info "Starting training."
start_time = time()
while true
    better = false
    # use multithreading on more CPU cores
    Threads.@threads for i in 1:9
    # for i in 1:9
        # get data based on key
        X, Y = dataset[ks[i]][:train]
        
        # model & optimiser based on key
        m = models[ks[i]]
        opt = opts[ks[i]]

        # loss function with chosen model
        loss(x, y) = lossf(m, x, y)
        
        # parameters
        ps = Flux.params(m)
    
        # sample batch and train
        batch = RandomBatch(X, Y)
        Flux.train!(loss, ps, repeated((batch[1], batch[2]), 5), opt)

        # validation score
        Xv, Yv = dataset[ks[i]][:val]
        Bv = dataset[ks[i]][:B][2]

        val_score = mean(map((x, y, b) -> score_model(m, x, y, b), Xv, Yv, Bv))

        if val_score < best_val_scores[ks[i]]
            best_models[ks[i]] = deepcopy(m)
            best_val_scores[ks[i]] = val_score
            better = true
        end
    end

    # print best scores if some scores changed
    if better
        for (key, v) in best_val_scores
            println("key: $key, score = $(round(v, digits=3))")
        end
        l = round(mean(values(best_val_scores)), digits=4)
        @show k l
    end
    global k += 1
    
    # stop when training time is exceeded
    if (time() - start_time > max_train_time)
        @info "Stopped training, time limit exceeded."
        break
    end
end

"""
    get_scores(dataset, models, ks, type::Symbol=:val)

Calculates the scores for all models and all 9 subsets of datasets.
Variable `type` controls whether to use train, validation or test data.
Returns a vector of scores (without R, C combination).
"""
function get_scores(dataset, models, ks, type::Symbol=:val)
    scores = Dict([(k, 0f0) for k in keys(dataset)])
    
    Threads.@threads for i in 1:9
        # model based on key
        m = models[ks[i]]

        # score
        Xv, Yv = dataset[ks[i]][type]
        num = Dict(:train => 1, :val => 2, :test => 3)
        Bv = dataset[ks[i]][:B][num[type]]
        score = mean(map((x, y, b) -> score_model(m, x, y, b), Xv, Yv, Bv))
        
        scores[ks[i]] = score
    end
    
    return scores
end

# calculate train, validation, test scores
train_sc = get_scores(dataset, best_models, ks, :train)
val_sc = get_scores(dataset, best_models, ks, :val)
test_sc = get_scores(dataset, best_models, ks, :test)

train, val, test = map(x -> mean(values(x)), [train_sc, val_sc, test_sc])

println("""

Training finished with the following results:

train score      = $(round(train, digits=4))
validation score = $(round(val, digits=4))
test score       = $(round(test, digits=4))
""")

# save the best model and the parameters
using BSON
d = Dict(
    :models => best_models,
    :seed => seed,
    :hdim => pvec.hdim,
    :activation => pvec.activation,
    :rnn_layers => pvec.rnn_layers,
    :dense_layers => pvec.dense_layers,
    :cell => pvec.rnn_cell,
    :train => train_sc,
    :val => val_sc,
    :test => test_sc,
    :train_score => train,
    :val_score => val,
    :test_score => test
)
name = savename("model", d, "bson")
safesave(datadir("9models", name), d)