using DrWatson
@quickactivate
using DataFrames, CSV
using Random
using StatsBase

"""
    load_data_bags(;seed=nothing, vars::Vector{Symbol}=[:time_step, :u_in, :u_out, :R, :C], rnn=false)

Loads `train.csv` data, groups them by `:breat_id` and prepares them in the desired format
as bags of sequences. If `rnn = true`, further divides the data to be used in RNN models.
Divided the data to train/validation/test in ratio 0.8/0.1/0.1.
"""
function load_data_bags(;seed=nothing, vars::Vector{Symbol}=[:time_step, :u_in, :u_out, :R, :C], rnn=false)

    data = CSV.read(datadir("train.csv"), DataFrame)
    groups = groupby(data, :breath_id)

    if rnn
        X_data, P_data, B_data = Vector{Vector{Float32}}[], Vector{Vector{Float32}}[], Array{Float32,2}[] #Array{Float32,2}[], Array{Float32,2}[], Array{Float32,2}[]
    else
        X_data, P_data, B_data = Array{Float32,2}[], Array{Float32,2}[], Array{Float32,2}[]
    end
    @time for g in groups
        x = collect(g[!, vars] |> Array |> transpose)
        p = collect(g[:, :pressure]')

        if rnn
            push!(X_data, [xi for xi in eachcol(x)])
            push!(P_data, [pi for pi in eachcol(p)])
        else
            push!(X_data, x)
            push!(P_data, p)
        end

        b = collect(g[:, :u_out]')
        push!(B_data, b)
    end

    # set seed
    (seed == nothing) ? nothing : Random.seed!(seed)

    idx = sample(1:75000, 75000, replace=false)
    tix, val_ix, test_ix = idx[1:60000], idx[60001:67500], idx[67501:end]
    X_train, P_train, B_train = X_data[tix], P_data[tix], B_data[tix]
    X_val, P_val, B_val = X_data[val_ix], P_data[val_ix], B_data[val_ix]
    X_test, P_test, B_test = X_data[test_ix], P_data[test_ix], B_data[test_ix]

    # reset seed
	(seed !== nothing) ? Random.seed!() : nothing

    return (X_train, P_train), (X_val, P_val), (X_test, P_test), (B_train, B_val, B_test)
end

"""
    load_data_bags_onehot(;seed=nothing, vars::Vector{Symbol}=[:time_step, :u_in, :u_out, :R, :C], rnn=false)

Loads `train.csv` data, groups them by `:breat_id` and prepares them in the desired format
as bags of sequences. If `rnn = true`, further divides the data to be used in RNN models.
Divided the data to train/validation/test in ratio 0.8/0.1/0.1. Uses R and C as onehot encoding.
"""
function load_data_bags_onehot(;seed=nothing, vars::Vector{Symbol}=[:time_step, :u_in, :u_out], rnn=false)
    data = CSV.read(datadir("train.csv"), DataFrame)
    groups = groupby(data, :breath_id)
    Rs, Cs = unique(data[:, :R]), unique(data[:, :C])

    if rnn
        X_data, P_data, B_data = Vector{Vector{Float32}}[], Vector{Vector{Float32}}[], Array{Float32,2}[] #Array{Float32,2}[], Array{Float32,2}[], Array{Float32,2}[]
    else
        X_data, P_data, B_data = Array{Float32,2}[], Array{Float32,2}[], Array{Float32,2}[]
    end
    @time for g in groups
        x = collect(g[!, vars] |> Array |> transpose)
        p = collect(g[:, :pressure]')
        R, C = Flux.onehot(g[1,:R], Rs), Flux.onehot(g[1,:C], Cs)
        Rm = reshape(repeat(R, 80), 3, 80)
        Cm = reshape(repeat(C, 80), 3, 80)
        xfull = vcat(x, Rm, Cm)

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

    # set seed
    (seed == nothing) ? nothing : Random.seed!(seed)

    idx = sample(1:75000, 75000, replace=false)
    tix, val_ix, test_ix = idx[1:60000], idx[60001:67500], idx[67501:end]
    X_train, P_train, B_train = X_data[tix], P_data[tix], B_data[tix]
    X_val, P_val, B_val = X_data[val_ix], P_data[val_ix], B_data[val_ix]
    X_test, P_test, B_test = X_data[test_ix], P_data[test_ix], B_data[test_ix]

    # reset seed
	(seed !== nothing) ? Random.seed!() : nothing

    return (X_train, P_train), (X_val, P_val), (X_test, P_test), (B_train, B_val, B_test)
end

"""
    load_data_bags_engineered(;seed=nothing, vars::Vector{Symbol}=[:time_step, :u_in, :u_out], rnn=false)

Loads `train.csv` data, groups them by `:breat_id` and prepares them in the desired format
as bags of sequences. Uses some feature engineering, summary statistics and difference values.
If `rnn = true`, further divides the data to be used in RNN models.
Divided the data to train/validation/test in ratio 0.8/0.1/0.1.
"""
function load_data_bags_engineered(;seed=nothing, vars::Vector{Symbol}=[:time_step, :u_in, :u_out], rnn=false)
    data = CSV.read(datadir("train.csv"), DataFrame)
    groups = groupby(data, :breath_id)
    Rs, Cs = unique(data[:, :R]), unique(data[:, :C])

    if rnn
        X_data, P_data, B_data = Vector{Vector{Float32}}[], Vector{Vector{Float32}}[], Array{Float32,2}[] #Array{Float32,2}[], Array{Float32,2}[], Array{Float32,2}[]
    else
        X_data, P_data, B_data = Array{Float32,2}[], Array{Float32,2}[], Array{Float32,2}[]
    end
    @time for g in groups
        # the engineering part
        _x = collect(g[!, vars] |> Array |> transpose)
        x_in = Ventilator.pool_source(_x; u = 0, rc = false)
        X_in = reshape(repeat(x_in, 80), length(x_in), 80)
        x_out = Ventilator.pool_source(_x; u = 1, rc = false)
        X_out = reshape(repeat(x_out, 80), length(x_out), 80)
        X = vcat(_x, X_in, X_out)

        p = collect(g[:, :pressure]')
        R, C = Flux.onehot(g[1,:R], Rs), Flux.onehot(g[1,:C], Cs)
        Rm = reshape(repeat(R, 80), 3, 80)
        Cm = reshape(repeat(C, 80), 3, 80)
        xfull = vcat(X, Rm, Cm)

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

    # set seed
    (seed == nothing) ? nothing : Random.seed!(seed)

    idx = sample(1:75000, 75000, replace=false)
    tix, val_ix, test_ix = idx[1:60000], idx[60001:67500], idx[67501:end]
    X_train, P_train, B_train = X_data[tix], P_data[tix], B_data[tix]
    X_val, P_val, B_val = X_data[val_ix], P_data[val_ix], B_data[val_ix]
    X_test, P_test, B_test = X_data[test_ix], P_data[test_ix], B_data[test_ix]

    # reset seed
	(seed !== nothing) ? Random.seed!() : nothing

    return (X_train, P_train), (X_val, P_val), (X_test, P_test), (B_train, B_val, B_test)
end

"""
    load_data_bags_engineered_lags(;seed=nothing, vars::Vector{Symbol}=[:time_step, :u_in, :u_out, :R, :C], rnn=false)

Loads `train.csv` data, groups them by `:breat_id` and prepares them in the desired format
as bags of sequences. Uses some feature engineering, summary statistics and difference values.
If `rnn = true`, further divides the data to be used in RNN models.
Divided the data to train/validation/test in ratio 0.8/0.1/0.1.
"""
function load_data_bags_engineered_lags(;seed=nothing, vars::Vector{Symbol}=[:time_step, :u_in, :u_out, :R, :C], rnn=false)
    data = CSV.read(datadir("train.csv"), DataFrame)
    groups = groupby(data, :breath_id)
    Rs, Cs = unique(data[:, :R]), unique(data[:, :C])

    if rnn
        X_data, P_data, B_data = Vector{Vector{Float32}}[], Vector{Vector{Float32}}[], Array{Float32,2}[] #Array{Float32,2}[], Array{Float32,2}[], Array{Float32,2}[]
    else
        X_data, P_data, B_data = Array{Float32,2}[], Array{Float32,2}[], Array{Float32,2}[]
    end
    @time for g in groups
        # the engineering part
        _x = collect(g[!, vars] |> Array |> transpose)
        xf0, x0 = pool_feature_add(_x; u = 0)
        # X_in = reshape(repeat(x_in, 80), length(x_in), 80)
        xf1, x1 = pool_feature_add(_x; u = 1)
        # X_out = reshape(repeat(x_out, 80), length(x_out), 80)
        X = vcat(hcat(x0,x1), xf0, xf1)

        p = collect(g[:, :pressure]')
        R, C = Flux.onehot(g[1,:R], Rs), Flux.onehot(g[1,:C], Cs)
        Rm = reshape(repeat(R, 80), 3, 80)
        Cm = reshape(repeat(C, 80), 3, 80)
        xfull = vcat(X, Rm, Cm)

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

    # set seed
    (seed == nothing) ? nothing : Random.seed!(seed)

    idx = sample(1:75000, 75000, replace=false)
    tix, val_ix, test_ix = idx[1:60000], idx[60001:67500], idx[67501:end]
    X_train, P_train, B_train = X_data[tix], P_data[tix], B_data[tix]
    X_val, P_val, B_val = X_data[val_ix], P_data[val_ix], B_data[val_ix]
    X_test, P_test, B_test = X_data[test_ix], P_data[test_ix], B_data[test_ix]

    # reset seed
	(seed !== nothing) ? Random.seed!() : nothing

    return (X_train, P_train), (X_val, P_val), (X_test, P_test), (B_train, B_val, B_test)
end

"""
    load_data_vectors(;seed=nothing)

Loads `train.csv` data, groups them by `:breat_id` and prepares them in the desired format
as vectors. Divided the data to train/validation/test in ratio 0.8/0.1/0.1.
"""
function load_data_vectors(;seed=nothing)

    data = CSV.read(datadir("train.csv"), DataFrame)
    groups = groupby(data, :breath_id)

    X_data, P_data, B_data = Array{Float32,1}[], Array{Float32,1}[], Array{Float32,1}[]
    @time for g in groups
        x = reshape(g[!, [:time_step, :u_in]] |> Array, 80*2)
        R, C = g[1,:R], g[1,:C]
        push!(X_data, vcat(x, R, C))
        p = g[:, :pressure]
        push!(P_data, p)
        b = g[:, :u_out]
        push!(B_data, b)
    end

    # set seed
    (seed == nothing) ? nothing : Random.seed!(seed)

    idx = sample(1:75000, 75000, replace=false)
    tix, val_ix, test_ix = idx[1:60000], idx[60001:67500], idx[67501:end]
    X_train, P_train, B_train = X_data[tix], P_data[tix], B_data[tix]
    X_val, P_val, B_val = X_data[val_ix], P_data[val_ix], B_data[val_ix]
    X_test, P_test, B_test = X_data[test_ix], P_data[test_ix], B_data[test_ix]

	# reset seed
	(seed !== nothing) ? Random.seed!() : nothing

    return (X_train, P_train), (X_val, P_val), (X_test, P_test), (B_train, B_val, B_test)
end

"""
    load_data_vectors_onehot(;seed=nothing)

Loads `train.csv` data, groups them by `:breat_id` and prepares them in the desired format
as vectors. Uses onehot encoding for R, C. Divided the data to train/validation/test in ratio 0.8/0.1/0.1.
"""
function load_data_vectors_onehot(;seed=nothing)

    data = CSV.read(datadir("train.csv"), DataFrame)
    Rs, Cs = unique(data[:, :R]), unique(data[:, :C])
    groups = groupby(data, :breath_id)

    X_data, P_data, B_data = Array{Float32,1}[], Array{Float32,1}[], Array{Float32,1}[]
    @time for g in groups
        x = reshape(g[!, [:time_step, :u_in]] |> Array, 80*2)
        R, C = Flux.onehot(g[1,:R], Rs), Flux.onehot(g[1,:C], Cs)
        push!(X_data, vcat(x, R, C))
        p = g[:, :pressure]
        push!(P_data, p)
        b = g[:, :u_out]
        push!(B_data, b)
    end

    # set seed
    (seed == nothing) ? nothing : Random.seed!(seed)

    idx = sample(1:75000, 75000, replace=false)
    tix, val_ix, test_ix = idx[1:60000], idx[60001:67500], idx[67501:end]
    X_train, P_train, B_train = X_data[tix], P_data[tix], B_data[tix]
    X_val, P_val, B_val = X_data[val_ix], P_data[val_ix], B_data[val_ix]
    X_test, P_test, B_test = X_data[test_ix], P_data[test_ix], B_data[test_ix]

	# reset seed
	(seed !== nothing) ? Random.seed!() : nothing

    return (X_train, P_train), (X_val, P_val), (X_test, P_test), (B_train, B_val, B_test)
end

"""
    load_data_single(;seed=nothing)

Loads `train.csv` data as single instances, no sequence grouping present.
Divided the data to train/validation/test in ratio 0.8/0.1/0.1.
"""
function load_data_single(;seed=nothing)

    data = CSV.read(datadir("train.csv"), DataFrame)
    groups = groupby(data, :breath_id)

    _X_data = data[:, [:time_step, :u_in, :u_out, :R, :C]] |> Array |> transpose |> collect
    X_data = [_X_data[:,i] for i in 1:size(_X_data,2)]
    P_data = data[:, :pressure]

    # set seed
    (seed == nothing) ? nothing : Random.seed!(seed)

    l = length(P_data)
    a = Int(l*0.8)
    b = (l - a) / 2 |> Int
    idx = sample(1:l, l, replace=false)
    tix, val_ix, test_ix = idx[1:a], idx[a+1:a+b], idx[a+b+1:end]
    X_train, P_train = X_data[tix], P_data[tix]
    X_val, P_val = X_data[val_ix], P_data[val_ix]
    X_test, P_test = X_data[test_ix], P_data[test_ix]

	# reset seed
	(seed !== nothing) ? Random.seed!() : nothing

    return (X_train, P_train), (X_val, P_val), (X_test, P_test)
end