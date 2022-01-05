# test data
# predictions

using DrWatson
@quickactivate
using CSV, DataFrames, Flux
using Ventilator

data = Ventilator.load_test_data(;rnn=true);
X, B = data;

df = collect_results(datadir("RNN"))
sort!(df, :val_score)
model = df[1,:model]

function predict(model, x)
    Flux.reset!(model)
    vcat([model(xi) for xi in x]...)
end

predict(model, X[1])

d = CSV.read(datadir("test.csv"), DataFrame)
id = d[!,:id]

predictions = map(x -> predict(model, x), X)
predictions = vcat(predictions...)

res = DataFrame(:id => id, :pressure => predictions)
wsave(datadir("submissions", "rnn_predictions.csv"), res)