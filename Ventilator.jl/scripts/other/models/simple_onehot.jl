using Flux
using Base.Iterators: repeated
using Flux: throttle, @epochs
using StatsBase
using Ventilator

data = load_data_vectors_onehot(;seed=1);
X_train, P_train = data[1];
X_val, P_val = data[2];
X_test, P_test = data[3];
x, p = X_train[1], P_train[1]
B_train, B_val, B_test = data[4];

idim = length(x)
hdim = 32
activation = "relu"

loss(x, y) = Flux.mae(model(x), y)
score(x, y, b) = Flux.mae(model(x)[b .== 0], y[b .== 0])

model = Ventilator.build_mlp(idim, hdim, 80, 3, activation="relu", lastlayer="linear")
opt = ADAM()

best_val_score = Inf
best_model = deepcopy(model)

for i in 1:10000
    batch = RandomBatch(X_train, P_train, batchsize=64)
    Flux.train!(loss, Flux.params(model), zip(batch...), opt)
    sc = mean(score.(X_val, P_val, B_val))
    if sc < best_val_score
        println("Score: $(round(sc, digits=3))")
        best_val_score = sc
        best_model = deepcopy(model)
    end
end