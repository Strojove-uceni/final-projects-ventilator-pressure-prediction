using DrWatson
@quickactivate
using GaussianProcesses
using Random
using Optim

using Plots
ENV["GKSwstype"] = "100"

##########################################################################
#           GaussianProcesses demo - Linear Regression                   #
##########################################################################

# create sample data
n = 10
x = 2π * rand(n)
y = sin.(x) + 0.05*randn(n)

# initialize GP
mZero = MeanZero()
kern = SE(0.0,0.0)
logObsNoise = -1.0
gp = GP(x,y,mZero,kern,logObsNoise)

# plot unoptimized GP
p = plot(gp)
savefig("GP.png")

# optimize GP and plot it
optimize!(gp)
p = plot(gp)
savefig("GP_optimized.png")

#Training data
d, n = 2, 50;         #Dimension and number of observations
x = 2π * rand(d, n);                               #Predictors
y = vec(sin.(x[1,:]).*sin.(x[2,:])) + 0.05*rand(n);  #Responses

mZero = MeanZero()                             # Zero mean function
kern = Matern(5/2,[0.0,0.0],0.0) + SE(0.0,0.0)

gp = GP(x,y,mZero,kern,-2.0)

##########################################################################
#              Sample data from Google Brain                             #
##########################################################################

using Ventilator
data = load_data_bags(;seed=1);
X_train, P_train = data[1];

# just for single breath
idx = rand(1:60000)
x = X_train[idx] .|> Float64
p = reshape(P_train[idx] |> transpose |> collect .|> Float64,80)

mZero = MeanZero() 
kern = Matern(5/2,[0.0,0.0,0.0,0.0,0.0],0.0) + SE(0.0,0.0)

gp = GP(x,p,mZero,kern,-1.0)

optimize!(gp)
p_pred = predict_f(gp, x)
plot(x[1,:], p, marker=:circle)
plot!(x[1,:], p_pred, marker=:square)
savefig("testGP.png")

# for more breaths
n = 20
idx = rand(1:60000,n)
x = hcat(X_train[idx]...) .|> Float64
p = reshape(hcat(P_train[idx]...) |> transpose |> collect .|> Float64,80*n)

mZero = MeanZero() 
kern = Matern(5/2,[0.0,0.0,0.0,0.0,0.0],0.0) + SE(0.0,0.0)

gp = GP(x,p,mZero,kern,-1.0)

optimize!(gp)

pl = []
for i in idx[1:6]
    x = X_train[i] .|> Float64
    p = reshape(P_train[i] |> transpose |> collect .|> Float64,80)
    p_pred = predict_f(gp, x)
    plot(x[1,:], p, marker=:circle)
    plt = plot!(x[1,:], p_pred, marker=:square, opacity=0.4)
    push!(pl, plt)
end
plot(pl..., layout=(3,2), size=(1000,800))
savefig("testGP_train.png")

idx_new = rand(1:60000,n)
pl = []
for i in idx_new[1:6]
    x = X_train[i] .|> Float64
    p = reshape(P_train[i] |> transpose |> collect .|> Float64,80)
    p_pred = predict_f(gp, x)
    plot(x[1,:], p, marker=:circle)
    plt = plot!(x[1,:], p_pred, marker=:square, opacity=0.4)
    push!(pl, plt)
end
plot(pl..., layout=(3,2), size=(1000,800))
savefig("testGP_test.png")