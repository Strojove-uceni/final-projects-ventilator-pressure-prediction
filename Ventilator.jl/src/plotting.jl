function scatter2(X, x=1, y=2; kwargs...)
    if size(X,1) > size(X,2)
        X = X'
    end
    scatter(X[x,:],X[y,:]; kwargs...)
end
function scatter2!(X, x=1, y=2; kwargs...)
    if size(X,1) > size(X,2)
        X = X'
    end
    scatter!(X[x,:],X[y,:]; kwargs...)
end

function plot_lungs(X_val, P_val)
    p = Plots.Plot{Plots.GRBackend}[]
    for k in 1:4
        i = rand(1:7500)
        X = X_val[i]
        P = P_val[i]
        W = predict(X)

        plot(P, marker=:circle, markersize=2,label="");
        #plot!(X[2,:],label="");
        px = plot!(W, marker=:square, markersize=2, label="")
        push!(p, px)
    end
    plot(p...,layout=(2,2))
end