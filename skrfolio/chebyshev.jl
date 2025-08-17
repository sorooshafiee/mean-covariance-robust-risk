using JuMP
using MosekTools
using LinearAlgebra
using Dualization


function solve_chebyshev_dro(
    mu::AbstractVector{Float64}, 
    Sigma::AbstractMatrix{Float64}, 
    p::Int, 
    solver_params
)
    
    n_features = length(mu)
    n_assets = n_features - 1
    mu = Vector(mu)
    M = Matrix(Sigma) + mu * mu'

    model = Model(dual_optimizer(Mosek.Optimizer))
    if get(solver_params, "silent", true)
        set_silent(model)
    end

    # --- Pre-calculate the values ---
    I_n = LinearAlgebra.I(n_features)
    
    # --- Define common variables ---
    w = @variable(model, w[1:n_features])
    
    
    # --- Common constraints ---
    @constraint(model, sum(w[i] for i in 1:n_assets) == 1)
    @constraint(model, w[1:n_assets] .>= 0)
    @constraint(model, w[n_features] == -1)

    if p == 2
        t = @variable(model, t >= 0)

        @constraint(model, [M  w; w'  t] in PSDCone())
        @objective(model, Min, t)
    elseif p == 1
        y0 = @variable(model, y0 >= 0)
        y = @variable(model, y[1:n_features])
        Y = @variable(model, Y[1:n_features, 1:n_features], PSD)

        @constraint(model, [Y  y; y'  y0] in PSDCone())
        @constraint(model, [Y  (y - w/2); (y - w/2)'  y0] in PSDCone())
        @constraint(model, [Y  (y + w/2); (y + w/2)'  y0] in PSDCone())

        @objective(model, Min, y0 + 2 * mu' * y + vec(M)' * vec(Y))
    else
        error("p must be 1 or 2.")
    end

    # Solve the model
    optimize!(model)
    
    return value.(w), objective_value(model), string(termination_status(model))
end