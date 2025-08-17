using JuMP
using MosekTools
using LinearAlgebra
using Dualization


function solve_delage_dro(
    mu::AbstractVector{Float64}, 
    Sigma::AbstractMatrix{Float64}, 
    rho::Real, 
    k::Real,
    p::Int, 
    solver_params
)
    
    n_features = length(mu)
    n_assets = n_features - 1
    
    model = Model(dual_optimizer(Mosek.Optimizer))
    if get(solver_params, "silent", true)
        set_silent(model)
    end

    w = @variable(model, w[1:n_features])
    s = @variable(model, s)
    p_vec = @variable(model, p_vec[1:n_features])
    P = @variable(model, P[1:n_features, 1:n_features], PSD)
    r = @variable(model, r)
    q = @variable(model, q[1:n_features])
    Q = @variable(model, Q[1:n_features, 1:n_features], PSD)
    
    @constraint(model, w[n_features] == -1)
    @constraint(model, sum(w[i] for i in 1:n_assets) == 1)
    @constraint(model, w[1:n_assets] .>= 0)
    @constraint(model, [P p_vec; p_vec' s] in PSDCone())
    @constraint(model, p_vec .== -q/2 - Q*mu)

    obj = (1 + k * rho^2) * tr(Sigma * Q) - mu' * Q * mu + r + tr(Sigma * P) - 2 * mu' * p_vec + (1 - k) * rho^2 * s
    @objective(model, Min, obj)

    if p == 2
        M = @variable(model, M[1:n_features, 1:n_features], PSD)
        @constraint(model, [Q - M  q/2; (q/2)'  r] in PSDCone())
        @constraint(model, [M w; w' 1] in PSDCone())
    elseif p == 1
        @constraint(model, [Q  (q/2 + w/2); (q/2 + w/2)'  r] in PSDCone())
        @constraint(model, [Q  (q/2 - w/2); (q/2 - w/2)'  r] in PSDCone())
    else
        error("p must be 1 or 2.")
    end

    optimize!(model)
    
    results = Dict(
        "weights" => value.(w),
        "objective" => objective_value(model),
        "status" => string(termination_status(model))
    )
    
    return results
end