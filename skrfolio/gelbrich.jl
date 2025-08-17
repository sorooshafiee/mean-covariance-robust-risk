using JuMP
using MosekTools
using LinearAlgebra
using Dualization


function solve_gelbrich_dro_sqrt(
    mu::AbstractVector{Float64}, 
    Sigma::AbstractMatrix{Float64}, 
    rho::Real, 
    p::Int, 
    solver_params
)
    
    n_features = length(mu)
    n_assets = n_features - 1

    model = Model(dual_optimizer(Mosek.Optimizer))
    if get(solver_params, "silent", true)
        set_silent(model)
    end

    # --- Pre-calculate the values ---
    Sigma_half = sqrt(Sigma)
    obj_scalar_part = rho^2 - norm(mu)^2 - tr(Sigma)
    I_n = LinearAlgebra.I(n_features)
    
    # --- Define common variables ---
    w = @variable(model, w[1:n_features])
    gamma = @variable(model, gamma >= 0)
    y0 = @variable(model, y0 >= 0)
    y = @variable(model, y[1:n_features])
    z = @variable(model, z >= 0)
    Y = @variable(model, Y[1:n_features, 1:n_features], PSD)
    Z = @variable(model, Z[1:n_features, 1:n_features], PSD)
    
    # --- Common constraints ---
    @constraint(model, sum(w[i] for i in 1:n_assets) == 1)
    @constraint(model, w[1:n_assets] .>= 0)
    @constraint(model, w[n_features] == -1)
    @constraint(model, [Y  y; y'  y0] in PSDCone())
    @constraint(model, [(gamma * I_n - Y)  (y + gamma * mu); (y + gamma * mu)'  z] in PSDCone())
    @constraint(model, [Z  (gamma * Sigma_half); (gamma * Sigma_half)'  (gamma * I_n - Y)] in PSDCone())

    # --- Objective function ---
    @objective(model, Min, y0 + obj_scalar_part * gamma + z + tr(Z))

    if p == 2
        M = @variable(model, M[1:n_features, 1:n_features], PSD)
        @constraint(model, [M  y; y'  y0] in PSDCone())
        @constraint(model, [(Y - M)  w; w'  1] in PSDCone())
    elseif p == 1
        @constraint(model, [Y  (y - w/2); (y - w/2)'  y0] in PSDCone())
        @constraint(model, [Y  (y + w/2); (y + w/2)'  y0] in PSDCone())
    else
        error("p must be 1 or 2.")
    end

    # Solve the model
    optimize!(model)
    
    return value.(w), objective_value(model), string(termination_status(model))
end


function solve_gelbrich_dro_chol(
    mu::AbstractVector{Float64}, 
    Sigma::AbstractMatrix{Float64}, 
    rho::Real, 
    p::Int, 
    solver_params
)
    
    n_features = length(mu)
    n_assets = n_features - 1

    model = Model(dual_optimizer(Mosek.Optimizer))
    if get(solver_params, "silent", true)
        set_silent(model)
    end

    # --- Pre-calculate the values ---
    Sigma = Matrix(Sigma)
    F = eigen(Symmetric(Sigma))
    eigenvalues = F.values
    eigenvectors = F.vectors
    lambda_ = eigenvalues[eigenvalues .> 1e-6]
    V_ = eigenvectors[:, eigenvalues .> 1e-6]
    n_eig = length(lambda_)
    L = V_ * diagm(sqrt.(lambda_))
    obj_scalar_part = rho^2 - norm(mu)^2 - tr(Sigma)
    I_n = LinearAlgebra.I(n_features)
    
    # --- Define common variables ---
    w = @variable(model, w[1:n_features])
    gamma = @variable(model, gamma >= 0)
    y0 = @variable(model, y0 >= 0)
    y = @variable(model, y[1:n_features])
    z = @variable(model, z >= 0)
    Y = @variable(model, Y[1:n_features, 1:n_features], PSD)
    Z = @variable(model, Z[1:n_eig, 1:n_eig], PSD)
    
    # --- Common constraints ---
    @constraint(model, sum(w[i] for i in 1:n_assets) == 1)
    @constraint(model, w[1:n_assets] .>= 0)
    @constraint(model, w[n_features] == -1)
    @constraint(model, [Y  y; y'  y0] in PSDCone())
    @constraint(model, [(gamma * I_n - Y)  (y + gamma * mu); (y + gamma * mu)'  z] in PSDCone())
    @constraint(model, [Z  (gamma * L)'; (gamma * L)  (gamma * I_n - Y)] in PSDCone())

    # --- Objective function ---
    @objective(model, Min, y0 + obj_scalar_part * gamma + z + tr(Z))

    if p == 2
        M = @variable(model, M[1:n_features, 1:n_features], PSD)
        @constraint(model, [M  y; y'  y0] in PSDCone())
        @constraint(model, [(Y - M)  w; w'  1] in PSDCone())
    elseif p == 1
        @constraint(model, [Y  (y - w/2); (y - w/2)'  y0] in PSDCone())
        @constraint(model, [Y  (y + w/2); (y + w/2)'  y0] in PSDCone())
    else
        error("p must be 1 or 2.")
    end

    # Solve the model
    optimize!(model)
    
    return value.(w), objective_value(model), string(termination_status(model))
end


function solve_gelbrich_dro_eye(
    mu::AbstractVector{Float64}, 
    Sigma::AbstractMatrix{Float64}, 
    rho::Real, 
    p::Int, 
    solver_params
)
    
    n_features = length(mu)
    n_assets = n_features - 1

    model = Model(dual_optimizer(Mosek.Optimizer))
    if get(solver_params, "silent", true)
        set_silent(model)
    end

    # --- Pre-calculate the values ---
    obj_scalar_part = rho^2 - norm(mu)^2 - tr(Sigma)
    I_n = LinearAlgebra.I(n_features)
    
    # --- Define common variables ---
    w = @variable(model, w[1:n_features])
    gamma = @variable(model, gamma >= 0)
    y0 = @variable(model, y0 >= 0)
    y = @variable(model, y[1:n_features])
    z = @variable(model, z >= 0)
    Y = @variable(model, Y[1:n_features, 1:n_features], PSD)
    Z = @variable(model, Z[1:n_features, 1:n_features], PSD)
    
    # --- Common constraints ---
    @constraint(model, sum(w[i] for i in 1:n_assets) == 1)
    @constraint(model, w[1:n_assets] .>= 0)
    @constraint(model, w[n_features] == -1)
    @constraint(model, [Y  y; y'  y0] in PSDCone())
    @constraint(model, [(gamma * I_n - Y)  (y + gamma * mu); (y + gamma * mu)'  z] in PSDCone())
    @constraint(model, [Z  gamma * I_n; gamma * I_n  (gamma * I_n - Y)] in PSDCone())

    # --- Objective function ---
    @objective(model, Min, y0 + obj_scalar_part * gamma + z + vec(Sigma)' * vec(Z))

    if p == 2
        M = @variable(model, M[1:n_features, 1:n_features], PSD)
        @constraint(model, [M  y; y'  y0] in PSDCone())
        @constraint(model, [(Y - M)  w; w'  1] in PSDCone())
    elseif p == 1
        @constraint(model, [Y  (y - w/2); (y - w/2)'  y0] in PSDCone())
        @constraint(model, [Y  (y + w/2); (y + w/2)'  y0] in PSDCone())
    else
        error("p must be 1 or 2.")
    end

    # Solve the model
    optimize!(model)
    
    return value.(w), objective_value(model), string(termination_status(model))
end

function solve_gelbrich_dro_eig(
    mu::AbstractVector{Float64}, 
    Sigma::AbstractMatrix{Float64}, 
    rho::Real, 
    p::Int, 
    solver_params
)
    n_features = length(mu)
    n_assets = n_features - 1
    F = eigen(Symmetric(Matrix(Sigma)))
    eigenvalues = F.values
    eigenvectors = F.vectors
    mask = eigenvalues .> 1e-6
    lambda_ = eigenvalues[mask]
    V_ = eigenvectors[:, mask]
    n_eig = length(lambda_)

    model = Model(dual_optimizer(Mosek.Optimizer))
    if get(solver_params, "silent", true)
        set_silent(model)
    end

    # --- Pre-calculate the values ---
    obj_scalar_part = rho^2 - norm(mu)^2 - tr(Sigma)
    I_n = LinearAlgebra.I(n_features)
    
    # --- Define common variables ---
    w = @variable(model, w[1:n_features])
    gamma = @variable(model, gamma >= 0)
    y0 = @variable(model, y0 >= 0)
    y = @variable(model, y[1:n_features])
    z = @variable(model, z >= 0)
    Y = @variable(model, Y[1:n_features, 1:n_features], PSD)
    t = @variable(model, t[1:n_eig] .>= 0)

    # --- Common constraints ---
    @constraint(model, sum(w[i] for i in 1:n_assets) == 1)
    @constraint(model, w[1:n_assets] .>= 0)
    @constraint(model, w[n_features] == -1)
    @constraint(model, [Y  y; y'  y0] in PSDCone())
    @constraint(model, [(gamma * I_n - Y)  (y + gamma * mu); (y + gamma * mu)'  z] in PSDCone())
    @constraint(model, [i = 1:n_eig], [(t[i])  (gamma * V_[:, i])'; (gamma * V_[:, i])  (gamma * I_n - Y)] in PSDCone())

    # --- Objective function ---
    @objective(model, Min, y0 + obj_scalar_part * gamma + z + lambda_' * t)

    if p == 2
        M = @variable(model, M[1:n_features, 1:n_features], PSD)
        @constraint(model, [M  y; y'  y0] in PSDCone())
        @constraint(model, [(Y - M)  w; w'  1] in PSDCone())
    elseif p == 1
        @constraint(model, [Y  (y - w/2); (y - w/2)'  y0] in PSDCone())
        @constraint(model, [Y  (y + w/2); (y + w/2)'  y0] in PSDCone())
    else
        error("p must be 1 or 2.")
    end

    # Solve the model
    optimize!(model)
    
    return value.(w), objective_value(model), string(termination_status(model))
end