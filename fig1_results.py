import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

import numpy as np
import pandas as pd
from skrfolio.portfolios import MarkowitzOptimizer
from skrfolio.portfolios import WassersteinOptimizer, ChiSquaredOptimizer


def normal_returns(
    n_assets,
    n_samples,
    n_factors=0,
    mu_step=0.01,
    sigma_0=0.01,
    sigma_step=0.01,
):
    target_mus_r = np.array([i * mu_step for i in range(1, n_assets + 1)])
    idiosyncratic_vars = np.array(
        [(i * sigma_step) ** 2 for i in range(1, n_assets + 1)]
    )
    D = np.diag(idiosyncratic_vars)
    if n_factors > 0:
        L = np.random.randn(n_assets, n_factors) * sigma_0 / np.sqrt(n_factors)
        cov_matrix = L @ L.T + D
    else:
        cov_matrix = sigma_0 * np.eye(n_assets) + D
    returns = np.random.multivariate_normal(target_mus_r, cov_matrix, n_samples)
    return np.maximum(returns, -0.9999)


def calculate_cvar(losses, beta):

    if not isinstance(beta, (int, float)):
        raise TypeError("beta must be a number (int or float).")
    if not (0.0 <= beta <= 1.0):
        raise ValueError("beta must be between 0.0 and 1.0 inclusive.")

    if beta == 0.0:
        return np.max(losses)

    if beta == 1.0:
        return np.mean(losses)

    n_samples = len(losses)
    k = int(np.ceil(n_samples * beta))
    ind = np.argpartition(losses, -k)[-k:]
    tail_losses = losses[ind]

    return np.mean(tail_losses)


def __main__():
    # --- Simulation Parameters ---
    n_train = 20
    n_assets = 20
    n_test = int(1e6)
    n_counts = int(1e2)
    n_factors = 0
    mu_step = 0.01
    sigma_0 = 0.04
    sigma_step = 0.02
    beta = 0.8
    beta_bad_tail = 1 - beta
    alpha = np.sqrt((1 - beta) / beta)

    # Updated base_range for a more detailed spectrum
    base_range = np.arange(1, 10)
    base_scale = [-3, -2, -1]
    all_rhos = [base_range * (10**p) for p in base_scale]
    all_rhos.append(np.array([10 ** (base_scale[-1] + 1)]))
    all_rhos = np.round(np.concatenate(all_rhos), -base_scale[0])

    # --- Optimizers ---
    mp = {"support_fraction": 0.9, "random_state": 42}
    hp = {"q": 0.9}
    tp = {"c": 30}
    optimizers = [
        MarkowitzOptimizer(estimator_type="HuberM", alpha=alpha, estimator_params=hp),
        MarkowitzOptimizer(estimator_type="LedoitWolf", alpha=alpha),
        MarkowitzOptimizer(estimator_type="MCD", alpha=alpha, estimator_params=mp),
        MarkowitzOptimizer(estimator_type="TukeyS", alpha=alpha, estimator_params=tp),
        WassersteinOptimizer(beta=beta),
        ChiSquaredOptimizer(beta=beta),
    ]

    # --- Data Loading/Simulation ---
    file_suffix = f"_{n_train}"
    results_file = f"csv/results{file_suffix}.csv"

    print("Running simulations...")
    results = []
    for i in range(n_counts):
        print(f"Iteration {i + 1}/{n_counts}")
        all_returns = normal_returns(
            n_assets, n_train + n_test, n_factors, mu_step, sigma_0, sigma_step
        )
        train_returns, test_returns = all_returns[:n_train], all_returns[n_train:]
        for rho in all_rhos:
            for optimizer in optimizers:
                optimizer.rho = rho
                optimizer.fit(train_returns)
                weights = optimizer.weights_
                test_returns_weighted = test_returns @ weights
                results.append(
                    {
                        "iteration": i,
                        "optimizer": type(optimizer).__name__,
                        "estimator_type": getattr(optimizer, "estimator_type", "()"),
                        "rho": rho,
                        "mean": np.mean(test_returns_weighted),
                        "std": np.std(test_returns_weighted),
                        "markowitz_return": np.mean(test_returns_weighted)
                        - alpha * np.std(test_returns_weighted),
                        "cvar_loss": calculate_cvar(-test_returns_weighted, beta),
                    }
                )
    df = pd.DataFrame(results)
    df.to_csv(results_file, index=False)
    print(f"Results saved to {results_file}")


if __name__ == "__main__":
    __main__()
