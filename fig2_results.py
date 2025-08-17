import sys
import os
import time
import numpy as np
import pandas as pd
from scipy.io import loadmat
from skrfolio.trackers import GelbrichOptimizer, DelageOptimizer
from skrfolio.trackers import WassersteinOptimizer, ChiSquaredOptimizer


def __main__():
    DATA_PATH = "datasets/"
    DATASETS = ["DowJones", "FF49Industries", "NASDAQ100", "FTSE100"]
    all_p = [1]
    T = 52
    tau = 12
    scale = 100
    all_rho = np.arange(0, 51, 2)
    optimizers = [
        WassersteinOptimizer(),
        ChiSquaredOptimizer(),
        GelbrichOptimizer(estimator_type="HuberM"),
        GelbrichOptimizer(estimator_type="LedoitWolf"),
        GelbrichOptimizer(estimator_type="MCD"),
        GelbrichOptimizer(estimator_type="TukeyS"),
        DelageOptimizer(estimator_type="HuberM"),
        DelageOptimizer(estimator_type="LedoitWolf"),
        DelageOptimizer(estimator_type="MCD"),
        DelageOptimizer(estimator_type="TukeyS"),
    ]

    for dataset_name in DATASETS:
        print(f"\n--- Running test for Dataset: {dataset_name} ---")
        mat_file_path = os.path.join(DATA_PATH, f"{dataset_name}.mat")
        mat_data = loadmat(mat_file_path)
        asset_returns = mat_data["Assets_Returns"]
        index_returns = mat_data["Index_Returns"]
        X = np.hstack([asset_returns, index_returns]) * scale
        n_samples, n_features = X.shape
        n_assets = n_features - 1

        for optimizer in optimizers:
            for p in all_p:
                for rho in all_rho:
                    optimizer.p = p
                    optimizer.rho = rho
                    print(
                        f"\t Optimizer: {type(optimizer).__name__}, Estimator: {getattr(optimizer, 'estimator_type', 'N/A')}, p: {p}, rho: {rho}"
                    )
                    results_file = f"csv/{dataset_name}_{type(optimizer).__name__}_"
                    results_file += f"{getattr(optimizer, 'estimator_type', 'NA')}_"
                    results_file += f"{int(p)}_{int(rho)}_results.csv"

                    if os.path.exists(results_file):
                        continue

                    try:
                        # --- Rolling Window Simulation ---
                        portfolio_returns_list = []
                        tracking_errors_list = []
                        start_idx = 0

                        while (start_idx + T + tau) <= n_samples:
                            train_end_idx = start_idx + T
                            X_train = X[start_idx:train_end_idx, :]

                            start_time = time.time()
                            optimizer.fit(X_train)
                            fit_time = time.time() - start_time
                            # print(f"\t\tOptimizer fit time: {fit_time:.4f} seconds")

                            asset_weights = optimizer.weights_[:n_assets]

                            test_start_idx = train_end_idx
                            test_end_idx = train_end_idx + tau

                            test_asset_returns = asset_returns[
                                test_start_idx:test_end_idx, :
                            ]
                            test_index_returns = index_returns[
                                test_start_idx:test_end_idx, 0
                            ]

                            batch_p_ret = scale * (test_asset_returns @ asset_weights)
                            portfolio_returns_list.extend(batch_p_ret)

                            if p == 1:
                                batch_p_diff = np.abs(
                                    batch_p_ret - (scale * test_index_returns)
                                )
                            elif p == 2:
                                batch_p_diff = (
                                    batch_p_ret - (scale * test_index_returns)
                                ) ** 2
                            else:
                                batch_p_diff = np.full(tau, np.nan)
                            tracking_errors_list.extend(batch_p_diff)

                            start_idx += tau

                        portfolio_returns = np.array(portfolio_returns_list)
                        tracking_errors = np.array(tracking_errors_list)

                        mean_error = np.mean(tracking_errors)
                        # print(f"\t\tMean Tracking Error: {mean_error:.6f}")

                        results = np.array([mean_error, np.std(tracking_errors)])
                        np.savetxt(results_file, results, delimiter=",")
                    except Exception as e:
                        print(f"\t\tError occurred: {e}")


if __name__ == "__main__":
    __main__()
