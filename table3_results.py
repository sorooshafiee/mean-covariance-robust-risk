import os
import time
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from skrfolio.trackers import GelbrichOptimizer, DelageOptimizer
from skrfolio.trackers import WassersteinOptimizer, ChiSquaredOptimizer


def __main__():
    DATA_PATH = "datasets/"
    DATASETS = ["DowJones", "FF49Industries", "FTSE100"]
    p_val = 1
    T = 52
    tau = 12
    scale = 100

    PARAM_GRID = {"rho": np.arange(0, 21, 2)}

    OPTIMIZERS_CONFIG = {
        "Gelbrich (LedoitWolf)": (GelbrichOptimizer, {"estimator_type": "LedoitWolf"}),
        "Delage (LedoitWolf)": (DelageOptimizer, {"estimator_type": "LedoitWolf"}),
        "Gelbrich (HuberM)": (GelbrichOptimizer, {"estimator_type": "HuberM"}),
        "Delage (HuberM)": (DelageOptimizer, {"estimator_type": "HuberM"}),
        "Gelbrich (TukeyS)": (GelbrichOptimizer, {"estimator_type": "TukeyS"}),
        "Delage (TukeyS)": (DelageOptimizer, {"estimator_type": "TukeyS"}),
        "Gelbrich (MCD)": (GelbrichOptimizer, {"estimator_type": "MCD"}),
        "Delage (MCD)": (DelageOptimizer, {"estimator_type": "MCD"}),
        "Wasserstein": (WassersteinOptimizer, {}),
        "ChiSquared": (ChiSquaredOptimizer, {}),
    }

    final_results = []

    for dataset_name in DATASETS:
        print(f"\n{'='*20} Processing Dataset: {dataset_name} {'='*20}")

        # Load and prepare data
        mat_file_path = os.path.join(DATA_PATH, f"{dataset_name}.mat")
        mat_data = loadmat(mat_file_path)
        asset_returns = mat_data["Assets_Returns"]
        index_returns = mat_data["Index_Returns"]
        X = np.hstack([asset_returns, index_returns]) * scale
        n_samples = X.shape[0]
        n_assets = X.shape[1] - 1

        for config_name, (
            OptimizerClass,
            extra_params,
        ) in OPTIMIZERS_CONFIG.items():

            print(f"\n--- Running: {config_name} ---")

            full_run_tracking_errors = []
            selected_rhos = []
            start_idx = 0

            while (start_idx + T + tau) <= n_samples:
                train_end_idx = start_idx + T
                X_train = X[start_idx:train_end_idx, :]

                base_optimizer = OptimizerClass(p=p_val, **extra_params)
                tscv = TimeSeriesSplit(n_splits=5)

                grid_search = GridSearchCV(
                    estimator=base_optimizer, param_grid=PARAM_GRID, cv=tscv
                )

                grid_search.fit(X_train)

                best_rho = grid_search.best_params_["rho"]
                best_optimizer = grid_search.best_estimator_
                selected_rhos.append(best_rho)

                # --- 5. Forward Test with the Best Model ---
                asset_weights = best_optimizer.weights_[:n_assets]

                test_start_idx = train_end_idx
                test_end_idx = train_end_idx + tau

                test_asset_returns = asset_returns[test_start_idx:test_end_idx, :]
                test_index_returns = index_returns[test_start_idx:test_end_idx, 0]
                batch_p_ret = scale * (test_asset_returns @ asset_weights)

                if p_val == 1:
                    batch_p_diff = np.abs(batch_p_ret - (scale * test_index_returns))
                else:
                    batch_p_diff = (batch_p_ret - (scale * test_index_returns)) ** 2

                full_run_tracking_errors.extend(batch_p_diff)
                start_idx += tau

            mean_tracking_error = np.mean(full_run_tracking_errors)

            print(
                f"--> Final Mean Tracking Error for {config_name}: {mean_tracking_error:.4f}"
            )

            final_results.append(
                {
                    "Dataset": dataset_name,
                    "Optimizer": config_name,
                    "p": p_val,
                    "Mean Tracking Error": mean_tracking_error,
                    "Avg Rho Selected": np.mean(selected_rhos),
                }
            )
    results_df = pd.DataFrame(final_results)
    summary_table = results_df.pivot_table(
        index="Optimizer",
        columns="Dataset",
        values="Mean Tracking Error",
    )

    print("\n\n" + "=" * 80)
    print(f" " * 15 + f"FINAL SUMMARY: MEAN TRACKING ERROR")
    print("=" * 80)
    print(summary_table.to_string(float_format="%.4f"))
    print("=" * 80)

    # Save the results
    results_df.to_csv(f"csv/adaptive_backtest_full_results.csv", index=False)
    summary_table.to_csv(f"csv/adaptive_backtest_summary_table.csv")
    print(f"\nFull results and summary table saved to 'csv/' directory.")


if __name__ == "__main__":
    __main__()
