import os
import numpy as np
import pandas as pd
from matplotlib import rc
import matplotlib.pyplot as plt

rc("font", family="serif")
rc("text", usetex=True)
rc("xtick", labelsize=18)
rc("ytick", labelsize=18)
rc("axes", labelsize=18, titlesize=24)
rc("legend", fontsize=20)
rc("figure", figsize=(8, 6))

methods_to_labels = {
    "GelbrichOptimizer (LedoitWolf)": r"\texttt{Gelbrich (Ledoit-Wolf)}",
    "GelbrichOptimizer (HuberM)": r"\texttt{Gelbrich (Huber-M)}",
    "GelbrichOptimizer (TukeyS)": r"\texttt{Gelbrich (Tukey-S)}",
    "GelbrichOptimizer (MCD)": r"\texttt{Gelbrich (MCD)}",
}


def __main__():
    DATA_PATH = "csv/"
    DATASETS = ["DowJones", "FF49Industries", "FTSE100"]
    ALL_P = [1]
    ALL_RHO = np.arange(0, 51, 2)

    for dataset_name in DATASETS:

        for p_val in ALL_P:
            print(f"Generating statistics for: {dataset_name}, p={p_val}")

            for method_key, styles in methods_to_labels.items():
                optimizer_name, estimator_type = method_key.split(" (")
                estimator_type = estimator_type.strip(")")

                mean_errors = []
                valid_rhos = []

                for rho in ALL_RHO:
                    estimator_filename = (
                        estimator_type if estimator_type != "N/A" else "NA"
                    )
                    filename = f"{dataset_name}_{optimizer_name}_{estimator_filename}_{p_val}_{rho}_results.csv"
                    filepath = os.path.join(DATA_PATH, filename)

                    if os.path.exists(filepath):
                        try:
                            # Load the result (mean_error is the first value)
                            result = np.loadtxt(filepath, delimiter=",")
                            if result.size > 0:
                                mean_errors.append(result[0])
                                valid_rhos.append(rho)
                        except Exception as e:
                            print(
                                f"Warning: Could not read file {filepath}. Error: {e}"
                            )

                if not valid_rhos:
                    continue

                min_error = np.min(mean_errors)
                zero_error = mean_errors[0]
                print(
                    f"Method: {method_key}, Error at zero: {zero_error:.4f}, Min Error: {min_error:.4f}"
                )

    print("All statistics have been reported successfully.")


if __name__ == "__main__":
    __main__()
