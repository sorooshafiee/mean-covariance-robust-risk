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
rc("figure", figsize=(8, 6))


methods_to_labels = {
    "GelbrichOptimizer (LedoitWolf)": r"\texttt{Gelbrich (Ledoit-Wolf)}",
    "DelageOptimizer (LedoitWolf)": r"\texttt{Delage (Ledoit-Wolf)}",
    "GelbrichOptimizer (HuberM)": r"\texttt{Gelbrich (Huber-M)}",
    "DelageOptimizer (HuberM)": r"\texttt{Delage (Huber-M)}",
    # "GelbrichOptimizer (TukeyS)": r"\texttt{Gelbrich (Tukey-S)}",
    # "DelageOptimizer (TukeyS)": r"\texttt{Delage (Tukey-S)}",
    # "GelbrichOptimizer (MCD)": r"\texttt{Gelbrich (MCD)}",
    # "DelageOptimizer (MCD)": r"\texttt{Delage (MCD)}",
    "WassersteinOptimizer (N/A)": r"\texttt{Wasserstein}",
    "ChiSquaredOptimizer (N/A)": r"\texttt{Chi-Squared}",
}


def __main__():
    DATA_PATH = "csv/"
    DATASETS = ["DowJones", "FF49Industries", "FTSE100"]
    ALL_P = [1]
    ALL_RHO = np.arange(0, 51, 2)

    # Define which rho values should have a marker on the plot
    rhos_to_mark_map = {
        "DowJones": [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
        "FF49Industries": [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
        "FTSE100": [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
    }

    # Store handles and labels for the final, standalone legend
    plot_handles = []

    for dataset_name in DATASETS:
        rhos_to_mark = rhos_to_mark_map.get(dataset_name, [0, 10, 20, 30])

        for p_val in ALL_P:
            print(f"Generating plot for: {dataset_name}, p={p_val}")
            fig, ax = plt.subplots()

            if dataset_name == "FF49Industries":
                methods_to_styles = {
                    "GelbrichOptimizer (LedoitWolf)": {
                        "color": "#2B5491",
                        "linestyle": "-",
                        "marker": "o",
                    },
                    "DelageOptimizer (LedoitWolf)": {
                        "color": "#2B5491",
                        "linestyle": "--",
                        "marker": "o",
                    },
                    "WassersteinOptimizer (N/A)": {
                        "color": "#6631a3",
                        "linestyle": "-",
                        "marker": "s",
                    },
                    "ChiSquaredOptimizer (N/A)": {
                        "color": "#774237",
                        "linestyle": "-",
                        "marker": "*",
                    },
                }
            else:
                methods_to_styles = {
                    "GelbrichOptimizer (HuberM)": {
                        "color": "#BA1B1D",
                        "linestyle": "-",
                        "marker": "v",
                    },
                    "DelageOptimizer (HuberM)": {
                        "color": "#BA1B1D",
                        "linestyle": "--",
                        "marker": "v",
                    },
                    "WassersteinOptimizer (N/A)": {
                        "color": "#6631a3",
                        "linestyle": "-",
                        "marker": "s",
                    },
                    "ChiSquaredOptimizer (N/A)": {
                        "color": "#774237",
                        "linestyle": "-",
                        "marker": "*",
                    },
                }

            for method_key, styles in methods_to_styles.items():
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
                mark_indices = [
                    idx for idx, rho in enumerate(valid_rhos) if rho in rhos_to_mark
                ]

                (line,) = ax.plot(
                    valid_rhos,
                    mean_errors,
                    marker=styles["marker"],
                    color=styles["color"],
                    linestyle=styles["linestyle"],
                    linewidth=3,
                    markevery=mark_indices,
                    markerfacecolor="white",  # Hollow markers
                    markeredgewidth=1.5,
                    markersize=10,
                )

                # Store the handle and label for the legend ONCE
                if dataset_name == DATASETS[0] and p_val == ALL_P[0]:
                    plot_handles.append((line, methods_to_labels[method_key]))

            # --- Configure and Save the Plot (NO LEGEND) ---
            ax.set_xlabel(r"$\rho$")
            ax.set_ylabel("Mean Tracking Error")
            ax.set_title(f"{dataset_name.replace('_', ' ')}")
            ax.grid(True, which="both", ls="--", linewidth=0.5)
            if p_val == 2:
                ax.set_ylim(top=0.5)
            fig.tight_layout()

            output_filename = f"pdf/fig2_{dataset_name}_p{p_val}.pdf"
            fig.savefig(output_filename)
            plt.close(fig)

    # --- Create a Standalone Legend (Identical to fig1_plotter) ---
    if plot_handles:
        # Sort handles by the label text to group them logically
        plot_handles.sort(key=lambda x: x[1])
        handles, labels = zip(*plot_handles)

        # Create a new figure just for the legend
        legend_fig = plt.figure(figsize=(12, 4))
        legend_fig.legend(handles, labels, loc="center", ncol=5, frameon=False)
        legend_fig.tight_layout()
        legend_fig.savefig("pdf/fig2_legend.pdf")
        plt.close(legend_fig)
        print("\nStandalone legend file 'fig2_legend.pdf' has been created.")

    print("All plots have been created successfully.")


if __name__ == "__main__":
    __main__()
