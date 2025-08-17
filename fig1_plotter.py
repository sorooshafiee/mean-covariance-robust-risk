import numpy as np
import pandas as pd
from matplotlib import rc
import matplotlib.pyplot as plt

# Your rc settings remain the same
rc("font", family="serif")
rc("text", usetex=True)
rc("xtick", labelsize=18)
rc("ytick", labelsize=18)
rc("axes", labelsize=18, titlesize=24)
rc("legend", fontsize=16)
rc("figure", figsize=(8, 6))

# --- Dictionaries defining styles, labels and colors ---
methods_to_markers = {
    "MarkowitzOptimizer (LedoitWolf)": "o",
    "MarkowitzOptimizer (HuberM)": "v",
    "MarkowitzOptimizer (TukeyS)": "^",
    "MarkowitzOptimizer (MCD)": "d",
    "WassersteinOptimizer": "s",
    "ChiSquaredOptimizer": "*",
}
methods_to_labels = {
    "MarkowitzOptimizer (LedoitWolf)": r"\texttt{Markowitz (Ledoit-Wolf)}",
    "MarkowitzOptimizer (HuberM)": r"\texttt{Markowitz (Huber-M)}",
    "MarkowitzOptimizer (TukeyS)": r"\texttt{Markowitz (Tukey-S)}",
    "MarkowitzOptimizer (MCD)": r"\texttt{Markowitz (MCD)}",
    "WassersteinOptimizer": r"\texttt{Wasserstein}",
    "ChiSquaredOptimizer": r"\texttt{Chai-Squared}",
}
methods_to_colors = {
    "MarkowitzOptimizer (LedoitWolf)": "#2B5491",  # blue
    "MarkowitzOptimizer (HuberM)": "#BA1B1D",  # red
    "MarkowitzOptimizer (TukeyS)": "#E38E07",  # orange
    "MarkowitzOptimizer (MCD)": "#316B48",  # green
    "WassersteinOptimizer": "#6631a3",  # purple
    "ChiSquaredOptimizer": "#774237",  # brown
}

metrics_to_titles = {
    "cvar_loss": "Conditional Value-at-Risk Loss",
    "markowitz_return": "Mean-Standard Deviation Return",
}


def __main__():
    n_train = 20
    file_suffix = f"_{n_train}"
    results_file = f"csv/results{file_suffix}.csv"
    rhos_to_mark = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    df = pd.read_csv(results_file)

    # To store handles and labels for the final legend
    plot_handles = []
    plot_labels = []

    # --- Generate each plot by iterating through predefined methods ---
    for metric_col, title in metrics_to_titles.items():
        print(f"Generating plot for: {title}")
        fig, ax = plt.subplots()

        # Iterate through the methods defined in your dictionaries
        for method_key, marker in methods_to_markers.items():
            # Parse the method_key to filter the DataFrame
            if "(" in method_key:
                optimizer_name, estimator_type = method_key.split(" (")
                estimator_type = estimator_type.strip(")")
            else:
                optimizer_name = method_key
                estimator_type = "()"

            # Filter the DataFrame for the current method
            method_df = df[
                (df["optimizer"] == optimizer_name)
                & (df["estimator_type"] == estimator_type)
            ]

            # Skip if no data is found for this method
            if method_df.empty:
                print(f"Warning: No data found for method: {method_key}")
                continue

            mean_vals = method_df.groupby("rho")[metric_col].mean()

            # Find indices for markers
            mark_indices = [
                idx
                for idx, rho in enumerate(mean_vals.index)
                if any(np.isclose(rho, rhos_to_mark))
            ]

            # Plot the data for the current method
            (line,) = ax.plot(
                mean_vals.index,
                mean_vals.values,
                marker=marker,
                color=methods_to_colors[method_key],
                linestyle="-",
                linewidth=3,
                markevery=mark_indices,
                markerfacecolor="white",
                markeredgewidth=1.5,
                markersize=10,
            )

            # On the first plot iteration, save handles and labels for the legend
            if metric_col == list(metrics_to_titles.keys())[0]:
                plot_handles.append(line)
                plot_labels.append(methods_to_labels[method_key])

        # Configure and save the plot
        ax.set_xlabel(r"$\rho$")
        ax.set_ylabel(title)
        # ax.set_title(title)
        ax.set_xscale("log")
        ax.grid(True, which="both", ls="--", linewidth=0.5)
        if n_train == 100:
            if metric_col == "cvar_loss":
                ax.set_ylim(top=-0.095)
            if metric_col == "markowitz_return":
                ax.set_ylim(bottom=0.075)
        if n_train == 1000:
            if metric_col == "cvar_loss":
                ax.set_ylim(top=-0.105)
            if metric_col == "markowitz_return":
                ax.set_ylim(bottom=0.09)
        fig.tight_layout()
        fig.savefig(f"pdf/{metric_col}_vs_rho{file_suffix}.pdf")
        plt.close(fig)

    print("All plots have been created successfully.")


if __name__ == "__main__":
    __main__()
