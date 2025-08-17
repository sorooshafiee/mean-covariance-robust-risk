import numpy as np
import pandas as pd
from matplotlib import rc
import matplotlib.pyplot as plt

# Your rc settings remain the same
rc("font", family="serif")
rc("text", usetex=True)
rc("xtick", labelsize=14)
rc("ytick", labelsize=14)
rc("axes", labelsize=18, titlesize=18)
rc("legend", fontsize=16)
rc("figure", figsize=(8, 6))

# Your dictionaries remain the same
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
    "cvar_loss": "Conditional Value-at-Risk",
    "markowitz_return": "Markowitz Return",
    "mean": "Average Return",
    "mean_log_return": "Average Log-Return",
}


def __main__():
    # Legend Plot
    legend_fig, legend_ax = plt.subplots(figsize=(10, 2))
    legend_ax.axis("off")
    method_keys = list(methods_to_markers.keys())

    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            marker=methods_to_markers[method],
            color="white",
            markerfacecolor="white",
            markeredgecolor=methods_to_colors[method],
            markersize=18,
            linestyle="-",
            markeredgewidth=2,
            linewidth=2,
        )
        for method in method_keys
    ]

    legend_labels = [methods_to_labels[method] for method in method_keys]

    legend = legend_ax.legend(
        handles=legend_handles,
        labels=legend_labels,
        loc="center",
        ncol=10,
        fontsize=20,
        frameon=True,
        edgecolor="black",
        facecolor="white",
        handletextpad=0.01,
        columnspacing=0.01,
    )

    legend_filename = "pdf/legend.pdf"
    legend_fig.savefig(legend_filename, dpi=300, format="pdf", bbox_inches="tight")


if __name__ == "__main__":
    __main__()
