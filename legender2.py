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
    "GelbrichOptimizer (LedoitWolf)": "o",
    "GelbrichOptimizer (HuberM)": "v",
    "DelageOptimizer (LedoitWolf)": "o",
    "DelageOptimizer (HuberM)": "v",
    "WassersteinOptimizer": "s",
    "ChiSquaredOptimizer": "*",
}
methods_to_labels = {
    "GelbrichOptimizer (LedoitWolf)": r"\texttt{Gelbrich (Ledoit-Wolf)}",
    "GelbrichOptimizer (HuberM)": r"\texttt{Gelbrich (Huber-M)}",
    "DelageOptimizer (LedoitWolf)": r"\texttt{Delage-Ye (Ledoit-Wolf)}",
    "DelageOptimizer (HuberM)": r"\texttt{Delage-Ye (Huber-M)}",
    "WassersteinOptimizer": r"\texttt{Wasserstein}",
    "ChiSquaredOptimizer": r"\texttt{Chai-Squared}",
}
methods_to_colors = {
    "GelbrichOptimizer (LedoitWolf)": "#2B5491",  # blue
    "GelbrichOptimizer (HuberM)": "#BA1B1D",  # red
    "DelageOptimizer (LedoitWolf)": "#2B5491",  # blue
    "DelageOptimizer (HuberM)": "#BA1B1D",  # red
    "WassersteinOptimizer": "#6631a3",  # purple
    "ChiSquaredOptimizer": "#774237",  # brown
}

methods_to_linestyle = {
    "GelbrichOptimizer (LedoitWolf)": "-",
    "GelbrichOptimizer (HuberM)": "-",
    "DelageOptimizer (LedoitWolf)": "--",
    "DelageOptimizer (HuberM)": "--",
    "WassersteinOptimizer": "-",
    "ChiSquaredOptimizer": "-",
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
            color=methods_to_colors[method],
            markerfacecolor="white",
            markeredgecolor=methods_to_colors[method],
            markersize=18,
            linestyle=methods_to_linestyle[method],
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
        handletextpad=0.2,
        columnspacing=0.5,
    )

    legend_filename = "pdf/legend2.pdf"
    legend_fig.savefig(legend_filename, dpi=300, format="pdf", bbox_inches="tight")


if __name__ == "__main__":
    __main__()
