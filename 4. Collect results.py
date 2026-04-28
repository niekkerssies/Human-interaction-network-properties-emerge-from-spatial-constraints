
import os
import json
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import pyabc

'''
4. COLLECTING AND VISUALIZING RESULTS.
This produces visualizations of the results, automatically storing them 
to the folder where this script is stored. 

'''

from preprocessing import (
    Alldays,
    whole_cumulative_graph,
    collect_stats_vector,
    stats_vector_whole_cumulative_graph_macro,
    stats_vector_weekday_avg_macro,
    stats_vector_whole_cumulative_graph_allstats,
    stats_vector_weekday_avg_allstats,
)

# =========================
# SETTINGS
# =========================

FIGURES_DIR = "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

RUN_NAMES = [
    "RM_MONTH",
    "RM_WEEKDAY",
    "SM_MONTH",
    "SM_WEEKDAY",
]

RUN_NAMES = [name for name in RUN_NAMES if os.path.exists(f"{name}.db")]

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 16,
    "axes.labelsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
})

observed_by_run = {
    "RM_MONTH": stats_vector_whole_cumulative_graph_macro,
    "RM_WEEKDAY": stats_vector_weekday_avg_macro,
    "SM_MONTH": stats_vector_whole_cumulative_graph_allstats,
    "SM_WEEKDAY": stats_vector_weekday_avg_allstats,
}

data_graph_by_run = {
    "RM_MONTH": whole_cumulative_graph,
    "SM_MONTH": whole_cumulative_graph,
    "RM_WEEKDAY": Alldays[1],
    "SM_WEEKDAY": Alldays[1],
}


# =========================
# HELPERS
# =========================

def savefig(name):
    path = os.path.join(FIGURES_DIR, name)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved: {path}")


def load_history(run_name):
    return pyabc.History(f"sqlite:///{run_name}.db")


def get_best_network(history):
    pop = history.get_population_extended()
    best = pop.sort_values("distance").iloc[0]

    edges_raw = best.get("sumstat_network_edges", None)

    if edges_raw is None or not isinstance(edges_raw, str):
        raise ValueError("No stored network found under sumstat_network_edges.")

    best_edges = json.loads(edges_raw)

    G = nx.Graph()
    for u, v, w in best_edges:
        G.add_edge(u, v, weight=w)

    return G, best, pop


def degree_distribution_compare(graph1, graph2, run_name, labels=("Best network", "Data network")):
    deg1 = [graph1.degree(n) for n in graph1.nodes]
    deg2 = [graph2.degree(n) for n in graph2.nodes]

    if len(deg1) == 0 or len(deg2) == 0:
        print(f"{run_name}: skipping degree comparison because one graph is empty.")
        return

    plt.figure(figsize=(6, 6))
    plt.hist(deg1, bins=60, alpha=0.5, label=labels[0], edgecolor="black")
    plt.hist(deg2, bins=60, alpha=0.5, label=labels[1], edgecolor="black")
    plt.title(f"{run_name}: Degree distribution comparison")
    plt.xlabel("Node degree")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(alpha=0.2)
    savefig(f"{run_name}_degree_distribution_compare.png")
    plt.close()


def weight_distribution_compare(graph1, graph2, run_name, labels=("Best network", "Data network")):
    w1 = [graph1.edges[e]["weight"] for e in graph1.edges]
    w2 = [graph2.edges[e]["weight"] for e in graph2.edges]

    if len(w1) == 0 or len(w2) == 0:
        print(f"{run_name}: skipping weight comparison because one graph has no edges.")
        return

    plt.figure(figsize=(6, 6))
    plt.hist(w1, bins=60, alpha=0.5, label=labels[0], edgecolor="black")
    plt.hist(w2, bins=60, alpha=0.5, label=labels[1], edgecolor="black")
    plt.xscale("log")
    plt.title(f"{run_name}: Edge weight distribution comparison")
    plt.xlabel("Interactions per link")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(alpha=0.2)
    savefig(f"{run_name}_weight_distribution_compare.png")
    plt.close()


def plot_parameter_posteriors(df, w, run_name):
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    axs[0].hist(df["r"], bins=30, weights=w, density=True, alpha=0.7)
    axs[0].set_xlabel("Interaction range r")
    axs[0].set_ylabel("Posterior density")
    axs[0].set_title("Posterior of r")
    axs[0].grid(alpha=0.2)

    axs[1].hist(df["s"], bins=30, weights=w, density=True, alpha=0.7)
    axs[1].set_xlabel("Step size s")
    axs[1].set_ylabel("Posterior density")
    axs[1].set_title("Posterior of s")
    axs[1].grid(alpha=0.2)

    plt.tight_layout()
    savefig(f"{run_name}_posterior_histograms.png")
    plt.close()


def plot_kde_2d(df, w, run_name):
    fig, ax = plt.subplots(figsize=(6, 5))

    pyabc.visualization.plot_kde_2d(
        df,
        w,
        x="r",
        y="s",
        xmin=float(df["r"].min()),
        xmax=float(df["r"].max()),
        ymin=float(df["s"].min()),
        ymax=float(df["s"].max()),
        numx=200,
        numy=200,
        xname="Interaction range r",
        yname="Step size s",
        ax=ax,
    )

    ax.set_title(f"{run_name}: Joint ABC posterior density")
    plt.tight_layout()
    savefig(f"{run_name}_posterior_kde_2d.png")
    plt.close()


def plot_summary_distributions(pop, best, observed, run_name):
    cols = [
        c for c in pop.columns
        if c.startswith("sumstat_")
        and c != "sumstat_network_edges"
        and c.replace("sumstat_", "") in observed
    ]

    if not cols:
        print(f"{run_name}: no matching summary-stat columns found.")
        return

    plt.figure(figsize=(3.2 * len(cols), 5))

    for i, col in enumerate(cols, 1):
        plt.subplot(1, len(cols), i)

        values = pop[col].astype(float)

        sns.violinplot(
            y=values,
            inner="quartile",
            linewidth=1.3,
            width=0.8,
        )

        obs_key = col.replace("sumstat_", "")
        true_val = observed[obs_key]
        best_val = float(best[col])

        plt.axhline(true_val, linestyle="--", linewidth=2)
        plt.scatter([0], [best_val], s=60, zorder=3)

        plt.title(obs_key, fontsize=11, pad=10)
        plt.ylabel("")
        plt.tick_params(axis="both", labelsize=8)
        plt.grid(axis="y", alpha=0.2)

    legend_elements = [
        Line2D([0], [0], linestyle="--", linewidth=2, label="Observed value"),
        Line2D([0], [0], marker="o", markersize=8, linestyle="None", label="Best run"),
    ]

    plt.figlegend(
        handles=legend_elements,
        loc="upper center",
        ncol=2,
        fontsize=9,
        frameon=False,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.9])
    savefig(f"{run_name}_summary_distributions.png")
    plt.close()


# =========================
# MAIN LOOP
# =========================

if not RUN_NAMES:
    print("No result databases found. Expected files like RM_MONTH.db or SM_WEEKDAY.db.")

for run_name in RUN_NAMES:
    print(f"\n=== Collecting results for {run_name} ===")

    history = load_history(run_name)

    df, w = history.get_distribution()
    print(df.describe())

    best_network, best, pop = get_best_network(history)

    observed = observed_by_run[run_name]
    data_graph = data_graph_by_run[run_name]

    plot_parameter_posteriors(df, w, run_name)
    plot_kde_2d(df, w, run_name)
    plot_summary_distributions(pop, best, observed, run_name)

    degree_distribution_compare(best_network, data_graph, run_name)
    weight_distribution_compare(best_network, data_graph, run_name)

    print("\nBest simulated network stats:")
    print(
        collect_stats_vector(
            best_network,
            links=True,
            weights=True,
            dyads=True,
            CC=True,
            ASPL=True,
            gini=True,
        )
    )

    print("\nObserved stats:")
    print(observed)

print("\nDone. Figures saved in the figures folder.")