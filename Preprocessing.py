from pathlib import Path
import networkx as nx
import numpy as np
import pandas as pd

'''
1. DATA PROCESSING
This code retrieves, preprocesses, and constructs networks from the data. It assumes the bt_symmetric.csv file to be in the same folder as the python file. 
This is the bluetooth interaction data from the publically available month of Copenhagen Social Network Data, 
available for download here: https://figshare.com/articles/dataset/The_Copenhagen_Networks_Study_interaction_data/7267433
'''

def gini_from_values(values):
    """
    Gini coefficient of a one-dimensional collection of non-negative values.
    """
    x = np.asarray(values, dtype=float)

    if len(x) == 0 or np.sum(x) == 0:
        return np.nan

    x = np.sort(x)
    N = len(x)

    B = sum(xi * (N - i) for i, xi in enumerate(x)) / (N * np.sum(x))
    return 1 + (1 / N) - 2 * B


def compute_degree_gini(G):
    degrees = [degree for _, degree in G.degree()]
    return gini_from_values(degrees)


def compute_weight_gini(G):
    weights = [
        data.get("weight", 1)
        for _, _, data in G.edges(data=True)
    ]
    return gini_from_values(weights)

def collect_stats_vector(
        graph,
        links=True,
        weights=True,
        nodes=False,
        CC=True,
        gini=True,
        weight_gini=True,
        ASPL=True,
        dyads=False):

    vector = {}
    n = len(graph.nodes)

    if dyads:
        if graph.number_of_edges() > 0:
            max_w = 0
            ones = 0

            for _, _, data in graph.edges(data=True):
                w = data.get("weight", 1)

                if w > max_w:
                    max_w = w

                if w == 1:
                    ones += 1

            vector.update({"Max weight dyad": max_w})
            vector.update({"Nr of 1-int links": ones})

        else:
            vector.update({"Max weight dyad": 0})
            vector.update({"Nr of 1-int links": 0})

    if nodes:
        vector.update({"Active nodes": n})

    if links:
        vector.update({"Links": len(graph.edges)})

    if weights:
        graphweights = sum(
            graph.edges[edge]["weight"]
            for edge in graph.edges
        )
        vector.update({"Weights": graphweights})

    if CC:
        if n == 0:
            vector.update({"CC": 0})
        else:
            vector.update({"CC": nx.average_clustering(graph)})

    if gini:
        vector.update({"Gini of degree": compute_degree_gini(graph)})

    if weight_gini:
        vector.update({"Gini of weights": compute_weight_gini(graph)})

    if ASPL:
        if graph.number_of_nodes() == 0 or graph.number_of_edges() == 0:
            vector.update({"ASPL": 0})
        else:
            largest_cc = max(nx.connected_components(graph), key=len)

            if len(largest_cc) <= 1:
                vector.update({"ASPL": 0})
            else:
                subgraph = graph.subgraph(largest_cc)
                vector.update({
                    "ASPL": nx.average_shortest_path_length(subgraph)
                })

    return vector


ABC_KEYS = ("Links", "Weights", "CC", "ASPL", "Gini of degree", "Gini of weights")
WEEKEND_IDX = {0, 6, 7, 13, 14, 20, 21, 27}
WEEKDAY_SDS = dict(zip(ABC_KEYS, [1453.1498, 14317.2470, .024286, .101558, .028388, .0225857]))
MONTH_SDS = dict(zip(ABC_KEYS, [2798.589303, 65329.249586, .020591, .027243, .007980, .007687]))


def load_data(path="bt_symmetric.csv"):
    data = pd.read_csv(path)
    study = data[data["user_b"] > -2]
    total = study["# timestamp"].nunique()
    coverage = study.groupby("user_a")["# timestamp"].nunique().add(
        study.groupby("user_b")["# timestamp"].nunique(), fill_value=0)
    eligible = coverage[coverage > total * .6].index
    data = study[study.user_a.isin(eligible) & study.user_b.isin(eligible)
                 & (study.rssi != 0)].copy()
    if (data.user_a == data.user_b).any():
        raise ValueError("Self contacts found: resolve before constructing simple graphs.")
    if data.empty or data["# timestamp"].min() < 0 or data["# timestamp"].max() >= 8064*300:
        raise ValueError("Expected nonempty data with timestamps within the 28-day month.")
    return data


def cumulative_graph(data, day=None):
    # day is numbered 1 to 28; None means the whole month.
    if day is not None:
        start = (day-1)*288*300
        data = data[(data["# timestamp"] >= start) & (data["# timestamp"] < start+288*300)]
    graph = nx.Graph()
    for row in data.itertuples():
        if graph.has_edge(row.user_a, row.user_b):
            graph[row.user_a][row.user_b]["weight"] += 1
        else:
            graph.add_edge(row.user_a, row.user_b, weight=1)
    return graph


def load_empirical_graphs(path="bt_symmetric.csv"):
    data = load_data(path)
    return {"Weekday": [cumulative_graph(data, day+1) for day in range(28)
                        if day not in WEEKEND_IDX],
            "Month": [cumulative_graph(data)]}


def load_targets(path="bt_symmetric.csv"):
    graphs = load_empirical_graphs(path)
    weekday = pd.DataFrame([collect_stats_vector(g) for g in graphs["Weekday"]]).mean().to_dict()
    month = collect_stats_vector(graphs["Month"][0])
    return weekday, month


if __name__ == "__main__":
    graphs = load_empirical_graphs()
    rows = []
    for scale, group in graphs.items():
        for number, graph in enumerate(group, 1):
            rows.append(dict(Scale=scale, Graph=number,
                             **collect_stats_vector(graph, nodes=True)))
    Path("results").mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv("results/empirical_graph_statistics.csv", index=False)
    print(pd.DataFrame(rows).groupby("Scale").mean(numeric_only=True))
