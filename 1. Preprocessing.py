import numpy as np
import pandas as pd
import networkx as nx


'''
1. DATA PROCESSING
This code retrieves, preprocesses, and constructs networks from the data. It assumes the bt_symmetric.csv file to be in the same folder as the python file. 
This is the bluetooth interaction data from the publically available month of Copenhagen Social Network Data, 
available for download here: https://figshare.com/articles/dataset/The_Copenhagen_Networks_Study_interaction_data/7267433
'''

#Filter the data: remove observations where a b-user was found to be proximate,
#but not an in-study user. For these interactions, "user_b" was labeled -2 in the dataset.
#Following Stopczynski et al. 2018 (https://www.nature.com/articles/s41598-018-36116-6), 
#the data is filtered for quality to only include the agents for which we observations at least 
#60% of the time.

data_bt = pd.read_csv('bt_symmetric.csv')
dfbt= pd.DataFrame(data_bt)
dfbt_instudy = dfbt[(dfbt["user_b"] >(-2))]
total_timestamps = dfbt_instudy["# timestamp"].nunique()
user_a_at_timestamp = dfbt_instudy.groupby('user_a')["# timestamp"].nunique()
user_b_at_timestamp = dfbt_instudy.groupby('user_b')["# timestamp"].nunique()
total_user_timestamps_count = user_a_at_timestamp.add(user_b_at_timestamp, fill_value=0)
eligible = total_user_timestamps_count[total_user_timestamps_count > (total_timestamps * 0.6) ]
dfbt_filtered = dfbt_instudy[dfbt_instudy['user_a'].isin(eligible.index) | dfbt_instudy['user_b'].isin(eligible.index)]

#Function to create either the whole cumulative graph over a month (28 days, 8064 5-minute
#timesteps), or any of the 28 days (288 timesteps) (Links: interactions at any time,
#link weights: number of interactions of that dyad).

def cumulative_graph(day=None, All=False):
    Cumulative_BT_Graph = nx.Graph()
    #Timestamps are logged in seconds but binned per 5 minutes. For example, the first timebin is 0 and the second is 300. 
    #So if we want one day, that's 288 bins of five minutes, and multiply by 300 to get the correct timestamp.
    if day != None:
      starttime = (day - 1)*288 *300
      endtime = day*288*300
    if All == True:
      df_day = dfbt_filtered
    else:
      df_day = dfbt_filtered[
            (dfbt_filtered["# timestamp"] >= starttime) &
            (dfbt_filtered["# timestamp"] < endtime)]
    for row in df_day.itertuples():
      user_a=row.user_a
      user_b=row.user_b
      if (user_a in eligible.index) and (user_b in eligible.index) and row.rssi != 0:
        if Cumulative_BT_Graph.has_edge(user_a, user_b):
            Cumulative_BT_Graph[user_a][user_b]["weight"] +=1
        else: 
            Cumulative_BT_Graph.add_edge(user_a,user_b,weight=1)
    return Cumulative_BT_Graph

#Function to collect all kinds of output measures from the cumulative networks
#(and some helper functions):

def dyadic_counts_from_graph(G):
  counts = [(u, v, data['weight']) for u, v, data in G.edges(data=True)]
  df_counts = pd.DataFrame(counts, columns=['user_a', 'user_b', 'interactions'])
  df_counts = df_counts.sort_values('interactions', ascending=False).reset_index(drop=True)
  return df_counts

def compute_gini(G):
  degrees = list(G.degree)
  N = len(degrees)
  x = sorted([i[1] for i in degrees])
  if N*sum(x) !=0:
    B = sum( xi * (N-i) for i,xi in enumerate(x) ) / (N*sum(x))
    return (1 + (1/N) - 2*B)
  else:
    return np.nan

def collect_stats_vector(graph,links=True,weights=True,nodes=False,CC=True,gini=True,ASPL=True,dyads=True):
    vector = {}
    n = len(graph.nodes)
    if dyads == True:
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
    #the collect_stats_vector is called either for real-data graphs, where
    #only nodes with measurements at T are included; or in the ABC simulation
    #after we filter for 0-degree nodes. So "active nodes" just = nodes:
    if nodes:
      vector.update({"Active nodes" : n})
    if links:
      vector.update({"Links": len(graph.edges)})
    if weights:
      graphweights = sum([graph.edges[edge]["weight"] for edge in graph.edges])
      vector.update({"Weights" : graphweights})
    if CC:
       if n == 0:
        vector.update({"CC" : 0})
       else:
        vector.update({"CC" : nx.average_clustering(graph)})
    if gini:
      vector.update({"Gini of degree": compute_gini(graph)})
    if ASPL:
      if graph.number_of_nodes() == 0 or graph.number_of_edges() == 0:
          vector.update({"ASPL": 0})
      else:
          # largest connected component
          largest_cc = max(nx.connected_components(graph), key=len)
          if len(largest_cc) <= 1:
              vector.update({"ASPL": 0})
          else:
              subgraph = graph.subgraph(largest_cc)
              vector.update({"ASPL": nx.average_shortest_path_length(subgraph)})
    return vector

#Function to calculate the average weekday graph properties:

def mean_weekday_stats_vector(
        Alldays,
        weekend_idx={0,6,7,13,14,20,21,27},
        links=True, weights=True, CC=True, ASPL=True, gini=False, dyads=False,nodes=False):
    
    #Compute the mean statistics vector across weekday cumulative graphs.

    weekday_graphs = [
        g for i, g in enumerate(Alldays)
        if i not in weekend_idx
    ]

    stat_vectors = [
        collect_stats_vector(
            g,
            links=links,
            weights=weights,
            CC=CC,
            ASPL=ASPL,
            gini=gini,
            dyads=dyads,
            nodes=nodes,
        )
        for g in weekday_graphs
    ]

    keys = stat_vectors[0].keys()

    mean_vector = {
        k: float(np.mean([sv[k] for sv in stat_vectors]))
        for k in keys
    }

    return mean_vector

#Single day cumulative graphs:
Alldays = list()
for day in list(range(1,29)):
  Cumgraph = cumulative_graph(day)
  Alldays.append(Cumgraph)

stats_vector_whole_cumulative_graph = collect_stats_vector(
    whole_cumulative_graph, gini=False, dyads=False, nodes=False
)
stats_vector_whole_cumulative_graph_allstats = collect_stats_vector(
    whole_cumulative_graph, gini=True, dyads=True, nodes=False
)
stats_vector_weekday_avg_macro = mean_weekday_stats_vector(Alldays, gini=False, dyads=False, nodes=False)
stats_vector_weekday_avg_allstats = mean_weekday_stats_vector(Alldays, gini=True, dyads=True, nodes=False)
stats_vector_day2 = collect_stats_vector(Alldays[1])

#aliases used by later scripts.
stats_vector_wholegraph = stats_vector_whole_cumulative_graph
stats_vector_wholegraph_allstats = stats_vector_whole_cumulative_graph_allstats
stats_vector_avg_weekday = stats_vector_weekday_avg_macro
stats_vector_weekday_avg_nogini = stats_vector_weekday_avg_allstats

