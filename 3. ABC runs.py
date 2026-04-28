import pyabc
from pyabc import RV
from pyabc import ABCSMC
from pyabc import Model
from pyabc.sampler import MulticoreEvalParallelSampler
import tempfile
import os, json
from scipy.sparse import csr_matrix
import math

'''
3. APPROXIMATE BAYESIAN COMPUTATION. 
Uses the pyabc package. To choose one of or more of the 4 runs presented in the paper,
set the RUN_TESTS list below to include the corresponding names. Default is the 
random movement model month test. Runtimes are as follows:
Test 1: Random movement model, month data, 4 macro-level network measures.
Runtime on 12 cores: 5 hours.
Test 2: Random movement model, average weekday data, 4 macro-level network measures.
Runtime on 12 cores: 1 hour.
Test 3: Scheduled mobility model, month data, all network measures.
Runtime on 12 cores: 41 hours!
Test 4: Scheduled mobility model, average weekday data, all network measures.
Runtime on 12 cores: 4 hours.

Alternatively, set your own ABC and simulation parameter settings and 
choose an observed stats vector to compare to.

'''

# Choose which tests to run when this file is executed directly.
# Options: RM_MONTH, RM_WEEKDAY, SM_MONTH, SM_WEEKDAY
RUN_TESTS = ["RM_MONTH"]

from preprocessing import (
    collect_stats_vector,
    stats_vector_wholegraph,
    stats_vector_avg_weekday,
    stats_vector_wholegraph_allstats,
    stats_vector_weekday_avg_nogini,
)
from Models import Simulation_lean, Simulation_scheduledmobility

#Global ABC settings that remain the same for all runs.
POPULATION_SIZE = 200
MAX_NR_POPULATIONS = 10
MINIMUM_EPSILON = 0.0125
SAMPLER = MulticoreEvalParallelSampler(pickle=True)


#In all tests, we use the following distance function calculating difference between simulated 
#and observed outcomes: Euclidean distance normalized by the observed value.
#This normalization is to prevent measures with large absolute values to disproportionately impact the 
#overall distance.

def relative_distance(x, x0):
  keys = [k for k in x0.keys()] #the simulation runs will also store network edges,
  #but we don't want those in the distance calculation.
  return math.sqrt(sum(((x[k] - x0[k]) / (x0[k]))**2 for k in keys))
  #Hypothetically there's a /0 error here, but we know that the observed
  #dictionaries have no 0 values for network measures.

#Functions that will be used in the ABC run:


#This stores the specific networks, so we don't just have network measure outputs but can also
#retrieve the best network object itself.
def graph_to_edges_json(G):
    return json.dumps([(u, v, int(data.get("weight", 1))) for u, v, data in G.edges(data=True)])

#The data that only includes users _when_ proximate. Our simulation includes the whole 
#mobility process, which will produce many timesteps with isolated nodes, not at the 
#moment connected to any other. To make a sound comparison, we chose to cut these
#isolates from the simulation results. This makes N not just population size, but 
#active population size, as in the data.
def active_subgraph(G):
    active_nodes = [node for node in G.nodes if G.degree(node) > 0]
    return G.subgraph(active_nodes).copy()


def simulate_random_month(parameters):
    sim = Simulation_lean(N=498, T=8064, s=parameters["s"], r=parameters["r"])
    sim.run()
    G_active = active_subgraph(sim.G)
    sim_vector = collect_stats_vector(G_active, nodes=False, gini=False, dyads=False)
    sim_vector["network_edges"] = graph_to_edges_json(sim.G)
    return sim_vector


def simulate_random_weekday(parameters):
    sim = Simulation_lean(N=409, T=288, s=parameters["s"], r=parameters["r"])
    sim.run()
    G_active = active_subgraph(sim.G)
    sim_vector = collect_stats_vector(G_active, nodes=False, gini=False, dyads=False)
    sim_vector["network_edges"] = graph_to_edges_json(sim.G)
    return sim_vector


def simulate_scheduled_month(parameters):
    sim = Simulation_scheduledmobility(N=498, T=8064, s=parameters["s"], r=parameters["r"], PW=0.5, WM=25)
    sim.run()
    G_active = active_subgraph(sim.G)
    sim_vector = collect_stats_vector(G_active, nodes=False, gini=True, dyads=True)
    sim_vector["network_edges"] = graph_to_edges_json(sim.G)
    return sim_vector


def simulate_scheduled_weekday(parameters):
    sim = Simulation_scheduledmobility(N=409, T=288, s=parameters["s"], r=parameters["r"], PW=0.5, WM=25)
    sim.run()
    G_active = active_subgraph(sim.G)
    sim_vector = collect_stats_vector(G_active, nodes=False, gini=True, dyads=True)
    sim_vector["network_edges"] = graph_to_edges_json(sim.G)
    return sim_vector


histories = {}

def run_abc(name, simulate_fn, prior, observed):
    abc = ABCSMC(
        simulate_fn,
        parameter_priors=prior,
        distance_function=relative_distance,
        population_size=POPULATION_SIZE,
        sampler=SAMPLER,
    )

    db = f"sqlite:///{name}.db"
    abc.new(db, observed)

    histories[name] = abc.run(
        minimum_epsilon=MINIMUM_EPSILON,
        max_nr_populations=MAX_NR_POPULATIONS
    )

    print(f"Finished {name}. Database: {name}.db")
    return histories[name]

#Defining prior distributions: we use the simplest assumption, uniform priors (aka naive or uninformed priors). 
#Ranges were set according to exploratory testing. 

prior_rm = pyabc.Distribution(
    r=RV("uniform", 0, 50),
    s=RV("uniform", 0, 100),
)

prior_sm_month = pyabc.Distribution(
    r=RV("uniform", 0, 40),
    s=RV("uniform", 0, 500),
)

prior_sm_weekday = pyabc.Distribution(
    r=RV("uniform", 0, 40),
    s=RV("uniform", 0, 300),
)

#The actual runs:

if "RM_MONTH" in RUN_TESTS:
    run_abc("RM_MONTH", simulate_rm_month, prior_rm, stats_vector_whole_cumulative_graph)

if "RM_WEEKDAY" in RUN_TESTS:
    run_abc("RM_WEEKDAY", simulate_rm_weekday, prior_rm, stats_vector_weekday_avg_allstats)

if "SM_MONTH" in RUN_TESTS:
    run_abc("SM_MONTH", simulate_sm_month, prior_sm_month, stats_vector_whole_cumulative_graph_allstats)

if "SM_WEEKDAY" in RUN_TESTS:
    run_abc("SM_WEEKDAY", simulate_sm_weekday, prior_sm_weekday, stats_vector_weekday_avg_allstats)