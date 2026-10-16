import json
import math
from pathlib import Path
import numpy as np
import pyabc
from pyabc import RV
from pyabc.sampler import MulticoreEvalParallelSampler
from Models import Simulation_lean, Simulation_scheduledmobility
from Preprocessing import ABC_KEYS, WEEKDAY_SDS, MONTH_SDS, collect_stats_vector, load_targets

'''
3. APPROXIMATE BAYESIAN COMPUTATION. 
Uses the pyabc package. To choose one of or more of the 4 runs presented in the paper,
set the RUNS list below to include the corresponding names. Default is the 
random movement model month test. Runtimes are as follows:
Test 1: Random movement model, month data.
Runtime on 12 cores: 5 hours.
Test 2: Random movement model, average weekday data.
Runtime on 12 cores: 1 hour.
Test 3: Scheduled mobility model, month data.
Runtime on 12 cores: 41 hours!
Test 4: Scheduled mobility model, average weekday data.
Runtime on 12 cores: 4 hours.

'''


# label, scale, database, number of fitted statistics
RUNS = [
    ("RM_WEEKDAY", "Weekday", "RM_WEEKDAY_FINAL.db", 6),
    ("SM_WEEKDAY", "Weekday", "SM_WEEKDAY_FINAL_V4.db", 6),
    ("RM_MONTH", "Month", "RM_MONTH_FINAL.db", 6),
    ("RM_MONTH_4STAT", "Month", "RM_MONTH_SD.db", 4),
    ("SM_MONTH", "Month", "SM_MONTH_FINAL_V3.db", 6),
]
RUN_TESTS = [row[0] for row in RUNS]
POPULATION_SIZE = 100
MAX_NR_POPULATIONS = 10
MINIMUM_EPSILON = .0125


def sd_distance(x, observed, sds, keys=ABC_KEYS):
    return math.sqrt(sum(((x[key]-observed[key])/sds[key])**2 for key in keys))


def distance_weekday(x, observed):
    return sd_distance(x, observed, WEEKDAY_SDS)


def distance_month(x, observed):
    return sd_distance(x, observed, MONTH_SDS)


def distance_month_4stat(x, observed):
    return sd_distance(x, observed, MONTH_SDS, ABC_KEYS[:4])


def simulate(parameters, n, t, model):
    simulation = model(N=n, T=t, **parameters)
    simulation.run()
    graph = simulation.G.subgraph([node for node, degree in simulation.G.degree() if degree > 0])
    result = collect_stats_vector(graph)
    result["network_edges"] = json.dumps([(int(u), int(v), int(d["weight"]))
                                          for u, v, d in graph.edges(data=True)])
    return result


def rm_weekday(p): return simulate(p, 409, 288, Simulation_lean)
def rm_month(p): return simulate(p, 498, 8064, Simulation_lean)
def sm_weekday(p): return simulate(p, 409, 288, Simulation_scheduledmobility)
def sm_month(p): return simulate(p, 498, 8064, Simulation_scheduledmobility)


def main():
    weekday, month = load_targets()
    rm_prior = pyabc.Distribution(r=RV("uniform", 0, 50), s=RV("uniform", 0, 100))
    # scipy uniform uses loc and width: WM is [1,101], as in the supplied runner.
    sm_prior = pyabc.Distribution(r=RV("uniform", 0, 40), s=RV("uniform", 0, 1000),
                                  PW=RV("uniform", 0, 1), WM=RV("uniform", 1, 100))
    definitions = {
        "RM_WEEKDAY": (rm_weekday, rm_prior, weekday, distance_weekday),
        "SM_WEEKDAY": (sm_weekday, sm_prior, weekday, distance_weekday),
        "RM_MONTH": (rm_month, rm_prior, month, distance_month),
        "RM_MONTH_4STAT": (rm_month, rm_prior, {k: month[k] for k in ABC_KEYS[:4]}, distance_month_4stat),
        "SM_MONTH": (sm_month, sm_prior, month, distance_month),
    }
    if len(set(RUN_TESTS)) != len(RUN_TESTS) or set(RUN_TESTS)-definitions.keys():
        raise ValueError("Invalid RUN_TESTS")
    selected = [row for row in RUNS if row[0] in RUN_TESTS]
    for _, _, database, _ in selected:
        if Path(database).exists():
            raise FileExistsError(f"{database} exists; remove this run from RUN_TESTS or choose a new name.")
    print("Weekday targets:", weekday)
    print("Month targets:", month)
    for name, scale, database, count in selected:
        model, prior, observed, distance = definitions[name]
        print(f"Starting {name}: {count} statistics", flush=True)
        abc = pyabc.ABCSMC(model, parameter_priors=prior, distance_function=distance,
                           population_size=POPULATION_SIZE,
                           sampler=MulticoreEvalParallelSampler(pickle=True))
        abc.new(f"sqlite:///{Path(database).resolve()}", observed)
        history = abc.run(minimum_epsilon=MINIMUM_EPSILON, max_nr_populations=MAX_NR_POPULATIONS)
        print(f"Finished {database}: generation {history.max_t}", flush=True)


if __name__ == "__main__":
    main()
