from pathlib import Path
import json
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from pyabc.storage.db_model import ABCSMC, Population, Model, Particle, Sample, SummaryStatistic
import networkx as nx
import numpy as np
import pandas as pd
import pyabc
from ABC_runs import RUNS
from Preprocessing import ABC_KEYS, WEEKDAY_SDS, MONTH_SDS, load_targets, load_empirical_graphs


'''
4. COLLECTING RESULTS.
This produces .csv files storing the results, retrieving them from the .db files created
by the ABC runs. 

'''

STATS = list(ABC_KEYS)
RESULTS_DIR = Path("results")

def weighted_quantile(values, weights):
    """Q05, median, Q95 using the original midpoint interpolation convention."""
    values, weights = np.asarray(values, float), np.asarray(weights, float)
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    positions = (np.cumsum(weights) - 0.5 * weights) / weights.sum()
    return np.interp([0.05, 0.5, 0.95], positions, values)


def gini(values):
    x = np.sort(np.asarray(values, float))
    n = len(x)
    return 2 * np.sum(np.arange(1, n + 1) * x) / (n * x.sum()) - (n + 1) / n


def compact_distribution(graph):
    """Only retain degrees and positive-weight frequencies, never a graph."""
    degrees = np.asarray([d for _, d in graph.degree()], dtype=np.int32)
    values, counts = np.unique(
        [d["weight"] for _, _, d in graph.edges(data=True)], return_counts=True)
    return {"degrees": degrees, "weight_values": values, "weight_counts": counts}


def population_compact(path, analysis_id, generation):
    """Read one particle's summaries at a time, without joining parameters.

    get_population_extended joins every statistic with every parameter, which
    repeats the large edge payload before pandas pivots it. This narrow query
    avoids that multiplication, uses pyABC's own value decoder, and immediately
    reduces each network to small arrays. Session/engine always close on error.
    """
    engine = create_engine(f"sqlite:///{path}")
    session = sessionmaker(bind=engine)()
    rows, distributions = [], []
    try:
        ids = (session.query(Particle.id, Particle.w)
               .select_from(Population).join(ABCSMC).join(Model).join(Particle)
               .filter(ABCSMC.id == analysis_id, Population.t == generation, Model.m == 0)
               .order_by(Particle.id).all())  # IDs/weights only: about 100 rows.
        for particle_id, weight in ids:
            summaries, sample_ids = {}, set()
            query = (session.query(SummaryStatistic.name, SummaryStatistic.value, Sample.id)
                     .select_from(SummaryStatistic).join(Sample).join(Particle)
                     .filter(Particle.id == particle_id)
                     .filter(SummaryStatistic.name.in_(STATS + ["network_edges"]))
                     .yield_per(1))
            for name, value, sample_id in query:
                sample_ids.add(sample_id)
                if name in summaries:
                    raise ValueError(f"Duplicate statistic {name!r} for particle {particle_id}")
                summaries[name] = value
            if len(sample_ids) != 1:
                raise ValueError(f"Expected one stored sample for particle {particle_id}")
            serialized = summaries.pop("network_edges")
            edges = json.loads(serialized) if isinstance(serialized, str) else serialized
            graph = nx.Graph()
            graph.add_weighted_edges_from(edges)
            dist = compact_distribution(graph)
            for stat in STATS[4:]:
                if stat not in summaries:
                    values = (dist["degrees"] if stat == STATS[4] else
                              np.asarray([d["weight"] for _, _, d in graph.edges(data=True)]))
                    summaries[stat] = gini(values)
            missing = set(STATS) - set(summaries)
            if missing:
                raise KeyError(f"Particle {particle_id} lacks {sorted(missing)}")
            rows.append({"particle_id": particle_id, "w": weight,
                         **{f"sumstat_{stat}": float(summaries[stat]) for stat in STATS}})
            distributions.append(dist)
            del graph, edges, serialized, summaries, query
        return pd.DataFrame(rows), distributions
    finally:
        session.close()
        engine.dispose()


def distribution_rows(distribution, run, scale, particle, weight):
    rows = []
    degree_values, degree_counts = np.unique(distribution["degrees"], return_counts=True)
    for statistic, values, counts in [
            ("Degree", degree_values, degree_counts),
            ("Weight", distribution["weight_values"], distribution["weight_counts"])]:
        for value, count in zip(values, counts):
            rows.append(dict(Run=run, Scale=scale, Particle=particle, Posterior_weight=weight,
                             Distribution=statistic, Value=int(value), Count=int(count)))
    return rows


def main():
    # Check all inputs before writing any output.
    for _, _, database, _ in RUNS:
        if not Path(database).is_file():
            raise FileNotFoundError(database)
    weekday, month = load_targets()
    observed = {"Weekday": weekday, "Month": month}
    sds = {"Weekday": WEEKDAY_SDS, "Month": MONTH_SDS}
    measures, parameters, particles, distributions, diagnostics = [], [], [], [], []
    RESULTS_DIR.mkdir(exist_ok=True)
    for run, scale, database, fitted_count in RUNS:
        path = Path(database).resolve()
        history = pyabc.History(f"sqlite:///{path}")
        generation = history.max_t
        posterior, parameter_weights = history.get_distribution(m=0, t=generation)
        population, compact = population_compact(path, history.id, generation)
        if set(population.particle_id) != set(posterior.index):
            raise ValueError(f"{run}: particle IDs differ between summaries and parameters")
        source_weights = pd.Series(np.asarray(parameter_weights), index=posterior.index)
        posterior = posterior.reindex(population.particle_id)
        weights = population.w.to_numpy(float)
        weights /= weights.sum()
        # The same particle weights govern parameters, summaries and distributions.
        aligned_weights = source_weights.reindex(population.particle_id).to_numpy(float)
        aligned_weights /= aligned_weights.sum()
        if not np.allclose(weights, aligned_weights):
            raise ValueError(f"{run}: inconsistent particle weights")
        print(f"{run}: generation {generation}, {len(population)} particles", flush=True)
        diagnostics.append(dict(Run=run, Scale=scale, Database=database, Generation=generation,
                                Particles=len(weights), ESS=1/np.sum(weights**2)))
        for parameter in posterior.columns:
            values = posterior[parameter].to_numpy(float)
            low, median, high = weighted_quantile(values, weights)
            mean = np.sum(weights*values)
            parameters.append(dict(Run=run, Scale=scale, Parameter=parameter,
                                   Mean=mean, SD=np.sqrt(np.sum(weights*(values-mean)**2)),
                                   Q05=low, Median=median, Q95=high))
        for index, statistic in enumerate(STATS):
            low, median, high = weighted_quantile(population[f"sumstat_{statistic}"], weights)
            target = observed[scale][statistic]
            measures.append(dict(Run=run, Scale=scale, Statistic=statistic,
                                 Fitted=index<fitted_count, Observed=target,
                                 Observed_SD=sds[scale][statistic], Q05=low, Median=median, Q95=high,
                                 Signed_difference_pct=100*(median/target-1)))
        for i, row in population.iterrows():
            particle = int(row.particle_id)
            record = dict(Run=run, Scale=scale, Particle=particle, Posterior_weight=weights[i])
            record.update({k:float(posterior.iloc[i][k]) for k in posterior.columns})
            record.update({k:float(row[f"sumstat_{k}"]) for k in STATS})
            particles.append(record)
            distributions.extend(distribution_rows(compact[i], run, scale, particle, weights[i]))
        # Progress survives failure in a later database.
        for name, rows in [("posterior_summary",measures),("parameter_summary",parameters),
                           ("posterior_particles",particles),("distributions",distributions),
                           ("calibration_diagnostics",diagnostics)]:
            pd.DataFrame(rows).to_csv(RESULTS_DIR/f"{name}.csv", index=False)
    empirical = []
    for scale, graphs in load_empirical_graphs().items():
        for i, graph in enumerate(graphs):
            empirical.extend(distribution_rows(compact_distribution(graph), "Data", scale, i, 1/len(graphs)))
    pd.DataFrame(empirical).to_csv(RESULTS_DIR/"empirical_distributions.csv", index=False)
    print(pd.DataFrame(measures).to_string(index=False))


if __name__ == "__main__":
    main()
