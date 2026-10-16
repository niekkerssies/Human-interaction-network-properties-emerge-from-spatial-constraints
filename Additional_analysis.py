from pathlib import Path
from itertools import combinations
from functools import lru_cache
import random
import time
import numpy as np
import pandas as pd
import networkx as nx
import pyabc
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, shortest_path
import Models as models
from Preprocessing import load_data, load_empirical_graphs
from Collecting_results import weighted_quantile

'''
6. ADDITIONAL ANALYSES
Run null models, temporal evolution analysis, and percolation analysis.

'''

RUN_NULLS = True
RUN_TEMPORAL = True
RUN_PERCOLATION = True
NULL_REPLICATES = 100
ATTEMPTS_PER_EDGE = 20
POSTERIOR_DRAWS = 50
SEED = 20261014
RESULTS = Path("results")
DAY, MONTH = 288, 8064
WEEKENDS = {0, 6, 7, 13, 14, 20, 21, 27}
DURATIONS = (1, 3, 6, 12, 36, 72, 144, 288, 576, 1152, 2016, 4032, 8064)
STATS = ('Links', 'Weights', 'CC', 'ASPL', 'Gini of degree', 'Gini of weights')
MEASURES = ('Active nodes', 'Largest component', *STATS, 'Density')
# Only the informative monthly RM four-statistic fit is used for temporal analysis.
TEMPORAL_RUNS = [
    ('RM_WEEKDAY', 'Weekday', 'RM_WEEKDAY_FINAL.db', 'RM'),
    ('SM_WEEKDAY', 'Weekday', 'SM_WEEKDAY_FINAL_V4.db', 'SM'),
    ('RM_MONTH_4STAT', 'Month', 'RM_MONTH_SD.db', 'RM'),
    ('SM_MONTH', 'Month', 'SM_MONTH_FINAL_V3.db', 'SM'),
]
PERCOLATION_REPLICATES = 10
R_VALUES = np.arange(0., 152., 2.)
BURN_IN_STEPS = 288
OBSERVATION_STEPS = 7*288
SAMPLE_EVERY = 12

def rewire(original, attempts, seed):
    """Uniform edge-pair proposals with random orientation; reject illegal swaps.

    Four distinct endpoints avoid loops. Existing candidate edges are rejected,
    preserving a simple graph. Keeping rejected proposals makes the transition
    symmetric; finite mixing is a practical approximation, not guaranteed.
    """
    g = original.copy()
    edges = list(g.edges())
    rng = random.Random(seed)
    accepted = 0
    if len(edges) < 2:
        return g, accepted
    for _ in range(attempts):
        i = rng.randrange(len(edges))
        j = rng.randrange(len(edges))
        if i == j:
            continue
        a, b = edges[i]
        c, d = edges[j]
        if rng.getrandbits(1):
            a, b = b, a
        if rng.getrandbits(1):
            c, d = d, c
        if len({a, b, c, d}) != 4 or g.has_edge(a, c) or g.has_edge(b, d):
            continue
        g.remove_edge(a, b)
        g.remove_edge(c, d)
        g.add_edge(a, c)
        g.add_edge(b, d)
        edges[i], edges[j] = (a, c), (b, d)
        accepted += 1
    if dict(g.degree()) != dict(original.degree()):
        raise AssertionError('Individual degree changed.')
    if nx.number_of_selfloops(g) or g.number_of_edges() != original.number_of_edges():
        raise AssertionError('Invalid randomized graph.')
    return g, accepted


def measures(g):
    nodes = list(g)
    n = len(nodes)
    if not n:
        raise ValueError('Empty empirical graph.')
    index = {v: i for i, v in enumerate(nodes)}
    rows, cols = [], []
    for u, v in g.edges():
        rows.extend([index[u], index[v]])
        cols.extend([index[v], index[u]])
    a = csr_matrix((np.ones(len(rows), dtype=np.int64), (rows, cols)), shape=(n, n))
    degree = np.asarray(a.sum(axis=1)).ravel()
    closed = np.asarray((a @ a).multiply(a).sum(axis=1)).ravel()
    local_cc = np.zeros(n)
    valid = degree > 1
    local_cc[valid] = closed[valid] / (degree[valid] * (degree[valid] - 1))
    _, labels = connected_components(a, directed=False)
    members = np.flatnonzero(labels == np.argmax(np.bincount(labels)))
    size = len(members)
    if size > 1:
        distances = shortest_path(a[members][:, members], directed=False, unweighted=True)
        aspl = float(distances.sum() / (size * (size - 1)))
    else:
        aspl = 0.0
    return {'N': n, 'M': g.number_of_edges(), 'Largest component': size,
            'CC': float(local_cc.mean()), 'ASPL': aspl}


def run_nulls():
    graphs = load_empirical_graphs()
    raw, observed = [], []
    rng = np.random.default_rng(SEED)
    for scale, group in graphs.items():
        observed.append(dict(Scale=scale, **pd.DataFrame([measures(g) for g in group]).mean().to_dict()))
        for rep in range(1, NULL_REPLICATES+1):
            for null in ["DENSITY_MATCHED", "DEGREE_PRESERVING"]:
                for i, reference in enumerate(group):
                    seed = int(rng.integers(0,2**32-1))
                    attempts = ATTEMPTS_PER_EDGE*reference.number_of_edges()
                    if null=="DENSITY_MATCHED":
                        graph=nx.gnm_random_graph(len(reference),reference.number_of_edges(),seed=seed)
                        accepted=np.nan;overlap=np.nan
                    else:
                        graph,accepted=rewire(reference,attempts,seed)
                        overlap=sum(graph.has_edge(u,v) for u,v in reference.edges())/reference.number_of_edges()
                    raw.append(dict(Run=null,Scale=scale,Graph=i+1,Replicate=rep,Seed=seed,
                                    Attempts=attempts if null=="DEGREE_PRESERVING" else 0,
                                    Accepted_swaps=accepted,Original_edge_overlap=overlap,**measures(graph)))
            # Save each complete ensemble before starting the next.
            pd.DataFrame(raw).to_csv(RESULTS/"null_replicates.csv",index=False)
            print(f"Nulls {scale}: {rep}/{NULL_REPLICATES}",flush=True)
    frame=pd.DataFrame(raw);summary=[]
    for (run,scale),group in frame.groupby(["Run","Scale"]):
        means=group.groupby("Replicate")[["CC","ASPL"]].mean()
        obs=next(row for row in observed if row["Scale"]==scale)
        for stat in ["CC","ASPL"]:
            x=means[stat].to_numpy();lo,median,hi=np.quantile(x,[.05,.5,.95])
            summary.append(dict(Run=run,Scale=scale,Statistic=stat,Observed=obs[stat],
                                Mean=x.mean(),Q05=lo,Median=median,Q95=hi,
                                Signed_difference_pct=100*(median/obs[stat]-1)))
    pd.DataFrame(observed).to_csv(RESULTS/"null_empirical_checks.csv",index=False)
    pd.DataFrame(summary).to_csv(RESULTS/"null_summary.csv",index=False)


def gini(values):
    x = np.sort(np.asarray(values, float))
    if len(x) == 0 or x.sum() == 0:
        return np.nan
    return float(2*np.dot(np.arange(1,len(x)+1),x)/(len(x)*x.sum())-(len(x)+1)/len(x))


def _measure(edges):
    """Active-node CC (degree < 2 contributes zero); unweighted LCC path length."""
    if not edges:
        result = {k: 0.0 for k in MEASURES}
        result['Gini of degree'] = result['Gini of weights'] = np.nan
        return result
    nodes = sorted({n for pair in edges for n in pair})
    index = {n:i for i,n in enumerate(nodes)}
    uv = np.asarray([(index[u],index[v]) for u,v in edges], dtype=int)
    n = len(nodes)
    a = csr_matrix((np.ones(2*len(uv), dtype=np.int64),
                    (np.r_[uv[:,0],uv[:,1]],np.r_[uv[:,1],uv[:,0]])),shape=(n,n))
    degree = np.asarray(a.sum(axis=1)).ravel()
    closed = np.asarray((a@a).multiply(a).sum(axis=1)).ravel()
    cc = np.zeros(n); valid = degree > 1
    cc[valid] = closed[valid]/(degree[valid]*(degree[valid]-1))
    _, labels = connected_components(a, directed=False)
    members = np.flatnonzero(labels == np.argmax(np.bincount(labels)))
    size = len(members)
    distance = shortest_path(a[members][:,members], directed=False, unweighted=True)
    weights = list(edges.values())
    return {'Active nodes':n,'Largest component':size,'Links':len(edges),
            'Weights':float(sum(weights)), 'CC':float(cc.mean()),
            'ASPL':float(distance.sum()/(size*(size-1))) if size>1 else 0.,
            'Gini of degree':gini(degree),'Gini of weights':gini(weights),
            'Density':2*len(edges)/(n*(n-1)) if n>1 else 0.}


@lru_cache(maxsize=64)
def _small_measure(items):
    # Repeated stationary SM snapshots often have exactly the same edge set.
    return _measure(dict(items))


def measure(edges):
    if len(edges) <= 1500:
        return dict(_small_measure(tuple(sorted(edges.items()))))
    return _measure(edges)


def add_edges(target, source):
    for pair, weight in source.items():
        target[pair] = target.get(pair,0) + weight


class WindowAccumulator:
    """Keep one current graph per duration and only completed-window summaries."""
    def __init__(self, run, scale, rep, total):
        self.run, self.scale, self.rep = run, scale, rep
        self.widths = [d for d in DURATIONS if d <= total]
        self.current = {d:{} for d in self.widths}
        self.aggregates = {}
        self.window_counts = {}
        self.daily, self.prefixes = [], []
        self.cumulative = {}
        self.steps = 0

    def _add_summary(self, width, group, row):
        key = (width,group)
        self.window_counts[key] = self.window_counts.get(key,0) + 1
        if key not in self.aggregates:
            self.aggregates[key] = {stat:[0.0,0] for stat in MEASURES}
        for stat in MEASURES:
            if np.isfinite(row[stat]):
                self.aggregates[key][stat][0] += row[stat]
                self.aggregates[key][stat][1] += 1

    def observe(self, contacts):
        self.steps += 1
        add_edges(self.cumulative,contacts)
        if self.steps in self.widths:
            self.prefixes.append(dict(Run=self.run,Scale=self.scale,Replicate=self.rep,
                                      Steps=self.steps,Days=self.steps/DAY,**measure(self.cumulative)))
        for width in self.widths:
            edges = self.current[width]
            add_edges(edges,contacts)
            if self.steps % width:
                continue
            row = measure(edges)
            self._add_summary(width,'All days',row)
            day = (self.steps-1)//DAY
            if width <= DAY and (self.scale=='Weekday' or day not in WEEKENDS):
                self._add_summary(width,'Weekdays',row)
            if width == DAY:
                self.daily.append(dict(Run=self.run,Scale=self.scale,Replicate=self.rep,
                                       Day_index=day,**row))
            edges.clear()

    def finish(self):
        windows = []
        for (width,group), stats in self.aggregates.items():
            for stat,(total,count) in stats.items():
                windows.append(dict(Run=self.run,Scale=self.scale,Replicate=self.rep,
                                    Window_steps=width,Days=width/DAY,Calendar=group,
                                    Statistic=stat,Windows=count,
                                    Total_windows=self.window_counts[(width,group)],
                                    Undefined_windows=self.window_counts[(width,group)]-count,
                                    Value=total/count if count else np.nan))
        paired = []
        if self.scale=='Month':
            weekdays = [r for r in self.daily if r['Day_index'] not in WEEKENDS]
            if len(weekdays)!=20:
                raise ValueError('Expected 20 weekday daily networks.')
            month = measure(self.cumulative)
            for stat in MEASURES:
                daily = float(np.mean([r[stat] for r in weekdays]))
                paired.append(dict(Run=self.run,Replicate=self.rep,Statistic=stat,
                                   Monthly=month[stat],Mean_weekday=daily,
                                   Ratio=month[stat]/daily if np.isfinite(daily) and daily!=0 else np.nan))
            # Ratio of ratios preserves dependence within each realization.
            for a,b in combinations(STATS,2):
                daily = float(np.mean([r[a]/r[b] if r[b]!=0 else np.nan for r in weekdays]))
                monthly = month[a]/month[b] if month[b]!=0 else np.nan
                paired.append(dict(Run=self.run,Replicate=self.rep,Statistic=f'{a} / {b}',
                                   Monthly=monthly,Mean_weekday=daily,
                                   Ratio=monthly/daily if np.isfinite(daily) and daily!=0 else np.nan))
        return dict(trajectories=self.prefixes,daily=self.daily,windows=windows,paired=paired)


def empirical(path):
    print('Reading empirical contacts and preserving the calibration coverage filter...',flush=True)
    data = load_data(path)
    data['Step'] = (data['# timestamp']//300).astype(int)
    data.sort_values('Step',inplace=True)
    stream = iter(data.groupby('Step',sort=False))
    upcoming = next(stream,None)
    acc = WindowAccumulator('Data','Month',0,MONTH)
    weekday_prefixes = []
    weekday_edges = {}
    for t in range(MONTH):
        contacts = {}
        if upcoming is not None and upcoming[0]==t:
            pairs = np.sort(upcoming[1][['user_a','user_b']].to_numpy(),axis=1)
            unique,counts = np.unique(pairs,axis=0,return_counts=True)
            contacts = {(int(u),int(v)):int(w) for (u,v),w in zip(unique,counts)}
            upcoming = next(stream,None)
        acc.observe(contacts)
        day, within = divmod(t,DAY)
        if day not in WEEKENDS:
            add_edges(weekday_edges,contacts)
            if within+1 in DURATIONS:
                weekday_prefixes.append(dict(Run='Data',Scale='Weekday',Replicate=day+1,
                                             Steps=within+1,Days=(within+1)/DAY,**measure(weekday_edges)))
        if within+1==DAY:
            weekday_edges.clear()
            print(f'Empirical day {day+1}/28 summarized',flush=True)
    result = acc.finish()
    result['trajectories'] += weekday_prefixes
    return result


def instant_contacts(sim, kind):
    nodes = list(sim.G.nodes)
    if kind=='SM':
        nodes = [n for n in nodes if not (sim.G.nodes[n]['external'] and
                 sim.G.nodes[n]['position']==sim.G.nodes[n]['home'])]
    if len(nodes)<2:
        return {}
    positions=np.asarray([sim.G.nodes[n]['position'] for n in nodes])
    return {tuple(sorted((nodes[i],nodes[j]))):1
            for i,j in cKDTree(positions).query_pairs(sim.r)}


def simulate(models, kind, scale, parameters, run, rep, seed):
    random.seed(seed);np.random.seed(seed)
    n,total = (409,DAY) if scale=='Weekday' else (498,MONTH)
    cls = models.Simulation_lean if kind=='RM' else models.Simulation_scheduledmobility
    sim = cls(N=n,T=total,**parameters)
    acc = WindowAccumulator(run,scale,rep,total)
    method = 'update_edges_2' if kind=='RM' else 'update_edges'
    native = getattr(sim,method)
    def record(*args,**kwargs):
        value = native(*args,**kwargs)
        acc.observe(instant_contacts(sim,kind))
        return value
    setattr(sim,method,record)
    start = time.perf_counter()
    sim.run()
    seconds = time.perf_counter()-start
    if acc.steps!=total:
        raise RuntimeError(f'Observed {acc.steps} contact updates; expected {total}.')
    native_edges = {tuple(sorted((u,v))):float(d['weight']) for u,v,d in sim.G.edges(data=True)}
    if native_edges != acc.cumulative:
        raise RuntimeError('Observer contacts differ from native model edges; no result saved.')
    result = acc.finish()
    result['parameters']=[dict(Run=run,Replicate=rep,Seed=seed,N=n,Runtime_seconds=seconds,**parameters)]
    _small_measure.cache_clear()
    return result


def summarize(frame, groups, value='Value'):
    rows=[]
    for key,g in frame.groupby(groups,dropna=False,sort=False):
        key = key if isinstance(key,tuple) else (key,)
        x = g[value].to_numpy(float);x=x[np.isfinite(x)]
        q=np.quantile(x,[.05,.25,.5,.75,.95]) if len(x) else [np.nan]*5
        rows.append(dict(zip(groups,key),Count=len(x),Mean=x.mean() if len(x) else np.nan,
                         SD=x.std(ddof=1) if len(x)>1 else np.nan,
                         Q05=q[0],Q25=q[1],Median=q[2],Q75=q[3],Q95=q[4]))
    return pd.DataFrame(rows)


def long_trajectories(frame):
    return frame.melt(id_vars=['Run','Scale','Replicate','Steps','Days'],
                      value_vars=MEASURES,var_name='Statistic',value_name='Value')


def read_posterior(database, kind):
    if not Path(database).is_file():
        raise FileNotFoundError(database)
    history=pyabc.History(f"sqlite:///{Path(database).resolve()}")
    posterior,weights=history.get_distribution(m=0,t=history.max_t)
    keys=['r','s'] if kind=='RM' else ['r','s','PW','WM']
    posterior=posterior[keys]
    weights=np.asarray(weights,float);weights/=weights.sum()
    return posterior,weights


def export_temporal(results):
    def combine(key):return pd.DataFrame([row for result in results for row in result.get(key,[])])
    for key,name in [('trajectories','trajectories'),('daily','daily_networks'),
                     ('windows','window_means_per_realization'),('paired','paired_month_weekday_ratios'),
                     ('parameters','sampled_parameters')]:
        combine(key).to_csv(RESULTS/f"{name}.csv",index=False)
    ts=summarize(long_trajectories(combine('trajectories')),['Run','Scale','Steps','Days','Statistic'])
    ws=summarize(combine('windows'),['Run','Scale','Window_steps','Days','Calendar','Statistic'])
    ts.to_csv(RESULTS/'trajectory_summary.csv',index=False)
    ws.to_csv(RESULTS/'window_mean_summary.csv',index=False)
    paired=combine('paired')
    ps=summarize(paired.rename(columns={'Ratio':'Value'}),['Run','Statistic'])
    observed=paired[paired.Run=='Data'].set_index('Statistic').Ratio
    ps['Empirical_ratio']=ps.Statistic.map(observed)
    ps['Signed_ratio_difference_pct']=100*(ps.Median/ps.Empirical_ratio-1)
    ps.to_csv(RESULTS/'paired_ratio_summary.csv',index=False)
    between=[]
    endpoints=combine('trajectories')
    endpoints=endpoints[endpoints.Steps.isin([DAY,MONTH])]
    for row in endpoints.to_dict('records'):
        for a,b in list(combinations(STATS,2))+[('Weights','Links')]:
            between.append(dict(Run=row['Run'],Scale=row['Scale'],Replicate=row['Replicate'],Steps=row['Steps'],
                                Ratio=f'{a} / {b}',Value=row[a]/row[b] if row[b]!=0 else np.nan))
    ratios=pd.DataFrame(between)
    ratios.to_csv(RESULTS/'within_network_ratios.csv',index=False)
    summarize(ratios,['Run','Scale','Steps','Ratio']).to_csv(RESULTS/'within_network_ratio_summary.csv',index=False)


def run_temporal():
    # Reuse one environment across draws, as in the supplied temporal analysis.
    # Intervals exclude uncertainty from reassigning the home/work building pools.
    results=[empirical('bt_symmetric.csv')]
    for index,(run,scale,database,kind) in enumerate(TEMPORAL_RUNS):
        posterior,weights=read_posterior(database,kind)
        for rep in range(1,POSTERIOR_DRAWS+1):
            seed=SEED+index*100003+rep
            particle=int(np.random.default_rng(seed).choice(len(posterior),p=weights))
            parameters=posterior.iloc[particle].to_dict()
            print(f"Temporal {run}: {rep}/{POSTERIOR_DRAWS}",flush=True)
            result=simulate(models,kind,scale,parameters,run,rep,seed)
            result['parameters'][0].update(Database=database,Posterior_particle_id=int(posterior.index[particle]))
            results.append(result)
            export_temporal(results)


class UnionFind:
    def __init__(self, n):
        self.parent = np.arange(n)
        self.size = np.ones(n, dtype=int)
        self.max_size = 1 if n else 0
        self.components = n

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        root_a = self.find(a)
        root_b = self.find(b)

        if root_a == root_b:
            return

        if self.size[root_a] < self.size[root_b]:
            root_a, root_b = root_b, root_a

        self.parent[root_b] = root_a
        self.size[root_a] += self.size[root_b]
        self.max_size = max(self.max_size, self.size[root_a])
        self.components -= 1


def connectivity_curve_for_positions(positions, radii):
    """Return LCC fraction and connected status for all radii efficiently."""
    positions = np.asarray(positions, dtype=float)
    n = len(positions)

    if n == 0:
        return np.full(len(radii), np.nan), np.zeros(len(radii), dtype=float)
    if n == 1:
        return np.ones(len(radii)), np.ones(len(radii))

    tree = cKDTree(positions)
    pairs = np.asarray(list(tree.query_pairs(float(np.max(radii)))))

    if len(pairs) == 0:
        return np.full(len(radii), 1.0 / n), np.zeros(len(radii), dtype=float)

    distances = np.linalg.norm(
        positions[pairs[:, 0]] - positions[pairs[:, 1]],
        axis=1,
    )
    order = np.argsort(distances)
    pairs = pairs[order]
    distances = distances[order]

    union_find = UnionFind(n)
    edge_index = 0
    lcc_fraction = np.empty(len(radii), dtype=float)
    connected = np.empty(len(radii), dtype=float)

    for radius_index, radius in enumerate(radii):
        while edge_index < len(distances) and distances[edge_index] <= radius:
            union_find.union(
                int(pairs[edge_index, 0]),
                int(pairs[edge_index, 1]),
            )
            edge_index += 1

        lcc_fraction[radius_index] = union_find.max_size / n
        connected[radius_index] = float(union_find.components == 1)

    return lcc_fraction, connected


def summarize_results(raw_results):
    return (
        raw_results
        .groupby(["Model", "Radius"], as_index=False)
        .agg(
            LCC_mean=("LCC fraction", "mean"),
            LCC_q025=("LCC fraction", lambda x: x.quantile(0.025)),
            LCC_q975=("LCC fraction", lambda x: x.quantile(0.975)),
            P_connected=("Connected", "mean"),
            Mean_nodes=("Nodes considered", "mean"),
            Snapshots=("Connected", "size"),
        )
    )


def first_crossing(radius, values, target):
    values = np.asarray(values, dtype=float)
    indices = np.flatnonzero(values >= target)
    return float(radius.iloc[indices[0]]) if len(indices) else np.nan


def transition_estimates(model_summary):
    model_summary = model_summary.sort_values("Radius")
    radius = model_summary["Radius"]
    lcc = model_summary["LCC_mean"].to_numpy()

    slope = np.gradient(lcc, radius.to_numpy())

    return {
        "maximum-slope radius": float(radius.iloc[np.argmax(slope)]),
        "LCC >= 0.5 radius": first_crossing(radius, lcc, 0.5),
        "P(connected) >= 0.5 radius": first_crossing(
            radius,
            model_summary["P_connected"],
            0.5,
        ),
    }


def run_percolation():
    rows,parameter_rows=[] , []
    for index,(kind,database) in enumerate([('RM','RM_MONTH_SD.db'),('SM','SM_MONTH_FINAL_V3.db')]):
        posterior,weights=read_posterior(database,kind)
        parameters={key:float(weighted_quantile(posterior[key],weights)[1]) for key in posterior.columns}
        for key in posterior.columns:
            lo,median,hi=weighted_quantile(posterior[key],weights)
            parameter_rows.append(dict(Model=kind,Database=database,Parameter=key,Q05=lo,Median=median,Q95=hi))
        cls=models.Simulation_lean if kind=='RM' else models.Simulation_scheduledmobility
        for rep in range(1,PERCOLATION_REPLICATES+1):
            seed=SEED+index*100003+rep
            random.seed(seed);np.random.seed(seed)
            sim=cls(N=498,T=0,**parameters)
            print(f"Percolation {kind}: {rep}/{PERCOLATION_REPLICATES}",flush=True)
            for t in range(BURN_IN_STEPS+OBSERVATION_STEPS):
                if kind=='RM':sim.update_positions()
                else:sim.update_positions(t)
                if t<BURN_IN_STEPS or (t-BURN_IN_STEPS)%SAMPLE_EVERY:continue
                # Use precisely the interaction-eligible SM population at this time.
                # Off-campus agents at boundary homes do not belong to the snapshot.
                nodes=list(sim.G)
                if kind=='SM':
                    nodes=[n for n in nodes if not (sim.G.nodes[n]['external'] and
                           sim.G.nodes[n]['position']==sim.G.nodes[n]['home'])]
                positions=np.asarray([sim.G.nodes[n]['position'] for n in nodes])
                lcc,connected=connectivity_curve_for_positions(positions,R_VALUES)
                for radius,largest,is_connected in zip(R_VALUES,lcc,connected):
                    rows.append(dict(Model=kind,Replicate=rep,Timestep=t,Hour=(t%288)/12,
                                     Radius=radius,**{'Nodes considered':len(nodes),'LCC fraction':largest,'Connected':is_connected}))
            pd.DataFrame(rows).to_csv(RESULTS/'percolation_snapshots.csv',index=False)
    summary=summarize_results(pd.DataFrame(rows))
    summary.to_csv(RESULTS/'percolation_summary.csv',index=False)
    pd.DataFrame(parameter_rows).to_csv(RESULTS/'percolation_parameters.csv',index=False)
    estimates=[dict(Model=kind,**transition_estimates(group)) for kind,group in summary.groupby('Model')]
    pd.DataFrame(estimates).to_csv(RESULTS/'percolation_transition_estimates.csv',index=False)


def main():
    RESULTS.mkdir(exist_ok=True)
    if NULL_REPLICATES<2 or POSTERIOR_DRAWS<1 or ATTEMPTS_PER_EDGE<1 or PERCOLATION_REPLICATES<1:
        raise ValueError('Use positive simulation counts and at least two null replicates.')
    # Fail early if a required database or the data is missing.
    if not Path('bt_symmetric.csv').is_file():raise FileNotFoundError('bt_symmetric.csv')
    if RUN_TEMPORAL:
        for _,_,database,kind in TEMPORAL_RUNS:read_posterior(database,kind)
    if RUN_PERCOLATION:
        read_posterior('RM_MONTH_SD.db','RM');read_posterior('SM_MONTH_FINAL_V3.db','SM')
    if RUN_NULLS:run_nulls()
    if RUN_TEMPORAL or RUN_PERCOLATION:
        random.seed(SEED);np.random.seed(SEED)
        models.extract_grid=lru_cache(None)(models.extract_grid)
        models.precompute_environment('Technical University of Denmark, Lyngby, Denmark',.3,.7)
    if RUN_TEMPORAL:run_temporal()
    if RUN_PERCOLATION:run_percolation()
    print('CSV files saved in results/. Run Figures.py to create figures.')


if __name__=='__main__':
    main()
