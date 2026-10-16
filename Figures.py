from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

'''
5. VISUALIZING RESULTS.
This produces visualizations of the results, automatically storing them 
to the folder where this script is stored. 

'''

RESULTS = Path("results")
FIGURES = Path("figures")
LABELS = {"RM_WEEKDAY":"RM weekday (6)", "SM_WEEKDAY":"SM weekday (6)",
          "RM_MONTH":"RM month (6)", "RM_MONTH_4STAT":"RM month (4)",
          "SM_MONTH":"SM month (6)", "Data":"Data",
          "DENSITY_MATCHED":"Density null", "DEGREE_PRESERVING":"Degree null"}
COLORS = {"RM_WEEKDAY":"#0072B2", "RM_MONTH":"#6A3D9A", "RM_MONTH_4STAT":"#009E73",
          "SM_WEEKDAY":"#D55E00", "SM_MONTH":"#E69F00", "Data":"black",
          "DENSITY_MATCHED":"#CC79A7", "DEGREE_PRESERVING":"#999999"}
STATS = ["Links","Weights","CC","ASPL","Gini of degree","Gini of weights"]
SHORT = ["M","W","CC","L",r"$G_k$",r"$G_w$"]
plt.rcParams.update({"font.size":14,"axes.labelsize":15,"axes.titlesize":16,
                     "xtick.labelsize":13,"ytick.labelsize":13,"legend.fontsize":12,
                     "axes.spines.top":False,"axes.spines.right":False,"pdf.fonttype":42})


def save(fig, name):
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES/f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIGURES/f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def weighted_quantile(values, weights):
    order = np.argsort(values)
    values, weights = np.asarray(values)[order], np.asarray(weights)[order]
    positions = (np.cumsum(weights)-.5*weights)/weights.sum()
    return np.interp([.05,.5,.95], positions, values)


def plot_fits(table):
    fig, axes = plt.subplots(1,2,figsize=(12,5),layout="constrained")
    for ax, scale in zip(axes,["Weekday","Month"]):
        group = table[table.Scale==scale]
        runs = list(group.Run.unique())
        for i,run in enumerate(runs):
            sub=group[group.Run==run].set_index("Statistic").loc[STATS]
            x=np.arange(6)+(i-(len(runs)-1)/2)*.15
            y=100*sub.Median.to_numpy()/sub.Observed.to_numpy()
            lo=100*sub.Q05.to_numpy()/sub.Observed.to_numpy()
            hi=100*sub.Q95.to_numpy()/sub.Observed.to_numpy()
            ax.errorbar(x,y,yerr=[y-lo,hi-y],fmt="o",capsize=3,color=COLORS[run],label=LABELS[run])
        ax.axhline(100,color="black",ls="--")
        ax.set(xticks=np.arange(6),xticklabels=SHORT,title=scale,ylabel="Simulated / observed (%)")
        ax.legend(loc="upper center",bbox_to_anchor=(.5,-.12),ncol=1,frameon=False)
        ax.grid(axis="y",alpha=.2)
    save(fig,"model_fits")


def plot_parameters(particles, summary):
    # Raw weighted particles show r/s dependence without unstable KDE contours.
    fig, axes = plt.subplots(1,2,figsize=(12,7),layout="constrained")
    for ax,model in zip(axes,["RM","SM"]):
        for run,sub in particles[particles.Run.str.startswith(model)].groupby("Run",sort=False):
            sizes=15+150*sub.Posterior_weight/sub.Posterior_weight.max()
            ax.scatter(sub.r,sub.s,s=sizes,alpha=.3,color=COLORS[run],edgecolors="none",label=LABELS[run])
        ax.set(title=model,xlabel="Interaction radius r (m)",ylabel="Step distance s (m)")
        ax.legend(loc="upper center",bbox_to_anchor=(.5,-.15),frameon=False)
    sub=summary[(summary.Run.str.startswith("SM")) & summary.Parameter.isin(["PW","WM"])]
    rows=[]
    for run in sub.Run.unique():
        row=[LABELS[run]]
        for parameter in ["PW","WM"]:
            q=sub[(sub.Run==run)&(sub.Parameter==parameter)].iloc[0]
            row.append(f"{q.Median:.2f} [{q.Q05:.2f}, {q.Q95:.2f}]")
        rows.append(row)
    # Separate table area below panels keeps labels and legends readable.
    fig.set_layout_engine(None)
    fig.subplots_adjust(bottom=.40,wspace=.28)
    tax=fig.add_axes([.08,.03,.84,.17]);tax.axis("off")
    table=tax.table(cellText=rows,colLabels=["SM fit","PW: median [5–95%]","WM (steps): median [5–95%]"],loc="center",cellLoc="center")
    table.auto_set_font_size(False);table.set_fontsize(12);table.scale(1,1.7)
    save(fig,"parameter_posteriors")


def plot_distributions(model, empirical):
    fig,axes=plt.subplots(2,2,figsize=(12,9),layout="constrained")
    for i,scale in enumerate(["Weekday","Month"]):
        for j,statistic in enumerate(["Degree","Weight"]):
            ax=axes[i,j]
            selected=pd.concat([model,empirical],ignore_index=True)
            selected=selected[(selected.Scale==scale)&(selected.Distribution==statistic)]
            for run,group in selected.groupby("Run",sort=False):
                maximum=int(group.Value.max())
                x=(np.arange(maximum+1) if statistic=="Degree" else
                   np.unique(np.geomspace(1,maximum,200).astype(int)))
                curves=[];weights=[]
                for _,part in group.groupby("Particle"):
                    values=part.Value.to_numpy();counts=part.Count.to_numpy()
                    curves.append(np.array([counts[values>=v].sum()/counts.sum() for v in x]))
                    weights.append(part.Posterior_weight.iloc[0])
                curves=np.asarray(curves)
                q=np.array([weighted_quantile(curves[:,k],weights) for k in range(len(x))])
                ax.plot(x,q[:,1],color=COLORS[run],label=LABELS[run])
                if run!="Data":ax.fill_between(x,q[:,0],q[:,2],color=COLORS[run],alpha=.12)
            if statistic=="Weight":ax.set(xscale="log",yscale="log",ylim=(1e-5,1.05))
            ax.set(title=f"{scale}: {statistic.lower()}",xlabel=statistic,ylabel="Fraction ≥ value")
            ax.legend(loc="upper center",bbox_to_anchor=(.5,-.18),frameon=False)
    save(fig,"degree_weight_distributions")


def plot_additional():
    path=RESULTS/"null_summary.csv"
    if path.exists():
        table=pd.read_csv(path)
        fitted=RESULTS/"posterior_summary.csv"
        if fitted.exists():table=pd.concat([table,pd.read_csv(fitted)],ignore_index=True)
        fig,axes=plt.subplots(2,2,figsize=(12,9),layout="constrained")
        for i,scale in enumerate(["Weekday","Month"]):
            for j,stat in enumerate(["CC","ASPL"]):
                ax=axes[i,j];sub=table[(table.Scale==scale)&(table.Statistic==stat)]
                for x,row in enumerate(sub.itertuples()):
                    ax.errorbar(x,row.Median,yerr=[[row.Median-row.Q05],[row.Q95-row.Median]],fmt="o",capsize=4,color=COLORS[row.Run])
                ax.axhline(sub.Observed.iloc[0],color="black",ls="--")
                ax.set(xticks=range(len(sub)),xticklabels=[LABELS[r].replace(" ","\n",1) for r in sub.Run],title=f"{scale}: {stat}",ylabel=stat)
        save(fig,"null_comparison")
    path=RESULTS/"window_mean_summary.csv"
    if path.exists():
        table=pd.read_csv(path)
        for scale in ["Weekday","Month"]:
            sub=table[(table.Scale==scale)&(table.Calendar==("Weekdays" if scale=="Weekday" else "All days"))]
            fig,axes=plt.subplots(2,3,figsize=(14,9),layout="constrained")
            for ax,stat in zip(axes.flat,STATS):
                for run,g in sub[sub.Statistic==stat].groupby("Run",sort=False):
                    g=g.sort_values("Days")
                    ax.plot(g.Days,g.Median,color=COLORS[run],label=LABELS[run])
                    if run!="Data":ax.fill_between(g.Days,g.Q25,g.Q75,color=COLORS[run],alpha=.2)
                ax.set(xscale="log",title=stat,xlabel="Window duration (days)")
            handles,labels=axes.flat[0].get_legend_handles_labels()
            fig.legend(handles,labels,loc="outside lower center",ncol=3,frameon=False)
            save(fig,f"temporal_{scale.lower()}")
    path=RESULTS/"paired_ratio_summary.csv"
    if path.exists():
        table=pd.read_csv(path)
        table=table[table.Statistic.isin(STATS) & (table.Run!="Data")]
        fig,axes=plt.subplots(1,2,figsize=(12,5),layout="constrained")
        for ax,(run,group) in zip(axes,table.groupby("Run",sort=False)):
            group=group.set_index("Statistic").loc[STATS]
            median=group.Median.to_numpy();low=group.Q05.to_numpy();high=group.Q95.to_numpy()
            ax.errorbar(range(6),median,yerr=[median-low,high-median],fmt="o",capsize=4,color=COLORS[run],label=LABELS[run])
            ax.scatter(range(6),group.Empirical_ratio,color="black",marker="x",label="Data")
            ax.set(xticks=range(6),xticklabels=SHORT,ylabel="Month / mean weekday",title=LABELS[run])
            ax.legend(frameon=False)
        save(fig,"month_weekday_ratios")
    path=RESULTS/"percolation_summary.csv"
    if path.exists():
        table=pd.read_csv(path)
        fig,axes=plt.subplots(1,2,figsize=(12,5),layout="constrained")
        for model,g in table.groupby("Model"):
            color=COLORS["RM_MONTH" if model=="RM" else "SM_MONTH"]
            axes[0].plot(g.Radius,g.LCC_mean,color=color,label=model)
            axes[0].fill_between(g.Radius,g.LCC_q025,g.LCC_q975,color=color,alpha=.15)
            axes[1].plot(g.Radius,g.P_connected,color=color,label=model)
        references=RESULTS/"percolation_parameters.csv"
        if references.exists():
            parameters=pd.read_csv(references)
            for row in parameters[parameters.Parameter=="r"].itertuples():
                color=COLORS["RM_MONTH" if row.Model=="RM" else "SM_MONTH"]
                for ax in axes:
                    ax.axvspan(row.Q05,row.Q95,color=color,alpha=.08)
                    ax.axvline(row.Median,color=color,ls="--",lw=1.5)
        for ax,label in zip(axes,["Largest component / eligible nodes","Fraction of connected snapshots"]):
            ax.set(xlabel="Interaction radius (m)",ylabel=label,ylim=(0,1.02));ax.legend(frameon=False)
        save(fig,"instantaneous_percolation")


def main():
    if (RESULTS/"posterior_summary.csv").exists():
        plot_fits(pd.read_csv(RESULTS/"posterior_summary.csv"))
        plot_parameters(pd.read_csv(RESULTS/"posterior_particles.csv"),pd.read_csv(RESULTS/"parameter_summary.csv"))
        plot_distributions(pd.read_csv(RESULTS/"distributions.csv"),pd.read_csv(RESULTS/"empirical_distributions.csv"))
    plot_additional()
    print("Figures saved in figures/")


if __name__=="__main__":
    main()
