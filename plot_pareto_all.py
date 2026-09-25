import glob, json, os
from collections import defaultdict
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter
from controls.complexity import count_dnf

ORDER = [("dyobs","Dynamic-Obstacles"),("doorkey","DoorKey-6x6"),
         ("cartpole","PixelCartPole"),("pong","Pong"),("boxing","Boxing")]
STY = {"A1":dict(c="#1f77b4",m="o",l="SAE + CART"),
       "A2":dict(c="#ff7f0e",m="o",l="SAE + L1 linear"),
       "A4":dict(c="#7f7f7f",m="s",l="Two-stage (frozen SAE + DNF)"),
       "A3":dict(c="#2ca02c",m="D",l="DNF, no SAE")}
plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7.5})

recs = defaultdict(lambda: defaultdict(list)); luc = defaultdict(list)
for f in glob.glob("results/controls_same/*.json"):
    r = json.load(open(f))
    env = os.path.basename(r["lucid_split_dir"]).replace("lucid_model_","").rsplit("_",1)[0]
    lit = r["complexity"]["literals"]
    if lit < 1: continue
    recs[env][r["arm"]].append((lit, r.get("fidelity_hard_pure", r["fidelity_hard"]),
                                json.dumps(r.get("knob",{}),sort_keys=True)))
for d in glob.glob("outputs/lucid_model_*_4[234]"):
    env = os.path.basename(d).replace("lucid_model_","").rsplit("_",1)[0]
    if os.path.isfile(f"{d}/scoring_check.json"):
        luc[env].append((count_dnf(json.load(open(f"{d}/learned_rules.json")))["literals"],
                         json.load(open(f"{d}/scoring_check.json"))["fidelity_weighted_pct"]))

fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.4))
axes = axes.ravel()
for ax, (env, title) in zip(axes, ORDER):
    for arm in ("A1","A2","A4","A3"):
        by = defaultdict(list)
        for lit, fid, knob in recs[env].get(arm, []): by[knob].append((lit, fid))
        pts = sorted((np.mean([p[0] for p in v]), np.mean([p[1] for p in v]),
                      np.std([p[1] for p in v])) for v in by.values())
        if not pts: continue
        s = STY[arm]; x=[p[0] for p in pts]; y=[p[1] for p in pts]; e=[p[2] for p in pts]
        if arm in ("A1","A2"):
            ax.errorbar(x,y,yerr=e,color=s["c"],marker=s["m"],ms=2.8,lw=1.1,capsize=1.5,label=s["l"])
        else:
            ax.errorbar(x,y,yerr=e,color=s["c"],marker=s["m"],ms=4.5,ls="none",capsize=1.5,label=s["l"])
    if luc.get(env):
        L = np.array(luc[env])
        ax.errorbar([L[:,0].mean()],[L[:,1].mean()],xerr=[L[:,0].std()],yerr=[L[:,1].std()],
                    color="crimson",marker="*",ms=11,capsize=2,lw=1.1,label="LUCID (joint)",zorder=5)
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(LogLocator(base=10))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_title(title); ax.grid(alpha=.3, ls=":")
for i in (0, 3): axes[i].set_ylabel("Hard fidelity (%)")
for i in (3, 4): axes[i].set_xlabel("Feature tests (log)")
axes[2].set_xlabel("Feature tests (log)")
h, l = axes[3].get_legend_handles_labels()
axes[5].axis("off")
axes[5].legend(h, l, loc="center", frameon=False)
fig.tight_layout(pad=0.4, w_pad=0.6, h_pad=0.8)
fig.savefig("figs/pareto_all.pdf", bbox_inches="tight")
fig.savefig("figs/pareto_all.png", dpi=250, bbox_inches="tight")
print("figs/pareto_all.{pdf,png}")
