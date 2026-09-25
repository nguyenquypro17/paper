"""
controls/pareto.py
Fidelity-vs-complexity Pareto curves.

Each method gets its own knob that trades complexity for accuracy:
    LUCID / DNF arms   : l0_penalty x n_clauses_per_action
    CART arms          : ccp_alpha (cost-complexity pruning)
    sparse linear      : C (inverse L1 strength)

`emit_commands` prints the exact run_controls.py invocations; `collect` reads
the resulting JSONs back; `plot_pareto` draws complexity (x, log) against the
chosen metric (y), one line per arm, averaged over training seeds.
"""
import glob
import json
import os
from collections import defaultdict
from typing import Dict, List

import numpy as np

SWEEPS = {
    "A0": [{"l0_penalty": l0, "n_clauses_per_action": c}
           for l0 in (1e-5, 1e-4, 1e-3, 5e-3, 1e-2) for c in (2, 5, 10)],
    "A3": [{"l0_penalty": l0, "n_clauses_per_action": c}
           for l0 in (1e-5, 1e-4, 1e-3, 5e-3, 1e-2) for c in (2, 5, 10)],
    "A4": [{"l0_penalty": l0, "n_clauses_per_action": c}
           for l0 in (1e-5, 1e-4, 1e-3, 5e-3, 1e-2) for c in (2, 5, 10)],
    "A1": [{"ccp_alpha": a} for a in (0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2)],
    "A5": [{"ccp_alpha": a} for a in (0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2)],
    "A2": [{"C": c} for c in (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)],
}

ARM_LABELS = {
    "A0": "LUCID (SAE joint + DNF)",
    "A1": "SAE + CART",
    "A2": "SAE + L1 linear",
    "A3": "DNF, no SAE",
    "A4": "Two-stage (frozen SAE + DNF)",
    "A5": "CART on raw CNN",
}


def _knob_tag(knob: Dict) -> str:
    return "_".join(f"{k}{v}" for k, v in sorted(knob.items()))


def emit_commands(arms: List[str], env_name: str, features_path: str, ppo_path: str,
                  seeds=(42, 43, 44), out="results/pareto", episodes: int = 100,
                  extra: str = "") -> List[str]:
    """Generate the shell commands for a full sweep. Pipe into `parallel` or bash."""
    cmds = []
    for arm in arms:
        for knob in SWEEPS[arm]:
            flags = " ".join(f"--{k} {v}" for k, v in knob.items())
            for seed in seeds:
                cmds.append(
                    f"python run_controls.py --arm {arm} --env_name {env_name} "
                    f"--features_path {features_path} --ppo_path {ppo_path} "
                    f"--seed {seed} --episodes {episodes} --out {out} "
                    f"--tag {_knob_tag(knob)} {flags} {extra}".strip()
                )
    return cmds


def collect(results_dir: str) -> List[Dict]:
    """Flatten every JSON in a results dir into plot-ready records."""
    records = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        with open(path) as f:
            r = json.load(f)
        online = r.get("online_hard") or {}
        records.append({
            "arm": r["arm"],
            "env": r["env_name"],
            "seed": r["seed"],
            "literals": r["complexity"]["literals"],
            "concepts": r["complexity"]["concepts"],
            "fidelity": r.get("fidelity_hard"),
            "score": (online.get("score") or {}).get("mean"),
            "success_rate": (online.get("success_rate") or {}).get("mean"),
            "knob": r.get("knob", {}),
        })
    return records


def aggregate(records: List[Dict], metric: str = "score") -> Dict[str, List[Dict]]:
    """Group by (arm, knob) and average over seeds; unit of std is the seed."""
    buckets = defaultdict(list)
    for r in records:
        buckets[(r["arm"], json.dumps(r["knob"], sort_keys=True))].append(r)

    out = defaultdict(list)
    for (arm, _), rows in buckets.items():
        ys = [r[metric] for r in rows if r.get(metric) is not None]
        if not ys:
            continue
        out[arm].append({
            "literals": float(np.mean([r["literals"] for r in rows])),
            "mean": float(np.mean(ys)),
            "std": float(np.std(ys)),
            "n_seeds": len(ys),
        })
    for arm in out:
        out[arm].sort(key=lambda d: d["literals"])
    return dict(out)


def pareto_front(points: List[Dict]) -> List[Dict]:
    """Keep points not dominated on (low literals, high metric)."""
    front, best = [], -np.inf
    for p in sorted(points, key=lambda d: d["literals"]):
        if p["mean"] > best:
            front.append(p)
            best = p["mean"]
    return front


def plot_pareto(records: List[Dict], out_png: str, metric: str = "score",
                title: str = "", front_only: bool = False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    agg = aggregate(records, metric=metric)
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    for arm in sorted(agg):
        pts = pareto_front(agg[arm]) if front_only else agg[arm]
        x = [p["literals"] for p in pts]
        y = [p["mean"] for p in pts]
        e = [p["std"] for p in pts]
        ax.errorbar(x, y, yerr=e, marker="o", capsize=2, linewidth=1.4,
                    markersize=4, label=ARM_LABELS.get(arm, arm))

    ax.set_xscale("log")
    ax.set_xlabel("Complexity (feature tests / literals, log scale)")
    ax.set_ylabel({"score": "Online return", "success_rate": "Success rate (%)",
                   "fidelity": "Action fidelity (%)"}.get(metric, metric))
    if title:
        ax.set_title(title)
    ax.grid(alpha=0.3, linestyle=":")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    return out_png


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Pareto sweep helper")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("emit")
    g.add_argument("--arms", nargs="+", default=list(SWEEPS))
    g.add_argument("--env_name", required=True)
    g.add_argument("--features_path", required=True)
    g.add_argument("--ppo_path", required=True)
    g.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    g.add_argument("--episodes", type=int, default=100)
    g.add_argument("--out", default="results/pareto")

    h = sub.add_parser("plot")
    h.add_argument("--results_dir", required=True)
    h.add_argument("--out_png", default="results/pareto/pareto.png")
    h.add_argument("--metric", default="score", choices=["score", "success_rate", "fidelity"])
    h.add_argument("--title", default="")
    h.add_argument("--front_only", action="store_true")

    a = p.parse_args()
    if a.cmd == "emit":
        for c in emit_commands(a.arms, a.env_name, a.features_path, a.ppo_path,
                               seeds=tuple(a.seeds), out=a.out, episodes=a.episodes):
            print(c)
    else:
        print(plot_pareto(collect(a.results_dir), a.out_png, metric=a.metric,
                          title=a.title, front_only=a.front_only))
