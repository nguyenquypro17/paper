"""
summarize_same_split.py
Bảng LUCID vs control trên CÙNG tập held-out, mean ± std qua 3 seed.

Với A1 (CART) và A2 (L1 linear) có sweep knob, báo cáo hai con số reviewer quan tâm:
  fid@budget  : fidelity tốt nhất khi complexity <= số literal của LUCID (cùng seed)
  lit→match   : số literal nhỏ nhất để đạt fidelity của LUCID (cùng seed)

  python summarize_same_split.py      -> same_split_summary.md
"""
import glob
import json
import os
from collections import defaultdict

import numpy as np

from controls.complexity import count_dnf

ENVS = ["doorkey", "dyobs", "cartpole", "pong", "boxing"]
SEEDS = [42, 43, 44]
RES = "results/controls_same"


def ms(xs, fmt="{:.2f}"):
    xs = [x for x in xs if x is not None and not np.isnan(x)]
    if not xs:
        return "—"
    return (fmt + " ± " + fmt).format(np.mean(xs), np.std(xs))


def load_lucid(tag, s):
    d = f"outputs/lucid_model_{tag}_{s}"
    sc = os.path.join(d, "scoring_check.json")
    rl = os.path.join(d, "learned_rules.json")
    if not (os.path.isfile(sc) and os.path.isfile(rl)):
        return None
    r = json.load(open(sc))
    cx = count_dnf(json.load(open(rl)))
    return {"fid": r["fidelity_weighted_pct"], "cov": r["coverage_pct"],
            "lit": cx["literals"], "conc": cx["concepts"]}


def load_controls():
    by = defaultdict(list)          # (env_tag, arm, seed) -> [records]
    for f in glob.glob(os.path.join(RES, "*.json")):
        r = json.load(open(f))
        tag = os.path.basename(r.get("lucid_split_dir", "")).replace("lucid_model_", "").rsplit("_", 1)[0]
        by[(tag, r["arm"], r["seed"])].append(r)
    return by


def main():
    ctrl = load_controls()
    lines = ["# LUCID vs control — cùng tập held-out (mean ± std, 3 seed)\n",
             "Held-out = `shuffle_indices[n_train:]` của từng run LUCID. "
             "Control chọn model trên val tách từ train; LUCID chọn checkpoint trên held-out "
             "→ protocol của control chặt hơn, gap là ước lượng bảo thủ.\n"]

    for tag in ENVS:
        luc = {s: load_lucid(tag, s) for s in SEEDS}
        if not any(luc.values()):
            continue
        lines.append(f"\n## {tag}\n")
        lines.append("| Method | Fidelity (hard, logic thuần) | Soft | Literals | Concepts | Coverage |")
        lines.append("|---|---|---|---|---|---|")
        L = [v for v in luc.values() if v]
        lines.append(f"| **LUCID (joint)** | {ms([v['fid'] for v in L])} | — | "
                     f"{ms([v['lit'] for v in L], '{:.1f}')} | {ms([v['conc'] for v in L], '{:.1f}')} | "
                     f"{ms([v['cov'] for v in L], '{:.1f}')} |")

        for arm, name in (("A4", "Two-stage (frozen SAE + DNF)"), ("A3", "DNF, no SAE")):
            rs = [r for s in SEEDS for r in ctrl.get((tag, arm, s), [])]
            if not rs:
                continue
            lines.append(f"| {name} | {ms([r.get('fidelity_hard_pure') for r in rs])} | "
                         f"{ms([r['fidelity_soft'] for r in rs])} | "
                         f"{ms([r['complexity']['literals'] for r in rs], '{:.1f}')} | "
                         f"{ms([r['complexity']['concepts'] for r in rs], '{:.1f}')} | "
                         f"{ms([r.get('coverage_pure') for r in rs], '{:.1f}')} |")

        lines.append("\n| Method | fid@budget (≤ LUCID literals) | literals→match LUCID fid | best fid (any size) |")
        lines.append("|---|---|---|---|")
        for arm, name in (("A1", "SAE + CART"), ("A2", "SAE + L1 linear")):
            at_budget, to_match, best = [], [], []
            for s in SEEDS:
                rs = ctrl.get((tag, arm, s), [])
                if not rs or not luc[s]:
                    continue
                pts = [(r["complexity"]["literals"], r["fidelity_hard"]) for r in rs]
                under = [f for l, f in pts if l <= luc[s]["lit"]]
                at_budget.append(max(under) if under else np.nan)
                reach = [l for l, f in pts if f >= luc[s]["fid"]]
                to_match.append(min(reach) if reach else np.nan)
                best.append(max(f for _, f in pts))
            if not best:
                continue
            miss = sum(np.isnan(x) for x in to_match)
            tm = ms(to_match, "{:.1f}") + (f" ({miss}/3 seed không đạt)" if miss else "")
            lines.append(f"| {name} | {ms(at_budget)} | {tm} | {ms(best)} |")

    open("same_split_summary.md", "w").write("\n".join(lines) + "\n")
    print("-> same_split_summary.md")


if __name__ == "__main__":
    main()
