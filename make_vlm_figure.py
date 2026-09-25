#!/usr/bin/env python3
"""
make_vlm_figure.py -- figure for the paper: the exact ON / OFF views the namer saw, with the label
it proposed and the blind-test AUC. No API calls.

Usage (on the server, in ~/XRL):
    python make_vlm_figure.py --out vlm_examples.pdf \
        --item experiments/vlm/doorkey_42 C131 labels_gemini31_a.json predict_deepseek.json \
        --item experiments/vlm/dyobs_42   C49  labels_gpt55_a.json     predict_deepseek-flash_gpt55.json
Each --item: <env dir> <concept id> <labels file> <predict file> (files relative to the env dir).
"""

import argparse
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ENV_NAME = {"doorkey": "DoorKey-6x6", "dyobs": "Dynamic-Obstacles-5x5", "cartpole": "PixelCartPole",
            "pong": "Pong", "boxing": "Boxing"}


def load_item(env_dir, cid, labels_file, predict_file, n):
    ndir = os.path.join(env_dir, "naming", cid)
    on = sorted(glob.glob(os.path.join(ndir, "on_*.png")))[:n]
    off = sorted(glob.glob(os.path.join(ndir, "off_*.png")))[:n]
    L = json.load(open(os.path.join(env_dir, labels_file)))
    P = json.load(open(os.path.join(env_dir, predict_file)))
    r = P["by_namer"][L["namer"]]["concepts"][cid]
    env = os.path.basename(os.path.normpath(env_dir)).rsplit("_", 1)[0]
    nm = L["namer"].lower()
    namer = ("Gemini 3.1 Pro" if nm.startswith("gemini-3.1") else "GPT-5.5" if nm.startswith("gpt-5.5")
             else L["namer"])
    return dict(on=on, off=off, label=L["concepts"][cid]["label"].replace("_", " "),
                namer=namer, auc=r["auc_true"], distr=float(np.mean(list(r["auc_distractors"].values()))),
                env=ENV_NAME.get(env, env), cid=cid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--item", nargs=4, action="append", required=True,
                    metavar=("ENV_DIR", "CONCEPT", "LABELS", "PREDICT"))
    ap.add_argument("--n", type=int, default=4, help="ON and OFF examples per concept")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    items = [load_item(*it, args.n) for it in args.item]
    n = args.n
    fig, axes = plt.subplots(len(items), 2 * n + 1, figsize=(1.35 * 2 * n + 0.6, 1.9 * len(items)),
                             squeeze=False, gridspec_kw=dict(wspace=0.06, hspace=0.75,
                                                             width_ratios=[1] * n + [0.18] + [1] * n))
    for r, it in enumerate(items):
        axes[r, n].axis("off")                          # gap between ON and OFF groups
        cols = list(range(n)) + list(range(n + 1, 2 * n + 1))
        for j, col in enumerate(cols):
            ax = axes[r, col]
            ax.set_xticks([]); ax.set_yticks([])
            is_on = j < n
            paths = it["on"] if is_on else it["off"]
            k = j if is_on else j - n
            if k < len(paths):
                ax.imshow(Image.open(paths[k]).convert("RGB"))
            for sp in ax.spines.values():
                sp.set_edgecolor("#1b7837" if is_on else "#b2182b"); sp.set_linewidth(1.6)
        for grp, name, colr in [(range(n), "concept ON", "#1b7837"),
                                (range(n + 1, 2 * n + 1), "concept OFF", "#b2182b")]:
            a, b = axes[r, grp[0]].get_position(), axes[r, grp[-1]].get_position()
            fig.text((a.x0 + b.x1) / 2, a.y1 + 0.006, name, ha="center", va="bottom", fontsize=7, color=colr)
        num = it["cid"][1:]
        title = (f"{it['env']}, $f_{{{num}}}$: “{it['label']}” ({it['namer']})   "
                 f"blind AUC {it['auc']:.2f} vs. distractors {it['distr']:.2f}")
        fig.text(0.5, axes[r, 0].get_position().y1 + 0.045, title, ha="center", va="bottom", fontsize=8)
    fig.savefig(args.out, bbox_inches="tight", dpi=300)
    print(f"[done] -> {args.out}")


if __name__ == "__main__":
    main()
