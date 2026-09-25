#!/usr/bin/env python3
"""
label_factor.py -- link between VLM labels and simulator factors (E2-style, same checkpoint).

render_concepts.py stores in meta.json the AUC of every concept's activation against every
visible ground-truth factor (all unique states). For each concept strongly aligned with one
factor (best AUC >= --min_auc), we check whether its label is closer in meaning to that
factor's statement than to the other factors' statements:

    rank        rank of the aligned factor among all factor statements by cosine to the label
    mrr         mean reciprocal rank over aligned concepts
    p           Monte-Carlo test against a random aligned factor per concept
Concepts aligned only negatively (AUC <= 1 - min_auc) are reported but not scored,
because sentence embedders handle negation poorly.

Usage:
    python label_factor.py --meta experiments/vlm/doorkey_42/meta.json \
        --labels experiments/vlm/doorkey_42/labels_qwen.json experiments/vlm/doorkey_42/labels_internvl.json \
        --env_name MiniGrid-DoorKey-6x6-v0 --out experiments/vlm/doorkey_42/label_factor.json
"""

import argparse
import json

import numpy as np

from label_agreement import EMBEDDERS, embed


def aligned_factors(cf_auc, min_auc=0.8):
    """{C: {f: auc}} -> pos {C: f}, neg {C: f}"""
    pos, neg = {}, {}
    for c, d in cf_auc.items():
        d = {f: a for f, a in d.items() if a == a}          # drop nan
        if not d:
            continue
        f_hi = max(d, key=d.get); f_lo = min(d, key=d.get)
        if d[f_hi] >= min_auc:
            pos[c] = f_hi
        elif d[f_lo] <= 1 - min_auc:
            neg[c] = f_lo
    return pos, neg


def ranks(sim, true_idx):
    """sim (C,K), true_idx (C,) -> rank (1 = most similar) of the true factor per row"""
    return np.array([1 + int((row > row[t]).sum()) for row, t in zip(sim, true_idx)])


def mc_test(sim, true_idx, n=10000, seed=0):
    """-> (mrr, p). Null: aligned factor drawn uniformly per concept"""
    rng = np.random.default_rng(seed)
    obs = float(np.mean(1 / ranks(sim, true_idx)))
    K = sim.shape[1]
    null = np.array([np.mean(1 / ranks(sim, rng.integers(K, size=len(sim)))) for _ in range(n)])
    return obs, float((1 + (null >= obs).sum()) / (1 + n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--env_name", required=True)
    ap.add_argument("--min_auc", type=float, default=0.8)
    ap.add_argument("--embedders", nargs="+", default=EMBEDDERS)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from vlm_common import factor_tests, label_to_text
    meta = json.load(open(args.meta))
    ft = factor_tests(args.env_name)
    fnames = [f for f in ft if any(f in d for d in meta["concept_factor_auc"].values())]
    stmts = [ft[f][0] for f in fnames]
    pos, neg = aligned_factors(meta["concept_factor_auc"], args.min_auc)
    print(f"[aligned] positive: {pos}\n[aligned] negative only (not scored): {neg}")

    res = dict(min_auc=args.min_auc, factors=fnames, aligned_pos=pos, aligned_neg=neg, by_namer={})
    for path in args.labels:
        L = json.load(open(path))
        cids = [c for c in pos if c in L["concepts"] and L["concepts"][c]["label"] != "unparsed"]
        if len(cids) < 2:
            print(f"[{L['namer']}] fewer than 2 aligned concepts with labels, skipped")
            continue
        texts = [label_to_text(L["concepts"][c]["label"]) for c in cids]
        t_idx = np.array([fnames.index(pos[c]) for c in cids])
        out = {}
        for e in args.embedders:
            sim = embed(e, texts) @ embed(e, stmts).T
            mrr, p = mc_test(sim, t_idx)
            out[e] = dict(mrr=mrr, p=p, chance_mrr=float(np.mean(1 / np.arange(1, len(fnames) + 1))),
                          ranks={c: int(r) for c, r in zip(cids, ranks(sim, t_idx))})
            print(f"[{L['namer']}] {e:<42} MRR={mrr:.3f} (chance {out[e]['chance_mrr']:.3f}) p={p:.4f}")
        res["by_namer"][L["namer"]] = dict(concepts={c: dict(label=L["concepts"][c]["label"], factor=pos[c])
                                                     for c in cids}, by_embedder=out)
    from vlm_common import manifest
    res["manifest"] = manifest(args)
    json.dump(res, open(args.out, "w"), indent=2)
    print(f"[done] -> {args.out}")


if __name__ == "__main__":
    main()
