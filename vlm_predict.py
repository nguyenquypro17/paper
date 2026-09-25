#!/usr/bin/env python3
"""
vlm_predict.py -- Step 2: blind predictive test of the labels.

For each concept c and each test image of c (never seen during naming, no heatmap),
a scorer VLM of a different family is asked whether the label holds in that state.
P(Yes) is compared with the true hard-concept state y.

    auc_true        AUC of P(Yes | true label of c) against y
    auc_distractor  same with labels of OTHER concepts of the same env (same namer)
    p_perm          permutation test: per concept, the "true" label is drawn at random
                    from {true} + distractors; statistic = mean(auc_true - mean auc_distr)
    factor_control  positive control: AUC of the scorer on ground-truth statements
                    (test/_factors/*), i.e. the ceiling a correct label could reach

Usage:
    python vlm_predict.py --test_dir experiments/vlm/doorkey_42/test \
        --labels experiments/vlm/doorkey_42/labels_qwen.json \
                 experiments/vlm/doorkey_42/labels_internvl.json \
        --env_name MiniGrid-DoorKey-6x6-v0 --model gemma --load_4bit \
        --out experiments/vlm/doorkey_42/predict_gemma.json
"""

import argparse
import glob
import json
import os

import numpy as np
from tqdm import tqdm

HEAD = """ENVIRONMENT: {env}

The image shows one state:"""
QUESTION = """Statement: "{label}"

Is the statement true in this state? Answer with Yes or No only."""


# ============================================================================
# Statistics (numpy only)
# ============================================================================

def auc(y, p):
    """ROC AUC via ranks (ties = 0.5); nan if one class is missing"""
    y, p = np.asarray(y), np.asarray(p, float)
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0) + 0.5 * (diff == 0)).mean())


def balanced_acc(y, p, thr=0.5):
    y, pred = np.asarray(y), np.asarray(p) > thr
    return float(0.5 * (pred[y == 1].mean() + (~pred[y == 0]).mean()))


def perm_test(auc_true, auc_distr, n_perm=10000, seed=0):
    """
    auc_true (C,), auc_distr list of (n_c,) arrays.
    -> (stat, p). Null: for each concept the true label is a uniformly random member
    of {true} + distractors.
    """
    rng = np.random.default_rng(seed)
    pools = [np.concatenate([[t], d]) for t, d in zip(auc_true, auc_distr)]

    def stat(pick):
        return float(np.mean([pl[k] - np.delete(pl, k).mean() for pl, k in zip(pools, pick)]))

    obs = stat([0] * len(pools))
    null = np.array([stat([rng.integers(len(pl)) for pl in pools]) for _ in range(n_perm)])
    return obs, float((1 + (null >= obs).sum()) / (1 + n_perm))


def choose_distractors(labels, c, n, seed=0):
    """labels {cid: label} -> up to n labels of other concepts, textually different from c's"""
    rng = np.random.default_rng(seed)
    others = sorted({v for k, v in labels.items() if k != c and v != labels[c] and v != "unparsed"})
    if len(others) <= n:
        return others
    return [others[i] for i in rng.choice(len(others), n, replace=False)]


# ============================================================================
# Scoring
# ============================================================================

def predict_active(model, proc, label, image_path, env_name, cache):
    """label: snake_case label or a full sentence -> (p_yes, mass); cached by (label, image)"""
    from vlm_common import yes_prob, env_context, label_to_text
    key = (label, image_path)
    if key not in cache:
        content = [HEAD.format(env=env_context(env_name)), image_path,
                   QUESTION.format(label=label_to_text(label))]
        cache[key] = yes_prob(model, proc, content)
    return cache[key]


def prefetch(model, proc, pairs, env_name, cache, workers=8):
    """fill cache for [(label, image_path)] -- threaded for API scorers, serial otherwise.
    Pairs cached with mass 0 (API failure / unparsable answer) are retried."""
    todo = [pr for pr in dict.fromkeys(pairs) if pr not in cache or cache[pr][1] == 0.0]
    for pr in todo:
        cache.pop(pr, None)
    if not todo:
        return
    from vlm_common import is_api
    if not is_api(model):
        for lab, im in tqdm(todo, desc="Scoring"):
            predict_active(model, proc, lab, im, env_name, cache)
        return
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(workers) as ex:
        list(tqdm(ex.map(lambda pr: predict_active(model, proc, pr[0], pr[1], env_name, cache), todo),
                  total=len(todo), desc="Scoring (API)"))


def factor_pairs(test_dir):
    out = []
    for d in sorted(glob.glob(os.path.join(test_dir, "_factors", "*"))):
        tg = json.load(open(os.path.join(d, "targets.json")))
        out += [(tg["statement"], os.path.join(d, f)) for f in tg["images"]]
    return out


def label_pairs(labels, test_dir, n_distr, seed):
    """-> [(label, image)] for true labels and distractors, same choice as evaluate()"""
    out = []
    for c in labels:
        tdir = os.path.join(test_dir, c)
        if labels[c] == "unparsed" or not os.path.exists(os.path.join(tdir, "targets.json")):
            continue
        imgs = [os.path.join(tdir, f) for f in json.load(open(os.path.join(tdir, "targets.json")))["images"]]
        for lab in [labels[c]] + choose_distractors(labels, c, n_distr, seed + int(c[1:])):
            out += [(lab, im) for im in imgs]
    return out


def predict_factor(model, proc, test_dir, env_name, cache):
    """positive control -> {factor: dict(statement, n, auc, bacc)} from test/_factors/*"""
    out = {}
    for d in sorted(glob.glob(os.path.join(test_dir, "_factors", "*"))):
        tg = json.load(open(os.path.join(d, "targets.json")))
        p = np.array([predict_active(model, proc, tg["statement"], os.path.join(d, f), env_name, cache)[0]
                      for f in tg["images"]])
        out[tg["factor"]] = dict(statement=tg["statement"], n=len(p),
                                 auc=auc(tg["y"], p), bacc=balanced_acc(tg["y"], p))
    return out


def evaluate(model, proc, labels, test_dir, env_name, n_distr=3, n_perm=10000, seed=0, cache=None):
    """labels {cid: label} of ONE namer -> dict per concept + summary"""
    cache = {} if cache is None else cache
    per, masses = {}, []
    cids = [c for c in sorted(labels, key=lambda s: int(s[1:]))
            if os.path.exists(os.path.join(test_dir, c, "targets.json")) and labels[c] != "unparsed"]
    for c in tqdm(cids, desc="Concepts"):
        tg = json.load(open(os.path.join(test_dir, c, "targets.json")))
        imgs = [os.path.join(test_dir, c, f) for f in tg["images"]]
        y = np.array(tg["y"])
        distr = choose_distractors(labels, c, n_distr, seed + int(c[1:]))

        def run(lab):
            out = [predict_active(model, proc, lab, im, env_name, cache) for im in imgs]
            masses.extend(m for _, m in out)
            return np.array([p for p, _ in out])

        p_true = run(labels[c])
        per[c] = dict(label=labels[c], n=int(len(y)),
                      auc_true=auc(y, p_true), bacc_true=balanced_acc(y, p_true),
                      auc_distractors={d: auc(y, run(d)) for d in distr})

    ok = [c for c in per if per[c]["auc_distractors"] and not np.isnan(per[c]["auc_true"])]
    a_t = np.array([per[c]["auc_true"] for c in ok])
    a_d = [np.array(list(per[c]["auc_distractors"].values())) for c in ok]
    stat, p = perm_test(a_t, a_d, n_perm, seed) if ok else (float("nan"), float("nan"))
    summary = dict(n_concepts=len(ok),
                   mean_auc_true=float(a_t.mean()) if ok else float("nan"),
                   mean_auc_distractor=float(np.mean([d.mean() for d in a_d])) if ok else float("nan"),
                   mean_bacc_true=float(np.mean([per[c]["bacc_true"] for c in ok])) if ok else float("nan"),
                   delta=stat, p_perm=p,
                   low_mass_frac=float(np.mean(np.array(masses) < 0.5)) if masses else float("nan"))
    return dict(concepts=per, summary=summary)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_dir", required=True, help="<render out_dir>/test")
    ap.add_argument("--labels", nargs="+", required=True, help="outputs of vlm_name.py")
    ap.add_argument("--env_name", required=True)
    ap.add_argument("--model", required=True, help="scorer; must differ from every namer")
    ap.add_argument("--load_4bit", action="store_true")
    ap.add_argument("--n_distractors", type=int, default=3)
    ap.add_argument("--n_perm", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8, help="parallel requests (API scorers)")
    ap.add_argument("--thinking_budget", type=int, default=128,
                    help="Gemini 2.5 scorer (>=1024 -> thinking_level HIGH on Gemini 3); use a NEW --cache when changing it")
    ap.add_argument("--cache", default=None, help="json file: reuse scores across runs (API cost)")
    ap.add_argument("--only_control", action="store_true", help="positive control only")
    ap.add_argument("--allow_same_model", action="store_true",
                    help="scorer may equal the namer (separate blind calls); recorded in the output")
    ap.add_argument("--budget_usd", type=float, default=None, help="API spend cap, shared across runs")
    ap.add_argument("--price_in", type=float, default=None, help="USD per 1M input tokens")
    ap.add_argument("--price_out", type=float, default=None, help="USD per 1M output tokens")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from vlm_common import load_vlm
    model, proc = load_vlm(args.model, args.load_4bit, args.thinking_budget)
    from vlm_common import attach_budget
    attach_budget(model, args.budget_usd, args.price_in, args.price_out)
    cache = {}
    if args.cache and os.path.exists(args.cache):
        cache = {tuple(k.split("||", 1)): tuple(v) for k, v in json.load(open(args.cache)).items()}
    labels_all = {path: {c: v["label"] for c, v in json.load(open(path))["concepts"].items()}
                  for path in args.labels}
    pairs = factor_pairs(args.test_dir)
    if not args.only_control:
        for L in labels_all.values():
            pairs += label_pairs(L, args.test_dir, args.n_distractors, args.seed)
    try:
        prefetch(model, proc, pairs, args.env_name, cache, args.workers)
    finally:
        if args.cache:
            json.dump({"||".join(k): list(v) for k, v in cache.items()}, open(args.cache, "w"))
    n_fail = sum(cache[pr][1] == 0.0 for pr in dict.fromkeys(pairs) if pr in cache)
    print(f"[scoring] {len(set(pairs))} pairs, failed/unparsable = {n_fail}  "
          f"errors = {getattr(model, 'errors', {})}")
    if n_fail:
        print("[warning] re-run the same command to retry the failed pairs; "
              "results below treat them as p = 0.5")

    res = dict(env=args.env_name, scorer=model.model_id, by_namer={})
    ctrl = predict_factor(model, proc, args.test_dir, args.env_name, cache)
    res["factor_control"] = ctrl
    if ctrl:
        print("[positive control] scorer AUC on ground-truth statements:")
        for f, v in ctrl.items():
            print(f"  {f:<16} AUC={v['auc']:.3f}  bacc={v['bacc']:.3f}  n={v['n']}  {v['statement']}")
        print(f"  mean AUC = {np.nanmean([v['auc'] for v in ctrl.values()]):.3f}  (ceiling for the labels)")
    for path in ([] if args.only_control else args.labels):
        L = json.load(open(path))
        same = L["namer"] == model.model_id
        assert not same or args.allow_same_model, "scorer equals namer: pass --allow_same_model"
        if same:
            print(f"[warning] scorer == namer ({model.model_id}): separate blind calls, "
                  "report as self-consistency, not independent validation")
        labels = labels_all[path]
        r = evaluate(model, proc, labels, args.test_dir, args.env_name,
                     args.n_distractors, args.n_perm, args.seed, cache)
        r["summary"]["scorer_is_namer"] = same
        res["by_namer"][L["namer"]] = r
        s = r["summary"]
        print(f"\n[{L['namer']}] concepts={s['n_concepts']}  AUC true={s['mean_auc_true']:.3f}  "
              f"distractor={s['mean_auc_distractor']:.3f}  delta={s['delta']:+.3f}  "
              f"p_perm={s['p_perm']:.4f}  bacc={s['mean_bacc_true']:.3f}  "
              f"low_mass={s['low_mass_frac']:.2f}")
        for c, v in r["concepts"].items():
            d = np.mean(list(v["auc_distractors"].values())) if v["auc_distractors"] else float("nan")
            print(f"  {c:>6}  {v['auc_true']:.3f} vs {d:.3f}  {v['label']}")
    from vlm_common import manifest
    res["manifest"] = manifest(args, model)
    from vlm_common import is_api, GeminiScorer
    q = (QUESTION.replace("Answer with Yes or No only.", "").rstrip() + GeminiScorer.PROB
         if is_api(model) else QUESTION)                 # the text actually sent to the scorer
    res["manifest"].update(n_scored_pairs=len(cache), prompt_head=HEAD, prompt_question=q)
    json.dump(res, open(args.out, "w"), indent=2, ensure_ascii=False)
    print(f"[done] -> {args.out}")


if __name__ == "__main__":
    main()
