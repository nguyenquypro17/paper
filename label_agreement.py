#!/usr/bin/env python3
"""
label_agreement.py -- Step 3: inter-model naming consistency (namer A vs namer B).

Signal: cosine between A's and B's labels of the SAME concept.
Null:   cosine between A's label of one concept and B's label of ANOTHER concept (same env).
Report per embedder: AUC(signal vs null), permutation p (shuffle B's concept assignment).
This measures agreement between namers, not whether the labels are correct.

Usage:
    python label_agreement.py \
        --labels_a experiments/vlm/doorkey_42/labels_qwen.json \
        --labels_b experiments/vlm/doorkey_42/labels_internvl.json \
        --out experiments/vlm/doorkey_42/agreement.json
"""

import argparse
import json

import numpy as np

EMBEDDERS = ["sentence-transformers/all-MiniLM-L6-v2", "BAAI/bge-base-en-v1.5",
             "intfloat/e5-base-v2", "thenlper/gte-base"]


def auc_diag(S):
    """S (C,C) similarity -> P(diag > offdiag), ties 0.5"""
    C = len(S)
    diag = np.diag(S)
    off = S[~np.eye(C, dtype=bool)]
    d = diag[:, None] - off[None, :]
    return float(((d > 0) + 0.5 * (d == 0)).mean())


def agreement(S, n_perm=10000, seed=0):
    """-> dict(auc, p_perm, mean_same, mean_diff)"""
    rng = np.random.default_rng(seed)
    obs = auc_diag(S)
    null = np.array([auc_diag(S[:, rng.permutation(len(S))]) for _ in range(n_perm)])
    C = len(S)
    return dict(auc=obs, p_perm=float((1 + (null >= obs).sum()) / (1 + n_perm)),
                mean_same=float(np.diag(S).mean()),
                mean_diff=float(S[~np.eye(C, dtype=bool)].mean()))


def embed(model_id, texts):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(model_id)
    if "e5" in model_id:
        texts = ["query: " + t for t in texts]
    return m.encode(texts, normalize_embeddings=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels_a", required=True)
    ap.add_argument("--labels_b", required=True)
    ap.add_argument("--embedders", nargs="+", default=EMBEDDERS)
    ap.add_argument("--n_perm", type=int, default=10000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from vlm_common import label_to_text
    A, B = json.load(open(args.labels_a)), json.load(open(args.labels_b))
    cids = sorted(set(A["concepts"]) & set(B["concepts"]), key=lambda s: int(s[1:]))
    cids = [c for c in cids if "unparsed" not in (A["concepts"][c]["label"], B["concepts"][c]["label"])]
    assert len(cids) >= 3, f"need >= 3 common concepts, got {len(cids)}"
    ta = [label_to_text(A["concepts"][c]["label"]) for c in cids]
    tb = [label_to_text(B["concepts"][c]["label"]) for c in cids]

    res = dict(namer_a=A["namer"], namer_b=B["namer"], concepts=cids,
               exact_match=float(np.mean([a == b for a, b in zip(ta, tb)])), by_embedder={})
    for e in args.embedders:
        S = embed(e, ta) @ embed(e, tb).T
        r = agreement(S, args.n_perm)
        res["by_embedder"][e] = r
        print(f"  {e:<42} AUC={r['auc']:.3f}  p={r['p_perm']:.4f}  "
              f"same={r['mean_same']:.3f}  diff={r['mean_diff']:.3f}")
    from vlm_common import manifest
    res["manifest"] = manifest(args)
    json.dump(res, open(args.out, "w"), indent=2)
    print(f"[done] {len(cids)} concepts -> {args.out}")


if __name__ == "__main__":
    main()
