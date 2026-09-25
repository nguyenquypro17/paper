#!/usr/bin/env python3
"""
vlm_name.py -- Step 1: name each concept from single-view ON / OFF examples only.

Input per concept: naming/C{c}/on_*.png and off_*.png (MiniGrid: agent's ego-centric view;
CartPole / Atari: full frame), sent as separate images in ONE call. No heatmap, no rules,
no actions, no score request. The model first describes every image, then gives the label.

Usage:
    export GEMINI_API_KEY=...
    python vlm_name.py --img_dir experiments/vlm/doorkey_42/naming \
        --env_name MiniGrid-DoorKey-6x6-v0 --model gemini-2.5-pro \
        --out experiments/vlm/doorkey_42/labels_gemini.json
"""

import argparse
import glob
import json
import os

from tqdm import tqdm

from vlm_common import load_vlm, chat, parse_json, env_context, manifest

INTRO = """You are analysing one internal feature ("concept") of an agent that plays a video game.

ENVIRONMENT: {env}

Below are {n_on} states where the concept is ON, followed by {n_off} states where it is OFF."""

TASK = """Task.
Step 1: for EVERY image above, write one short line describing what is visible and where each
object is (for a grid view: relative to the agent -- in front, left, right, far or near; for a
full frame: positions and tilts of the objects).
Step 2: find the property that holds in all (or almost all) ON images and fails in most OFF
images. Use only what you can see. Do not reuse wording from these instructions unless it
really is the distinguishing property.
Step 3: end your answer with JSON on the last line:
{{"label": "short_snake_case_description_of_that_property", "evidence": "one sentence"}}"""


def name_concept(model, proc, on_paths, off_paths, env_name, max_new_tokens=2000):
    """-> dict(label, evidence, raw)"""
    content = [INTRO.format(env=env_context(env_name), n_on=len(on_paths), n_off=len(off_paths))]
    for j, p in enumerate(on_paths):
        content += [f"ON example {j + 1}:", p]
    for j, p in enumerate(off_paths):
        content += [f"OFF example {j + 1}:", p]
    content.append(TASK)
    raw = chat(model, proc, content, max_new_tokens)
    js = parse_json(raw)
    label = str(js.get("label", "")).strip().lower().replace(" ", "_") or "unparsed"
    return dict(label=label, evidence=js.get("evidence", ""), raw=raw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_dir", required=True, help="<render out_dir>/naming")
    ap.add_argument("--env_name", required=True)
    ap.add_argument("--model", required=True, help="qwen | internvl | gemma | HF repo id | path")
    ap.add_argument("--load_4bit", action="store_true")
    ap.add_argument("--max_new_tokens", type=int, default=2000)
    ap.add_argument("--thinking_budget", type=int, default=4096,
                    help="Gemini budget; OpenAI: >=4096 high, >=1024 medium, else low")
    ap.add_argument("--max_concepts", type=int, default=None, help="random subset (seeded)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget_usd", type=float, default=None, help="API spend cap, shared across runs")
    ap.add_argument("--price_in", type=float, default=None, help="USD per 1M input tokens")
    ap.add_argument("--price_out", type=float, default=None, help="USD per 1M output tokens")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    dirs = sorted([d for d in glob.glob(os.path.join(args.img_dir, "C*")) if os.path.isdir(d)],
                  key=lambda d: int(os.path.basename(d)[1:]))
    assert dirs, f"no C*/ folders in {args.img_dir}: run render_concepts.py first"
    if args.max_concepts and len(dirs) > args.max_concepts:
        import numpy as np
        pick = np.random.default_rng(args.seed).choice(len(dirs), args.max_concepts, replace=False)
        dirs = [dirs[i] for i in sorted(pick)]
        print(f"[subset] {args.max_concepts} concepts (seed {args.seed}): {[os.path.basename(d) for d in dirs]}")
    from vlm_common import attach_budget
    model, proc = load_vlm(args.model, args.load_4bit, args.thinking_budget)
    attach_budget(model, args.budget_usd, args.price_in, args.price_out)
    res = dict(env=args.env_name, namer=model.model_id, concepts={})
    partial = args.out + ".partial"                  # paid answers survive a stop (budget, crash)
    if os.path.exists(partial):
        res["concepts"] = {c: v for c, v in json.load(open(partial))["concepts"].items() if v["raw"].strip()}
        print(f"[resume] {len(res['concepts'])} concepts already named in {partial}")
    for d in tqdm(dirs, desc=f"Naming ({args.model})"):
        cid = os.path.basename(d)
        if cid in res["concepts"]:
            continue
        on = sorted(glob.glob(os.path.join(d, "on_*.png")))
        off = sorted(glob.glob(os.path.join(d, "off_*.png")))
        res["concepts"][cid] = name_concept(model, proc, on, off, args.env_name, args.max_new_tokens)
        if res["concepts"][cid]["raw"].strip():
            json.dump(res, open(partial, "w"), indent=2, ensure_ascii=False)
    res["concepts"] = {os.path.basename(d): res["concepts"][os.path.basename(d)] for d in dirs}
    res["manifest"] = manifest(args, model)
    failed = [c for c, v in res["concepts"].items() if not v["raw"].strip()]
    if failed:                      # empty answers = API failure: do not save, so a re-run retries
        print(f"[error] empty API response for {failed}; errors = {getattr(model, 'errors', {})}")
        print(f"[error] {args.out} NOT written. Fix the API problem and re-run.")
        raise SystemExit(1)
    json.dump(res, open(args.out, "w"), indent=2, ensure_ascii=False)
    if os.path.exists(partial):
        os.remove(partial)

    labels = [v["label"] for v in res["concepts"].values()]
    for cid, v in res["concepts"].items():
        print(f"  {cid:>6}: {v['label']}")
    print(f"[done] {len(dirs)} concepts, unparsed={labels.count('unparsed')}, "
          f"distinct labels={len(set(labels))} -> {args.out}")


if __name__ == "__main__":
    main()
