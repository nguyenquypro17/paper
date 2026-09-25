import json, torch, numpy as np, argparse
from intervention import load_lucid, parse_rules, hard_policy, winning_clauses

ap = argparse.ArgumentParser()
ap.add_argument("--model_path", required=True)
ap.add_argument("--data_path", required=True)
ap.add_argument("--train_data_path", default=None)
ap.add_argument("--hard_threshold", type=float, default=None)
a = ap.parse_args()

model, cfg, rules = load_lucid(a.model_path, "cpu", a.hard_threshold)
thr = cfg.hard_threshold
d = torch.load(a.data_path, map_location="cpu", weights_only=False)
H = d["features"].float(); act = d["actions"].long().view(-1).numpy()
R = parse_rules(rules, cfg.hidden_dim)

with torch.no_grad():
    zb = []
    for s in range(0, len(H), 4096):
        zs, _ = model.sae.encode(model.normalize_input(H[s:s+4096]))
        zb.append(model.bottleneck(model.normalize_z(zs)))
    ZB = torch.cat(zb)
Z = (ZB > thr).cpu().numpy()

src = torch.load(a.train_data_path, map_location="cpu", weights_only=False)["actions"] \
      if a.train_data_path else d["actions"]
fb = int(np.bincount(src.long().view(-1).numpy()).argmax())
a_hard, Fire = hard_policy(Z, R, fb)

has_lit = np.array([len(winning_clauses(Fire[i], a_hard[i], R)) > 0 for i in range(len(Z))])
for name, m in [("CO literal", has_lit), ("CHI (True)", ~has_lit), ("TAT CA", np.ones_like(has_lit))]:
    if m.sum() == 0: continue
    print(f"{name}: n={m.sum()} ({m.mean()*100:.1f}%)  khop_teacher={(a_hard[m]==act[m]).mean():.4f}")
    print("   a_hard:", dict(zip(*np.unique(a_hard[m], return_counts=True))))
    print("   teacher:", dict(zip(*np.unique(act[m], return_counts=True))))
