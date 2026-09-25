"""
patch_lucid_split.py
Cho control dùng ĐÚNG tập held-out của từng run LUCID.

  held-out (test) = shuffle_indices[n_train:] trong <lucid_run>/training_data.pt
                    -> y hệt tập mà check_scoring.py dùng để đo LUCID
  train           = shuffle_indices[:n_train], rồi tách 10% cuối làm val để chọn model

Control chọn model trên val tách từ train (không đụng test), trong khi LUCID
chọn checkpoint trên chính tập held-out. Protocol của control vì vậy CHẶT hơn
-> mọi gap đo được là ước lượng bảo thủ, có lợi cho baseline.

Chạy một lần trong ~/XRL:  python patch_lucid_split.py
"""
import re

# ---------------------------------------------------------------- data.py
DATA_FN = '''

def load_offline_from_lucid(run_dir: str, env_name: str, seed: int = 42,
                            val_frac: float = 0.1) -> OfflineData:
    """
    Reuse the exact split of a trained LUCID run.

    test  = shuffle_indices[n_train:]            (LUCID's held-out set)
    train = shuffle_indices[:n_train] minus the last `val_frac` of it
    val   = that last `val_frac`, used only for control model selection
    """
    import os
    d = torch.load(os.path.join(run_dir, "training_data.pt"), map_location="cpu", weights_only=False)
    features = d["features"].float()
    actions = d["actions"].long().view(-1)
    feat_mean = d["feature_mean"].float().view(-1)
    feat_std = d["feature_std"].float().view(-1).clamp(min=1e-6)
    if not bool(d.get("pre_normalized", False)):
        features = (features - feat_mean) / feat_std

    n_actions, action_names = resolve_actions(env_name, actions)
    actions = torch.clamp(actions, min=0, max=n_actions - 1)

    idx = torch.as_tensor(d["shuffle_indices"]).long()
    n_train = int(d["n_train"])
    train_all, te_idx = idx[:n_train], idx[n_train:]

    g = torch.Generator().manual_seed(seed)
    train_all = train_all[torch.randperm(len(train_all), generator=g)]
    n_val = int(val_frac * len(train_all))
    val_idx, tr_idx = train_all[:n_val], train_all[n_val:]

    return OfflineData(
        X_tr=features[tr_idx], a_tr=actions[tr_idx],
        X_val=features[val_idx], a_val=actions[val_idx],
        X_te=features[te_idx], a_te=actions[te_idx],
        feat_mean=feat_mean, feat_std=feat_std,
        n_actions=n_actions, action_names=action_names,
    )
'''

p = "controls/data.py"
s = open(p).read()
if "def load_offline_from_lucid" not in s:
    s = s.rstrip() + "\n" + DATA_FN
    open(p, "w").write(s)
    print("[ok] controls/data.py : thêm load_offline_from_lucid")
else:
    print("[skip] controls/data.py đã có load_offline_from_lucid")

# ---------------------------------------------------------- run_controls.py
p = "run_controls.py"
s = open(p).read()
changes = 0

old = "from controls.data import load_offline\n"
new = "from controls.data import load_offline, load_offline_from_lucid\n"
if old in s:
    s = s.replace(old, new); changes += 1

old = ("    data = load_offline(args.features_path, args.env_name, seed=args.seed,\n"
       "                        stage1_path=args.stage1_path or None)\n")
new = ("    if args.lucid_split_dir:\n"
       "        data = load_offline_from_lucid(args.lucid_split_dir, args.env_name, seed=args.seed)\n"
       "        print(f\"  split = LUCID held-out of {args.lucid_split_dir}\")\n"
       "    else:\n"
       "        data = load_offline(args.features_path, args.env_name, seed=args.seed,\n"
       "                            stage1_path=args.stage1_path or None)\n")
if old in s:
    s = s.replace(old, new); changes += 1

old = '    p.add_argument("--features_path", required=True)\n'
new = ('    p.add_argument("--features_path", default="")\n'
       '    p.add_argument("--lucid_split_dir", default="",\n'
       '                   help="dùng đúng split train/held-out của run LUCID này")\n')
if old in s:
    s = s.replace(old, new); changes += 1

old = '        "learner": learner_kind, **meta,\n'
new = '        "learner": learner_kind, "lucid_split_dir": args.lucid_split_dir, **meta,\n'
if old in s:
    s = s.replace(old, new); changes += 1

open(p, "w").write(s)
print(f"[ok] run_controls.py : {changes}/4 chỗ vá")
if changes != 4:
    print("[!!] thiếu chỗ vá — gửi output này cho Claude")
