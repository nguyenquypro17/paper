# LUCID control-arm ablations

Trả lời trực tiếp W3 (JCx1), "baseline study does not isolate components" (suEM)
và yêu cầu Pareto curve của AC.

## Đặt file

Copy `controls/` và `run_controls.py` vào repo gốc (cùng cấp với `train_joint.py`).
Nếu để chỗ khác thì set `LUCID_ROOT=/đường/dẫn/repo`.

## 6 arm

| Arm | Representation | Learner | Cô lập |
|-----|----------------|---------|--------|
| A0 | SAE joint (checkpoint LUCID) | DNF | LUCID, đo lại bằng cùng protocol |
| A1 | SAE recon-only | CART | representation hay learner tạo ra gain? |
| A2 | SAE recon-only | L1-logistic | DNF có hơn sparse linear không? |
| A3 | raw CNN | bottleneck+DNF | SAE đóng góp gì |
| A4 | SAE recon-only **đóng băng** | bottleneck+DNF | joint vs two-stage |
| A5 | raw CNN | CART ép đúng literal budget của LUCID | so ở cùng complexity |

A1 và A4 là hai arm quyết định framing — chạy trước.

## Complexity thống nhất

`literals` = số lần một feature bị **test**:
DNF → tổng literal (bỏ clause `(True)`/`(False)`); tree → số internal node;
linear → số hệ số khác 0. `concepts` = số feature phân biệt.

## Chạy

```bash
# A1 (chỉ offline, không rollout)
python run_controls.py --arm A1 --env_name MiniGrid-DoorKey-6x6-v0 \
  --features_path data/doorkey/features.pt --episodes 0 --out results/controls

# A4 đầy đủ, có rollout 5 seed
python run_controls.py --arm A4 --env_name MiniGrid-DoorKey-6x6-v0 \
  --features_path data/doorkey/features.pt --ppo_path models/ppo_doorkey.zip \
  --seed 42 --episodes 200 --out results/controls --save_rules

# A0: cần checkpoint LUCID đã train
python run_controls.py --arm A0 --lucid_ckpt runs/doorkey/sae_logic_joint_model.pt ...

# A5: truyền literal count của LUCID (DoorKey=11, CartPole=19)
python run_controls.py --arm A5 --match_literals 11 ...
```

## Pareto

```bash
python -m controls.pareto emit --env_name MiniGrid-DoorKey-6x6-v0 \
  --features_path data/doorkey/features.pt --ppo_path models/ppo_doorkey.zip \
  --seeds 42 43 44 --episodes 100 --out results/pareto > sweep.sh
bash sweep.sh          # hoặc: parallel -j4 < sweep.sh

python -m controls.pareto plot --results_dir results/pareto \
  --out_png figs/pareto_doorkey.png --metric score --title "DoorKey-6x6"
```

Trục y mặc định là **online return**, không phải offline fidelity — đúng chỗ
SA-DT sập ở DoorKey. Đổi `--metric fidelity` để vẽ panel thứ hai.

## Bảng đưa vào paper

Từ các JSON trong `results/controls`, mỗi dòng một arm:
`fidelity_hard` | `complexity.literals` | `complexity.concepts` |
`online_hard.success_rate.mean ± std` | `online_hard.score.mean ± std` |
`coverage_offline`.

Ghi rõ trong caption: cùng offline file, cùng split (seed điều khiển split),
cùng teacher CNN, cùng eval seed 42–46 — đây chính là câu hỏi "do all baselines
receive identical access...?" của suEM.

## Lưu ý

- `--seed` điều khiển **cả** split lẫn khởi tạo. Muốn tách hai nguồn nhiễu thì
  cố định split seed và chỉ đổi seed model (sửa `load_offline(..., seed=split_seed)`).
- Đơn vị của std trong `online_*` là **eval seed**; std giữa các training seed
  lấy bằng cách gộp nhiều file JSON.
- A2 cần `n_jobs=-1` + `saga`, chạy chậm trên D=300; hạ `max_iter` nếu cần.
