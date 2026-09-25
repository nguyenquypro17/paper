# LUCID vs control — cùng tập held-out (mean ± std, 3 seed)

Held-out = `shuffle_indices[n_train:]` của từng run LUCID. Control chọn model trên val tách từ train; LUCID chọn checkpoint trên held-out → protocol của control chặt hơn, gap là ước lượng bảo thủ.


## doorkey

| Method | Fidelity (hard, logic thuần) | Soft | Literals | Concepts | Coverage |
|---|---|---|---|---|---|
| **LUCID (joint)** | 99.85 ± 0.07 | — | 15.7 ± 5.3 | 11.0 ± 1.4 | 46.5 ± 0.4 |
| Two-stage (frozen SAE + DNF) | 62.09 ± 6.91 | 99.65 ± 0.14 | 37.7 ± 9.3 | 22.3 ± 2.5 | 49.0 ± 30.9 |
| DNF, no SAE | 81.26 ± 0.72 | 99.95 ± 0.02 | 53.0 ± 7.8 | 27.0 ± 0.0 | 74.7 ± 0.3 |

| Method | fid@budget (≤ LUCID literals) | literals→match LUCID fid | best fid (any size) |
|---|---|---|---|
| SAE + CART | 98.67 ± 0.51 | 30.3 ± 2.1 | 99.97 ± 0.04 |
| SAE + L1 linear | 96.41 ± 0.00 | 127.0 ± 52.1 | 99.97 ± 0.02 |

## dyobs

| Method | Fidelity (hard, logic thuần) | Soft | Literals | Concepts | Coverage |
|---|---|---|---|---|---|
| **LUCID (joint)** | 99.99 ± 0.01 | — | 4.3 ± 1.2 | 4.0 ± 0.8 | 100.0 ± 0.0 |
| Two-stage (frozen SAE + DNF) | 77.08 ± 20.20 | 100.00 ± 0.00 | 49.0 ± 12.2 | 27.0 ± 7.1 | 67.9 ± 33.2 |
| DNF, no SAE | 82.93 ± 8.37 | 100.00 ± 0.00 | 32.0 ± 8.5 | 22.7 ± 3.3 | 79.2 ± 13.6 |

| Method | fid@budget (≤ LUCID literals) | literals→match LUCID fid | best fid (any size) |
|---|---|---|---|
| SAE + CART | 97.75 ± 0.49 | 7.7 ± 1.2 | 100.00 ± 0.00 |
| SAE + L1 linear | — | 16.7 ± 4.5 | 100.00 ± 0.00 |

## cartpole

| Method | Fidelity (hard, logic thuần) | Soft | Literals | Concepts | Coverage |
|---|---|---|---|---|---|
| **LUCID (joint)** | 97.45 ± 0.96 | — | 32.0 ± 17.0 | 8.3 ± 1.7 | 95.1 ± 1.6 |
| Two-stage (frozen SAE + DNF) | 83.39 ± 2.00 | 97.35 ± 0.25 | 54.7 ± 29.0 | 19.3 ± 3.1 | 68.0 ± 6.9 |
| DNF, no SAE | 82.07 ± 2.14 | 99.07 ± 0.11 | 72.3 ± 4.8 | 15.0 ± 0.0 | 74.9 ± 10.0 |

| Method | fid@budget (≤ LUCID literals) | literals→match LUCID fid | best fid (any size) |
|---|---|---|---|
| SAE + CART | 93.96 ± 2.52 | 297.0 ± 201.0 (1/3 seed không đạt) | 97.92 ± 0.07 |
| SAE + L1 linear | 96.90 ± 0.06 | 63.3 ± 27.0 | 99.17 ± 0.12 |

## pong

| Method | Fidelity (hard, logic thuần) | Soft | Literals | Concepts | Coverage |
|---|---|---|---|---|---|
| **LUCID (joint)** | 99.78 ± 0.10 | — | 15.0 ± 2.9 | 14.3 ± 2.1 | 99.8 ± 0.1 |
| Two-stage (frozen SAE + DNF) | 51.73 ± 1.52 | 98.36 ± 0.21 | 7841.3 ± 96.4 | 430.3 ± 3.9 | 52.3 ± 2.3 |
| DNF, no SAE | 55.04 ± 1.62 | 99.04 ± 0.04 | 4126.0 ± 119.5 | 395.7 ± 3.3 | 49.6 ± 2.0 |

| Method | fid@budget (≤ LUCID literals) | literals→match LUCID fid | best fid (any size) |
|---|---|---|---|
| SAE + CART | 48.97 ± 1.38 | 692.3 ± 26.7 | 100.00 ± 0.00 |
| SAE + L1 linear | — | — (3/3 seed không đạt) | 99.46 ± 0.09 |

## boxing

| Method | Fidelity (hard, logic thuần) | Soft | Literals | Concepts | Coverage |
|---|---|---|---|---|---|
| **LUCID (joint)** | 98.39 ± 0.24 | — | 99.0 ± 5.9 | 44.0 ± 2.9 | 99.1 ± 0.4 |
| Two-stage (frozen SAE + DNF) | 54.75 ± 7.20 | 92.84 ± 0.23 | 22530.3 ± 602.6 | 481.3 ± 2.6 | 89.1 ± 0.9 |
| DNF, no SAE | 58.67 ± 0.86 | 95.52 ± 0.08 | 19624.3 ± 1194.0 | 490.7 ± 3.4 | 83.9 ± 0.7 |

| Method | fid@budget (≤ LUCID literals) | literals→match LUCID fid | best fid (any size) |
|---|---|---|---|
| SAE + CART | 71.88 ± 3.68 | 2754.0 ± 63.8 | 99.81 ± 0.02 |
| SAE + L1 linear | — | — (3/3 seed không đạt) | 97.86 ± 0.13 |
