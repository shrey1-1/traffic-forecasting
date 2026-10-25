"""03f GÉANT tuning: does tuning the CNN-LSTM on GÉANT itself add anything?

03e retrained the Abilene settings on GÉANT with NO retuning (+2.7% vs
persistence at 60 minutes). This script is a separate, clearly labelled
experiment: the residual CNN-LSTM at horizon 4 (60 min) is tuned on GÉANT's
own validation split, with the same rule as 03c on Abilene:

1. Start from the Phase 4 settings (the 03e model, seed 42, reused as the base).
   Change ONE setting at a time; keep a change only if validation MAE improves by
   more than MIN_GAIN. The test split is not touched in this step.
2. If any setting changed, train the final settings with 3 seeds and compute
   test numbers once. If nothing changed, the 03e result stands.

The 03e models start overfitting almost at once (best epoch 3-4 of 18-19),
so lower learning rates and more dropout are included in the grid.

Isolation: models are saved as gt_*.keras and results go to
results/03f_geant_tuning.csv, NOT metrics.csv, so the logged project results
and the report page are unchanged unless this result is adopted.
"""

# %% header: locate common.py, import it, seed, show status
import importlib, sys
from pathlib import Path

_p = Path(globals().get("__file__", Path.cwd())).resolve()
while not (_p / "common.py").exists():
    if _p.parent == _p:
        raise FileNotFoundError("common.py not found above this script or the working directory")
    _p = _p.parent
if str(_p) not in sys.path:
    sys.path.insert(0, str(_p))

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import csv, time
import numpy as np
import tensorflow as tf  # imported before set_seeds() so TF is seeded too
import common
importlib.reload(common)
from common import *
set_seeds()
status()

RETRAIN = False
EPOCHS, PATIENCE = 200, 15
MIN_GAIN = 0.005
H = 4
SEEDS = (42, 43, 44)
CAP = GEANT_MAX_FLOW_KBPS
OUT = RESULTS / "03f_geant_tuning.csv"
mb = lambda m: m["mae"] / 1000            # kbit/s -> Mbit/s

# %% 1. data and persistence
d = PROCESSED / "geant" / f"h{H}"
X_train, y_train, X_val, y_val, X_test, y_test = load_splits(d)
scaler = FlowScaler.load(d / "scaler.npz")
LOSS = make_loss("mae_raw", scaler, float(scaler.inverse_transform(y_train).mean()))
pers = {"val": evaluate_scaled(y_val, persistence(X_val), scaler),
        "test": evaluate_scaled(y_test, persistence(X_test), scaler)}
print(f"\nGEANT horizon {H} ({H * 15} min): train {len(y_train)}, val {len(y_val)}, test {len(y_test)}")
print(f"persistence  val {mb(pers['val']):.3f}  test {mb(pers['test']):.3f} Mbit/s")

BASE = {"filters": 64, "kernel": 3, "units": (100, 50), "dropout": 0.2, "lr": 1e-3}


def name_for(cfg, seed):
    if cfg == BASE:
        return f"g_h{H}_cnn_lstm_s{seed}"          # the 03e model, reused
    u = "-".join(str(x) for x in cfg["units"])
    return (f"gt_h{H}_cnn_lstm_f{cfg['filters']}_k{cfg['kernel']}_u{u}"
            f"_d{int(round(cfg['dropout'] * 100))}_lr{cfg['lr']:g}_s{seed}")


rows = []


def run(cfg, seed=42, with_test=False, label=""):
    name = name_for(cfg, seed)
    t = time.time()
    build = lambda: build_cnn_lstm(n_flows=GEANT_FLOWS, filters=cfg["filters"], kernel=cfg["kernel"],
                                   units=cfg["units"], dropout=cfg["dropout"], residual=True)
    model, hist, trained = fit_or_load(build, name, X_train, y_train, X_val, y_val, retrain=RETRAIN,
                                       seed=seed, epochs=EPOCHS, patience=PATIENCE, lr=cfg["lr"], loss=LOSS)
    best = int(np.argmin(hist["val_loss"])) + 1
    mv = evaluate_scaled(y_val, predict(model, X_val, scaler=scaler, max_raw=CAP), scaler)
    mt = evaluate_scaled(y_test, predict(model, X_test, scaler=scaler, max_raw=CAP), scaler) if with_test else None
    row = {"label": label, "model": name, "seed": seed, "val_mae_mbit": round(mb(mv), 4),
           "val_vs_pers_pct": round((1 - mv["mae"] / pers["val"]["mae"]) * 100, 2),
           "test_mae_mbit": round(mb(mt), 4) if mt else "",
           "test_vs_pers_pct": round((1 - mt["mae"] / pers["test"]["mae"]) * 100, 2) if mt else "",
           "epochs": len(hist["loss"]), "best_epoch": best, "seconds": round(time.time() - t)}
    rows.append(row)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row)); w.writeheader(); w.writerows(rows)
    print(f"  {label:22s} val {row['val_mae_mbit']:.3f} ({row['val_vs_pers_pct']:+.2f}% vs pers)"
          + (f"  test {row['test_mae_mbit']:.3f} ({row['test_vs_pers_pct']:+.2f}%)" if mt else "")
          + f"  [{row['epochs']} epochs, best {best}{', trained' if trained else ', loaded'}]", flush=True)
    return mv, mt


# %% 2. tuning, one setting at a time (validation only)
print("\n2. Tuning one setting at a time (validation only)")
GRID = [("lr", [3e-4, 1e-4, 3e-3]), ("dropout", [0.4, 0.5, 0.0]), ("units", [(64, 32), (128, 64)]),
        ("filters", [32, 128]), ("kernel", [2, 5])]
cfg = dict(BASE)
m, _ = run(cfg, label="base (03e settings)")
best_mae = mb(m)
for key, options in GRID:
    winner, winner_mae = cfg[key], best_mae
    for opt in options:
        m, _ = run({**cfg, key: opt}, label=f"{key}={opt}")
        v = mb(m)
        if v < winner_mae * (1 - MIN_GAIN) and v < best_mae * (1 - MIN_GAIN):
            winner, winner_mae = opt, v
    if winner != cfg[key]:
        print(f"  -> keep {key}={winner} (val {winner_mae:.3f} vs {best_mae:.3f})")
        cfg[key], best_mae = winner, winner_mae
    else:
        print(f"  -> keep {key}={cfg[key]} (no option better by more than {MIN_GAIN:.1%})")

# %% 3. final settings: 3 seeds, test once
print(f"\n3. Final settings: {cfg}")
if cfg == BASE:
    print("No setting beat the base by more than 0.5% on validation: the 03e (untuned) result stands.")
else:
    vals, tests = [], []
    for s in SEEDS:
        mv, mt = run(cfg, seed=s, with_test=True, label=f"final seed {s}")
        vals.append(mb(mv)); tests.append(mb(mt))
    vals, tests = np.array(vals), np.array(tests)
    pv, pt = mb(pers["val"]), mb(pers["test"])
    print(f"\npersistence      val {pv:.3f}   test {pt:.3f} Mbit/s")
    print(f"tuned CNN-LSTM   val {vals.mean():.3f} ± {vals.std():.3f} ({(1 - vals.mean() / pv) * 100:+.2f}%)   "
          f"test {tests.mean():.3f} ± {tests.std():.3f} ({(1 - tests.mean() / pt) * 100:+.2f}%), "
          f"{(tests < pt).sum()}/3 seeds beat persistence")
    print("untuned (03e)    test 2.313 (+2.7%)  [for comparison]")
print(f"\nAll rows saved to {OUT}")
