"""03c Tuning: Phase 4, tuning the residual CNN-LSTM one setting at a time.

Everything here uses the residual model and the original-units MAE loss, the
combination that first beat persistence in 03b. Training is longer than in
03b (up to 200 epochs, patience 15), because the best 03b model was still
improving when it hit the 100-epoch cap.

1. Tuning: starting from the 03b settings, change ONE setting at a time and
   keep a change only if validation MAE improves by more than MIN_GAIN. The
   test split is not touched in this step.
2. Seeds: the final CNN-LSTM and an LSTM-only model with the same settings are
   trained with 3 seeds each, because the gains over persistence are only a few
   percent. Test numbers are computed here, once, for the final models.
Window length is not tuned here (it changes the data; see the report).

Saved models are reused on later runs; set RETRAIN = True to train again.
Output: results/report.html
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
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")     # hide TensorFlow's start-up messages
import numpy as np
import tensorflow as tf  # imported before set_seeds() so TF is seeded too
import common
importlib.reload(common)
from common import *
report = Report("03c_tuning", "Phase 4: Tuning and seeds")
report.section("Project status")
set_seeds()
status()

import matplotlib.pyplot as plt
use_plot_style()
RETRAIN = False          # True: train again even if a saved model exists
EPOCHS, PATIENCE = 200, 15
MIN_GAIN = 0.005         # a change must lower validation MAE by more than 0.5% to be kept
SEEDS = (42, 43, 44)


def describe(m):
    return (f"MAE {m['mae'] * MBIT:7.3f} Mbit/s   RMSE {m['rmse'] * MBIT:7.3f} Mbit/s   "
            f"MAPE {m['mape']:5.1f}% (coverage {m['mape_coverage'] * 100:.0f}%)")


# %% 1. data and the persistence reference
report.section("1. Data and the persistence reference")
X_train, y_train, X_val, y_val, X_test, y_test = load_splits()
scaler = FlowScaler.load()
pers = {"val": evaluate_scaled(y_val, persistence(X_val), scaler),
        "test": evaluate_scaled(y_test, persistence(X_test), scaler)}
print(f"persistence val   {describe(pers['val'])}")
print(f"persistence test  {describe(pers['test'])}")
REF = float(scaler.inverse_transform(y_train).mean())
LOSS = make_loss("mae_raw", scaler, REF)


def tag(kind, cfg, seed):
    u = "-".join(str(x) for x in cfg["units"])
    conv = f"f{cfg['filters']}_k{cfg['kernel']}_" if kind == "cnn_lstm" else ""
    return f"p4_{kind}_{conv}u{u}_d{int(round(cfg['dropout'] * 100))}_lr{cfg['lr']:g}_s{seed}"


def build_for(kind, cfg):
    if kind == "cnn_lstm":
        return lambda: build_cnn_lstm(filters=cfg["filters"], kernel=cfg["kernel"],
                                      units=cfg["units"], dropout=cfg["dropout"], residual=True)
    return lambda: build_lstm(units=cfg["units"], dropout=cfg["dropout"], residual=True)


def run(kind, cfg, seed=SEED, with_test=False):
    """Train (or load) one model; return (val metrics, test metrics or None, epochs, best epoch)."""
    name = tag(kind, cfg, seed)
    model, hist, _ = fit_or_load(build_for(kind, cfg), name, X_train, y_train, X_val, y_val,
                                 retrain=RETRAIN, seed=seed, epochs=EPOCHS, patience=PATIENCE,
                                 lr=cfg["lr"], loss=LOSS)
    best = int(np.argmin(hist["val_loss"])) + 1
    m_val = evaluate_scaled(y_val, predict(model, X_val, scaler=scaler), scaler)
    note = f"loss mae_raw, best epoch {best} of {len(hist['loss'])}, seed {seed}"
    log_result(name, m_val, notes=f"val split, all 144 flows, {note}")
    m_test = None
    if with_test:
        m_test = evaluate_scaled(y_test, predict(model, X_test, scaler=scaler), scaler)
        log_result(name, m_test, notes=f"test split, all 144 flows, {note}")
    return m_val, m_test, len(hist["loss"]), best


# %% 2. tuning, one setting at a time (validation only)
report.section("2. Tuning one setting at a time")
cfg = {"filters": 64, "kernel": 3, "units": (100, 50), "dropout": 0.2, "lr": 1e-3}
GRID = [("filters", [32, 128]), ("kernel", [2, 5]), ("units", [(64, 32), (128, 64)]),
        ("dropout", [0.0, 0.4]), ("lr", [3e-4, 3e-3])]
val_mae = lambda m: m["mae"] * MBIT

print("Base: the 03b settings, trained longer")
m, _, ep, best = run("cnn_lstm", cfg)
best_mae = val_mae(m)
trials = [("base", str(cfg), best_mae, True)]
print(f"  base  val MAE {best_mae:.3f} Mbit/s  ({ep} epochs, best {best})")
for key, options in GRID:
    print(f"\n{key}: current {cfg[key]}")
    winner, winner_mae = cfg[key], best_mae
    for opt in options:
        trial = {**cfg, key: opt}
        m, _, ep, best = run("cnn_lstm", trial)
        v = val_mae(m)
        print(f"  {key}={opt}  val MAE {v:.3f} Mbit/s  ({ep} epochs, best {best})")
        trials.append((f"{key}={opt}", None, v, False))
        if v < winner_mae * (1 - MIN_GAIN) and v < best_mae * (1 - MIN_GAIN):
            winner, winner_mae = opt, v
    if winner != cfg[key]:
        print(f"  -> keep {key}={winner} (val MAE {best_mae:.3f} -> {winner_mae:.3f})")
        cfg[key], best_mae = winner, winner_mae
        trials = [(n, c, v, k or n == f"{key}={winner}") for n, c, v, k in trials]
    else:
        print(f"  -> keep {key}={cfg[key]}")
print(f"\nFinal settings: {cfg}\nvalidation MAE {best_mae:.3f} Mbit/s "
      f"({(1 - best_mae / val_mae(pers['val'])) * 100:+.1f}% vs persistence)")

# %% 3. seeds: final CNN-LSTM against LSTM-only
report.section("3. Three seeds: CNN-LSTM against LSTM-only")
seed_res = {"CNN-LSTM": [], "LSTM-only": []}
for seed in SEEDS:
    for label, kind in (("CNN-LSTM", "cnn_lstm"), ("LSTM-only", "lstm")):
        mv, mt, ep, best = run(kind, cfg, seed=seed, with_test=True)
        seed_res[label].append((mv, mt))
        print(f"seed {seed}  {label:10s}  val {val_mae(mv):.3f}  test {val_mae(mt):.3f} Mbit/s  "
              f"({ep} epochs, best {best})")
print(f"\n{'':12s} {'val MAE':>16s} {'vs pers':>9s}   {'test MAE':>16s} {'vs pers':>9s}   seeds beating persistence (test)")
print(f"  {'persistence':10s} {val_mae(pers['val']):12.3f}          {val_mae(pers['test']):12.3f}")
summary = {}
for label, rows in seed_res.items():
    v = np.array([val_mae(a) for a, _ in rows])
    t = np.array([val_mae(b) for _, b in rows])
    summary[label] = (v, t)
    print(f"  {label:10s} {v.mean():8.3f} ± {v.std():5.3f} {(1 - v.mean() / val_mae(pers['val'])) * 100:+8.1f}%   "
          f"{t.mean():8.3f} ± {t.std():5.3f} {(1 - t.mean() / val_mae(pers['test'])) * 100:+8.1f}%   "
          f"{(t < val_mae(pers['test'])).sum()}/{len(t)}")

# %% 4. figures
report.section("4. Figures")
fig, ax = plt.subplots(figsize=(10, 0.36 * len(trials) + 1.4))
yy = np.arange(len(trials))[::-1]
for yi, (n, _, v, kept) in zip(yy, trials):
    ax.barh(yi, v, 0.62, color=COLORS["aqua"] if kept else COLORS["ink2"], alpha=1.0 if kept else 0.45,
            edgecolor="white")
    ax.text(v, yi, f"  {v:.3f}", va="center", fontsize=8.5, color=COLORS["ink2"])
ax.set_yticks(yy, [t[0] for t in trials], fontsize=8.5)
ax.axvline(val_mae(pers["val"]), color=COLORS["orange"], linewidth=1, linestyle="--", label="persistence")
lo = min(t[2] for t in trials)
ax.set_xlim(lo * 0.9, max(t[2] for t in trials) * 1.06)
ax.set_xlabel("validation MAE, Mbit/s (axis starts above zero; lower is better)")
ax.set_title("Tuning trials, one setting at a time (solid = kept)")
ax.legend(frameon=False, loc="lower right"); ax.grid(axis="y", visible=False)
fig.tight_layout()
show(fig, "03c_tuning.png", "Validation MAE of each trial. Solid bars are the settings that were kept.")

fig, ax = plt.subplots(figsize=(8.5, 3.6))
for i, (label, (v, t)) in enumerate(summary.items()):
    c = COLORS["aqua"] if label == "LSTM-only" else COLORS["violet"]
    ax.scatter(np.full(len(t), i), t, color=c, s=40, zorder=3)
    ax.hlines(t.mean(), i - 0.2, i + 0.2, color=c, linewidth=2.5)
ax.axhline(val_mae(pers["test"]), color=COLORS["orange"], linewidth=1.2, linestyle="--", label="persistence")
ax.set_xticks(range(len(summary)), list(summary)); ax.set_xlim(-0.6, len(summary) - 0.4)
ax.set_ylabel("test MAE, Mbit/s"); ax.legend(frameon=False)
ax.set_title("Test MAE over 3 seeds (dots = seeds, bar = mean)")
fig.tight_layout()
show(fig, "03c_seeds.png", "Test MAE for each seed. The dashed line marks persistence.")

# %% write the results page and open it in the browser
report.finish()
