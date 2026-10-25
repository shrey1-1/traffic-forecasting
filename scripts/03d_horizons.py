"""03d Horizons and ablations: Phase 5.

Predict 1, 6 and 12 steps ahead (5, 30 and 60 minutes) and take parts of the
CNN-LSTM away to see which ones matter. Every model is residual and trained on
MAE in original units, with the settings from Phase 4 (tuning changed nothing).
Each horizon has its own data and scaler (processed/h6, processed/h12; the
horizon 1 data is the Phase 1 data) and its own persistence reference.

Main comparison (3 seeds each): CNN-LSTM, LSTM-only, CNN-only.
Ablations (1 seed, 42): the CNN-LSTM with one convolution layer, without
max-pooling, or with one LSTM layer. Nothing is chosen from the test split:
there is no selection step in this script, so val and test are both reported.

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
report = Report("03d_horizons", "Phase 5: Horizons and ablations")
report.section("Project status")
set_seeds()
status()

import matplotlib.pyplot as plt
use_plot_style()
RETRAIN = False          # True: train again even if a saved model exists
EPOCHS, PATIENCE = 200, 15
HORIZONS = (1, 6, 12)
SEEDS = (42, 43, 44)
CFG = {"filters": 64, "kernel": 3, "units": (100, 50), "dropout": 0.2, "lr": 1e-3}   # Phase 4 result

VARIANTS = {   # key: (label, builder, seeds)
    "cnn_lstm": ("CNN-LSTM", lambda: build_cnn_lstm(residual=True), SEEDS),
    "lstm": ("LSTM-only", lambda: build_lstm(residual=True), SEEDS),
    "cnn": ("CNN-only", lambda: build_cnn(residual=True), SEEDS),
    "cnn_lstm_1conv": ("CNN-LSTM, 1 conv layer", lambda: build_cnn_lstm(residual=True, n_conv=1), (42,)),
    "cnn_lstm_nopool": ("CNN-LSTM, no pooling", lambda: build_cnn_lstm(residual=True, pool=False), (42,)),
    "cnn_lstm_1lstm": ("CNN-LSTM, 1 LSTM layer", lambda: build_cnn_lstm(residual=True, units=(100,)), (42,)),
}
MAIN = ("cnn_lstm", "lstm", "cnn")
mbit = lambda m: m["mae"] * MBIT


def describe(m):
    return (f"MAE {m['mae'] * MBIT:7.3f} Mbit/s   RMSE {m['rmse'] * MBIT:7.3f} Mbit/s   "
            f"MAPE {m['mape']:5.1f}% (coverage {m['mape_coverage'] * 100:.0f}%)")


def model_name(h, key, seed):
    """Horizon 1 reuses the Phase 4 models where the settings are identical."""
    if h == 1 and key in ("cnn_lstm", "lstm"):
        u = "-".join(str(x) for x in CFG["units"])
        conv = f"f{CFG['filters']}_k{CFG['kernel']}_" if key == "cnn_lstm" else ""
        return f"p4_{key}_{conv}u{u}_d20_lr0.001_s{seed}"
    return f"p5_h{h}_{key}_s{seed}"


# %% 1. train (or load) everything, one horizon at a time
res = {}        # res[h]["persistence"] = {"val","test"};  res[h][key] = [(seed, val, test), ...]
for h in HORIZONS:
    report.section(f"1. Horizon {h} ({h * 5} minutes ahead)")
    d = PROCESSED if h == 1 else PROCESSED / f"h{h}"
    if not (d / "scaler.npz").exists():
        prepare_abilene(horizon=h, directory=d)
    X_train, y_train, X_val, y_val, X_test, y_test = load_splits(d)
    scaler = FlowScaler.load(d / "scaler.npz")
    loss = make_loss("mae_raw", scaler, float(scaler.inverse_transform(y_train).mean()))
    res[h] = {"persistence": {"val": evaluate_scaled(y_val, persistence(X_val), scaler),
                              "test": evaluate_scaled(y_test, persistence(X_test), scaler)}}
    pv, pt = res[h]["persistence"]["val"], res[h]["persistence"]["test"]
    print(f"train {len(y_train)}, val {len(y_val)}, test {len(y_test)} examples")
    print(f"persistence val   {describe(pv)}")
    print(f"persistence test  {describe(pt)}")
    for key, (label, build, seeds) in VARIANTS.items():
        res[h][key] = []
        for seed in seeds:
            name = model_name(h, key, seed)
            model, hist, _ = fit_or_load(build, name, X_train, y_train, X_val, y_val, retrain=RETRAIN,
                                         seed=seed, epochs=EPOCHS, patience=PATIENCE, lr=CFG["lr"], loss=loss)
            mv = evaluate_scaled(y_val, predict(model, X_val, scaler=scaler), scaler)
            mt = evaluate_scaled(y_test, predict(model, X_test, scaler=scaler), scaler)
            note = f"loss mae_raw, best epoch {int(np.argmin(hist['val_loss'])) + 1} of {len(hist['loss'])}, seed {seed}"
            log_result(name, mv, horizon=h, notes=f"val split, all 144 flows, {note}")
            log_result(name, mt, horizon=h, notes=f"test split, all 144 flows, {note}")
            res[h][key].append((seed, mv, mt))
            print(f"  {label:24s} seed {seed}  val {mbit(mv):7.3f}  test {mbit(mt):7.3f} Mbit/s  "
                  f"(test {(1 - mt['mae'] / pt['mae']) * 100:+.1f}% vs persistence)")

# %% 2. results
report.section("2. Results")
gain = lambda m, base: (1 - m["mae"] / base["mae"]) * 100
for h in HORIZONS:
    pv, pt = res[h]["persistence"]["val"], res[h]["persistence"]["test"]
    print(f"\nHorizon {h} ({h * 5} min). MAE in Mbit/s; change: positive = better than persistence")
    print(f"  {'':26s} {'val MAE':>16s} {'change':>8s}   {'test MAE':>16s} {'change':>8s}   seeds beating pers. (test)")
    print(f"  {'persistence':26s} {mbit(pv):16.3f} {'':8s}   {mbit(pt):16.3f}")
    for key, (label, _, _) in VARIANTS.items():
        rows = res[h][key]
        v = np.array([mbit(a) for _, a, _ in rows]); t = np.array([mbit(b) for _, _, b in rows])
        pm = f" ± {v.std():5.3f}" if len(rows) > 1 else "        "
        tm = f" ± {t.std():5.3f}" if len(rows) > 1 else "        "
        print(f"  {label:26s} {v.mean():8.3f}{pm} {(1 - v.mean() / mbit(pv)) * 100:+7.1f}%   "
              f"{t.mean():8.3f}{tm} {(1 - t.mean() / mbit(pt)) * 100:+7.1f}%   {(t < mbit(pt)).sum()}/{len(t)}")

# %% 3. figures
report.section("3. Figures")
HUE = {"cnn_lstm": COLORS["violet"], "lstm": COLORS["aqua"], "cnn": COLORS["orange"]}
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
for ax, split, title in zip(axes, (1, 2), ("validation", "test")):
    for i, key in enumerate(MAIN):
        mean, sd = [], []
        for h in HORIZONS:
            base = mbit(res[h]["persistence"]["val" if split == 1 else "test"])
            g = np.array([(1 - mbit(r[split]) / base) * 100 for r in res[h][key]])
            mean.append(g.mean()); sd.append(g.std())
        x = np.arange(len(HORIZONS)) + (i - 1) * 0.26
        ax.bar(x, mean, 0.24, yerr=sd, color=HUE[key], label=VARIANTS[key][0], capsize=3)
    ax.axhline(0, color=COLORS["orange"] if False else COLORS["ink2"], linewidth=1)
    ax.set_xticks(range(len(HORIZONS)), [f"{h} step{'s' if h > 1 else ''}\n({h * 5} min)" for h in HORIZONS])
    ax.set_title(f"{title}: MAE change vs persistence"); ax.grid(axis="x", visible=False)
axes[0].set_ylabel("% better than persistence (higher is better)")
axes[0].legend(frameon=False, loc="upper left")
fig.tight_layout()
show(fig, "03d_horizons.png", "Change in MAE relative to persistence at the same horizon, mean of 3 seeds "
     "with one standard deviation. Positive: better than persistence.")

fig, ax = plt.subplots(figsize=(10, 3.8))
abl = [k for k in VARIANTS if k not in ("lstm", "cnn")]
for i, key in enumerate(abl):
    vals = [mbit(res[h][key][0][2]) / mbit(res[h]["cnn_lstm"][0][2]) * 100 - 100 for h in HORIZONS]
    ax.bar(np.arange(len(HORIZONS)) + (i - 1.5) * 0.2, vals, 0.19, label=VARIANTS[key][0],
           color=[COLORS["violet"], COLORS["aqua"], COLORS["orange"], COLORS["ink2"]][i])
ax.axhline(0, color=COLORS["ink2"], linewidth=1)
ax.set_xticks(range(len(HORIZONS)), [f"{h} step{'s' if h > 1 else ''}" for h in HORIZONS])
ax.set_ylabel("test MAE vs full CNN-LSTM, % (lower is better)")
ax.set_title("Ablations (seed 42): removing a part from the CNN-LSTM")
ax.legend(frameon=False, fontsize=8.5); ax.grid(axis="x", visible=False)
fig.tight_layout()
show(fig, "03d_ablations.png", "Test MAE of each ablated model relative to the full CNN-LSTM with the same seed "
     "(the first bar group member is the full model at 0%). One seed only: differences of about 1% are noise.")

# %% write the results page and open it in the browser
report.finish()
