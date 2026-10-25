"""05 Errors: Phase 7, where do the forecasts go wrong?

Tests the over-smoothing idea from Phases 2, 3 and 5: do the models gain on
ordinary 5-minute wobble and lose on the big moves and peaks that matter?
Uses the same forecasts as Phases 5 and 6 (CNN-LSTM, mean of 3 seeds; LSTM-only
too on Abilene), on the test split, for Abilene (1, 6, 12 steps) and GÉANT
(1, 4 steps). Everything is in original units, pooled over flows, so busy flows
count more, as in MAE.

Definitions (all fixed here, none tuned on test):
 move      |true - last observed value| / the flow's mean training traffic
 big move  the top 5% of moves (95th percentile of the pooled test moves)
 peak      a target above the flow's own 95th percentile of training targets
 damping   slope of (forecast - last value) on (true - last value) through the
           origin: 1 = the move is fully forecast, 0 = persistence, <0 = wrong way
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
import tensorflow as tf
import common
importlib.reload(common)
from common import *
report = Report("05_errors", "Phase 7: Error analysis")
report.section("Project status")
set_seeds()
status()

import matplotlib.pyplot as plt
use_plot_style()
SEEDS = (42, 43, 44)
CASES = [("Abilene", 1), ("Abilene", 6), ("Abilene", 12), ("GÉANT", 1), ("GÉANT", 4)]
UNIT = {"Abilene": MBIT, "GÉANT": 1 / 1000}                     # raw units -> Mbit/s
STEP_MIN = {"Abilene": 5, "GÉANT": 15}
LOSA_CHIN = 7 * 12 + 2                                          # Abilene flow 86
BIN_EDGES = (0, 50, 80, 95, 99, 100)                            # percentiles of the move size
BIN_NAMES = ("smallest 50%", "50-80%", "80-95%", "95-99%", "top 1%")


def model_files(ds, h, key):
    if ds == "Abilene":
        if h == 1:
            return [f"p4_{key}_f64_k3_u100-50_d20_lr0.001_s{s}" if key == "cnn_lstm"
                    else f"p4_{key}_u100-50_d20_lr0.001_s{s}" for s in SEEDS]
        return [f"p5_h{h}_{key}_s{s}" for s in SEEDS]
    return [f"g_h{h}_{key}_s{s}" for s in SEEDS]


def load_case(ds, h, keys=("cnn_lstm",)):
    d = (PROCESSED if h == 1 else PROCESSED / f"h{h}") if ds == "Abilene" else PROCESSED / "geant" / f"h{h}"
    X_tr, y_tr, X_va, y_va, X_te, y_te = load_splits(d)
    sc = FlowScaler.load(d / "scaler.npz")
    cap = None if ds == "Abilene" else GEANT_MAX_FLOW_KBPS
    out = {"y": sc.inverse_transform(y_te).astype(np.float64), "last": sc.inverse_transform(persistence(X_te)).astype(np.float64),
           "y_train": sc.inverse_transform(y_tr).astype(np.float64),
           "steps": np.load(d / "target_steps.npy")[len(y_tr) + len(y_va):]}
    for key in keys:
        ps = [sc.inverse_transform(predict(tf.keras.models.load_model(MODELS / f"{n}.keras", compile=False),
                                           X_te, scaler=sc, max_raw=cap)).astype(np.float64)
              for n in model_files(ds, h, key)]
        out[key] = np.mean(ps, axis=0)
    return out


def damping(pred, last, y, mask=None):
    a, b = (pred - last), (y - last)
    if mask is not None:
        a, b = a[mask], b[mask]
    return float((a * b).sum() / (b * b).sum())


# %% 1. load the forecasts
report.section("1. Forecasts")
C = {}
for ds, h in CASES:
    C[(ds, h)] = load_case(ds, h, ("cnn_lstm", "lstm") if ds == "Abilene" else ("cnn_lstm",))
    c = C[(ds, h)]
    print(f"{ds} h={h}: {c['y'].shape[0]} test examples x {c['y'].shape[1]} flows")

# %% 2. gains and losses by size of the true move
report.section("2. Where the CNN-LSTM gains and loses")
rows = {}
for (ds, h), c in C.items():
    scale = c["y_train"].mean(0)
    ok = scale > 0
    move = np.abs(c["y"] - c["last"])[:, ok] / scale[ok]
    edges = np.percentile(move, BIN_EDGES)
    which = np.clip(np.searchsorted(edges, move, side="right") - 1, 0, len(BIN_NAMES) - 1)
    e_p = np.abs(c["last"] - c["y"])[:, ok]
    e_m = np.abs(c["cnn_lstm"] - c["y"])[:, ok]
    u = UNIT[ds]
    print(f"\n{ds}, {h * STEP_MIN[ds]} min ahead. MAE in Mbit/s over all points in the bin; change: positive = model better")
    print(f"  {'size of the true move':22s} {'persistence':>11s} {'CNN-LSTM':>9s} {'change':>8s}   share of all error (persistence)")
    rows[(ds, h)] = []
    tot = e_p.sum()
    for b, name in enumerate(BIN_NAMES):
        m = which == b
        mp, mm = e_p[m].mean() * u, e_m[m].mean() * u
        rows[(ds, h)].append((name, mp, mm, (1 - mm / mp) * 100, e_p[m].sum() / tot))
        print(f"  {name:22s} {mp:11.3f} {mm:9.3f} {(1 - mm / mp) * 100:+7.1f}%   {e_p[m].sum() / tot * 100:5.1f}%")
    print(f"  {'all points':22s} {e_p.mean() * u:11.3f} {e_m.mean() * u:9.3f} {(1 - e_m.sum() / e_p.sum()) * 100:+7.1f}%")

# %% 3. damping of moves, and peaks
report.section("3. Damping and peaks")
print("damping = slope of (forecast - last value) on (true - last value): 1 = move fully forecast, 0 = persistence")
print("peak = target above the flow's own 95th percentile of training targets\n")
print(f"  {'':16s} {'damping':>8s} {'damping':>11s} | {'peak bias (forecast - true) / true':>36s} | {'peak MAE / mean true':>24s}")
print(f"  {'':16s} {'all':>8s} {'big moves':>11s} | {'persistence':>14s} {'CNN-LSTM':>10s} | {'persistence':>12s} {'CNN-LSTM':>10s}")
stats = {}
for (ds, h), c in C.items():
    scale = c["y_train"].mean(0)
    ok = scale > 0
    y, last, pred = c["y"][:, ok], c["last"][:, ok], c["cnn_lstm"][:, ok]
    move = np.abs(y - last) / scale[ok]
    big = move >= np.percentile(move, 95)
    peak = y >= np.percentile(c["y_train"][:, ok], 95, axis=0)
    d_all, d_big = damping(pred, last, y), damping(pred, last, y, big)
    bias = lambda f: (f[peak] - y[peak]).sum() / y[peak].sum() * 100
    pmae = lambda f: np.abs(f[peak] - y[peak]).sum() / y[peak].sum() * 100
    stats[(ds, h)] = (d_all, d_big)
    print(f"  {ds + ' ' + str(h * STEP_MIN[ds]) + ' min':16s} {d_all:8.3f} {d_big:11.3f} | {bias(last):13.1f}% {bias(pred):9.1f}% | "
          f"{pmae(last):11.1f}% {pmae(pred):9.1f}%")
print("\n(peak bias is negative when the forecast is below the true peak. Peaks are chosen on the target, so any forecast built from")
print("earlier values is below them on average; the comparison that matters is CNN-LSTM against persistence.)")

# %% 4. how much does one flow decide the Abilene result?
report.section("4. The LOSA to CHIN flow on Abilene")
print(f"Test MAE in Mbit/s over all 144 flows, and without the {ABILENE_NODES[7]}->{ABILENE_NODES[2]} flow (flow {LOSA_CHIN}).")
print(f"  {'':10s} {'':10s} {'persistence':>12s} {'CNN-LSTM':>9s} {'change':>8s} | {'LSTM-only':>9s} {'change':>8s}")
for h in (1, 6, 12):
    c = C[("Abilene", h)]
    keep = np.arange(144) != LOSA_CHIN
    for label, sel in (("all 144", slice(None)), ("without it", keep)):
        mp = np.abs(c["last"][:, sel] - c["y"][:, sel]).mean() * MBIT
        mc = np.abs(c["cnn_lstm"][:, sel] - c["y"][:, sel]).mean() * MBIT
        ml = np.abs(c["lstm"][:, sel] - c["y"][:, sel]).mean() * MBIT
        print(f"  h={h:2d} {label:10s} {mp:12.3f} {mc:9.3f} {(1 - mc / mp) * 100:+7.1f}% | {ml:9.3f} {(1 - ml / mp) * 100:+7.1f}%")
c = C[("Abilene", 6)]
share = np.abs(c["last"] - c["y"]).sum(0)
print(f"\nThe flow carries {share[LOSA_CHIN] / share.sum() * 100:.1f}% of persistence's total test error at 30 minutes "
      f"(1 of 144 flows), and its largest values reach {c['y'][:, LOSA_CHIN].max() * MBIT:,.0f} Mbit/s against a mean of "
      f"{c['y'][:, LOSA_CHIN].mean() * MBIT:,.0f}.")

# %% 5. figures
report.section("5. Figures")
cols = {("Abilene", 1): COLORS["aqua"], ("Abilene", 6): COLORS["violet"], ("Abilene", 12): COLORS["orange"],
        ("GÉANT", 1): COLORS["ink2"], ("GÉANT", 4): "red"}
fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
x = np.arange(len(BIN_NAMES))
for k, (case, r) in enumerate(rows.items()):
    axes[0].plot(x, [v[3] for v in r], marker="o", color=cols[case], label=f"{case[0]} {case[1] * STEP_MIN[case[0]]} min")
    axes[1].plot(x, [v[4] * 100 for v in r], marker="o", color=cols[case])
axes[0].axhline(0, color=COLORS["ink2"], linewidth=1)
axes[0].set_ylabel("MAE change vs persistence, % (positive = model better)")
axes[0].set_title("Gain by size of the true move")
axes[1].set_ylabel("share of persistence's total error, %"); axes[1].set_title("Where the error is")
for ax in axes:
    ax.set_xticks(x, BIN_NAMES, fontsize=8.5, rotation=15); ax.set_xlabel("true 'move' size, percentile of all test points")
axes[0].legend(frameon=False, fontsize=8.5)
fig.tight_layout()
show(fig, "05_move_bins.png", "Left: the CNN-LSTM's gain over persistence by how much the true value moved from the "
     "last observed one. Right: where persistence's error comes from. The biggest moves are few points but a large part of the error.")

fig, ax = plt.subplots(figsize=(7.5, 4.2))
for case, c in C.items():
    scale = c["y_train"].mean(0); ok = scale > 0
    tm = ((c["y"] - c["last"])[:, ok] / scale[ok]).ravel()
    pm = ((c["cnn_lstm"] - c["last"])[:, ok] / scale[ok]).ravel()
    qs = np.quantile(tm, np.linspace(0, 1, 41)); mid, val = [], []
    for lo, hi in zip(qs[:-1], qs[1:]):
        m = (tm >= lo) & (tm <= hi)
        mid.append(tm[m].mean()); val.append(pm[m].mean())
    ax.plot(mid, val, color=cols[case], label=f"{case[0]} {case[1] * STEP_MIN[case[0]]} min", linewidth=1.5)
lim = 1.0
ax.plot([-lim, lim], [-lim, lim], color=COLORS["ink2"], linestyle="--", linewidth=1, label="perfect (slope 1)")
ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
ax.set_xlabel("true change from the last value (fraction of the flow's mean)"); ax.set_ylabel("forecast change")
ax.set_title("Forecast change against true change (binned means)")
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
show(fig, "05_damping.png", "Average forecast change for each level of true change. A line flatter than the dashed "
     "one means the model forecasts only part of each move.")

c = C[("Abilene", 6)]
k = LOSA_CHIN
j = int(np.argmax(c["y"][:, k]))
sl = slice(max(j - 30, 0), j + 40)
hrs = (np.arange(len(c["y"]))[sl] - j) * 5 / 60
fig, ax = plt.subplots(figsize=(10.5, 3.8))
ax.plot(hrs, c["y"][sl, k] * MBIT, color=COLORS["ink2"], linewidth=2.4, label="actual")
ax.plot(hrs, c["last"][sl, k] * MBIT, color=COLORS["orange"], linewidth=1.3, label="persistence (value 30 min earlier)")
ax.plot(hrs, c["cnn_lstm"][sl, k] * MBIT, color=COLORS["violet"], linewidth=1.6, label="CNN-LSTM, 30 min ahead")
ax.plot(hrs, c["lstm"][sl, k] * MBIT, color=COLORS["aqua"], linewidth=1.3, linestyle="--", label="LSTM-only, 30 min ahead")
ax.set_xlabel("hours from the largest value"); ax.set_ylabel("Mbit/s")
ax.set_title(f"{ABILENE_NODES[7]} to {ABILENE_NODES[2]}, test: the largest burst, forecasts made 30 minutes earlier")
ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize=8.5)
fig.tight_layout()
show(fig, "05_burst.png", "The flow behind most of Abilene's large errors and congestion events. Every forecast "
     "line is the forecast for that moment, made 30 minutes before it.")

# %% write the results page and open it in the browser
report.finish()
