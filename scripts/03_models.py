"""03 Models: Phases 3 to 5.

Phase 3 (this version): LSTM-only and CNN-only, each reported next to persistence,
in two forms: as planned (predict the next value directly) and residual
(predict last value + a learned correction; see build_* in common.py).
Phase 4 adds the CNN-LSTM and tuning; Phase 5 adds horizons 6 and 12 and ablations.

Models are trained on the scaled training split, stopped on the validation
split, and scored once on the test split in original units. Saved models are
reused on later runs; set RETRAIN = True to train again.
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
report = Report("03_models", "Phase 3: Single models")
report.section("Project status")
set_seeds()
status()

import matplotlib.pyplot as plt
use_plot_style()
RETRAIN = False          # True: train again even if a saved model exists


def describe(m):
    return (f"MAE {m['mae'] * MBIT:7.3f} Mbit/s   RMSE {m['rmse'] * MBIT:7.3f} Mbit/s   "
            f"MAPE {m['mape']:5.1f}% (coverage {m['mape_coverage'] * 100:.0f}%)")


def vs(m, base):
    """Change in MAE relative to a baseline; positive = better than the baseline."""
    return f"{(1 - m['mae'] / base['mae']) * 100:+.1f}%"


def flow_name(k):
    return f"{ABILENE_NODES[k // 12]} -> {ABILENE_NODES[k % 12]}"


arrow = lambda k: flow_name(k).replace(" -> ", "→")   # figure labels only

# %% 1. data and the persistence reference
report.section("1. Data and the persistence reference")
X_train, y_train, X_val, y_val, X_test, y_test = load_splits()
scaler = FlowScaler.load()
steps = np.load(PROCESSED / "target_steps.npy")
test_steps = steps[len(y_train) + len(y_val):]
print(f"train {len(y_train)}, val {len(y_val)}, test {len(y_test)} examples, input {X_train.shape[1:]}")

top = np.argsort(scaler.inverse_transform(y_train).mean(0))[::-1][:5]     # same 5 flows as Phase 2
y_val_raw, y_test_raw = scaler.inverse_transform(y_val), scaler.inverse_transform(y_test)
pers = {"val": evaluate_scaled(y_val, persistence(X_val), scaler),
        "test": evaluate_scaled(y_test, persistence(X_test), scaler),
        "test_top5": evaluate(y_test_raw[:, top], scaler.inverse_transform(persistence(X_test))[:, top])}
pers_val_mse = float(np.mean((persistence(X_val) - y_val) ** 2))
print(f"persistence val   {describe(pers['val'])}")
print(f"persistence test  {describe(pers['test'])}")
print(f"persistence val loss in scaled units (MSE, what the models are trained on): {pers_val_mse:.5f}")

# %% 2. train (or load) the models
MODELS_P3 = [  # (file name, label, builder)
    ("lstm", "LSTM-only", build_lstm),
    ("cnn", "CNN-only", build_cnn),
    ("lstm_res", "LSTM-only, residual", lambda: build_lstm(residual=True)),
    ("cnn_res", "CNN-only, residual", lambda: build_cnn(residual=True)),
]
results, preds, histories = {}, {}, {}
for name, label, build in MODELS_P3:
    report.section(f"2. {label}")
    model, hist, trained = fit_or_load(build, name, X_train, y_train, X_val, y_val, retrain=RETRAIN)
    print(f"  {model.count_params():,} parameters")
    histories[label] = hist
    p_val, p_test = predict(model, X_val), predict(model, X_test)
    preds[label] = scaler.inverse_transform(p_test)
    m_val = evaluate_scaled(y_val, p_val, scaler)
    m_test = evaluate_scaled(y_test, p_test, scaler)
    m_top = evaluate(y_test_raw[:, top], preds[label][:, top])
    results[label] = {"val": m_val, "test": m_test, "test_top5": m_top}
    note = f"best epoch {int(np.argmin(hist['val_loss'])) + 1} of {len(hist['loss'])}"
    log_result(name, m_val, notes=f"val split, all 144 flows, {note}")
    log_result(name, m_test, notes=f"test split, all 144 flows, {note}")
    log_result(f"{name}_top5", m_top, notes="test split, 5 busiest flows (same as arima_top5)")
    print(f"val   {describe(m_val)}   vs persistence {vs(m_val, pers['val'])}")
    print(f"test  {describe(m_test)}   vs persistence {vs(m_test, pers['test'])}")

# %% 3. comparison
report.section("3. Comparison with the baselines")
arima = show_results().query("name == 'arima_top5'").iloc[-1]
W = 21
print("All 144 flows (MAE in Mbit/s; change: positive = better than persistence)")
print(f"  {'':{W}s} {'val MAE':>8s} {'change':>8s}   {'test MAE':>8s} {'change':>8s}   test MAPE (coverage)")
print(f"  {'persistence':{W}s} {pers['val']['mae'] * MBIT:8.3f} {'':8s}   {pers['test']['mae'] * MBIT:8.3f} {'':8s}   "
      f"{pers['test']['mape']:.1f}% ({pers['test']['mape_coverage'] * 100:.0f}%)")
for label, r in results.items():
    print(f"  {label:{W}s} {r['val']['mae'] * MBIT:8.3f} {vs(r['val'], pers['val']):>8s}   "
          f"{r['test']['mae'] * MBIT:8.3f} {vs(r['test'], pers['test']):>8s}   "
          f"{r['test']['mape']:.1f}% ({r['test']['mape_coverage'] * 100:.0f}%)")
print("\n5 busiest flows, test split")
print(f"  {'persistence':{W}s} {describe(pers['test_top5'])}")
print(f"  {'ARIMA':{W}s} {describe(arima)}   {vs(arima, pers['test_top5'])}")
for label, r in results.items():
    print(f"  {label:{W}s} {describe(r['test_top5'])}   {vs(r['test_top5'], pers['test_top5'])}")

# per flow: how many flows does each model beat persistence on?
p_test_raw = scaler.inverse_transform(persistence(X_test))
mae_pers_flow = np.abs(p_test_raw - y_test_raw).mean(0)
size = y_test_raw.mean(0)
quart = np.digitize(size, np.quantile(size, [0.25, 0.5, 0.75]))
print("\nFlows where the model's test MAE is lower than persistence's (by flow size quarter):")
print(f"  {'':{W}s} {'all':>8s} {'smallest':>9s} {'2nd':>6s} {'3rd':>6s} {'largest':>8s}")
for label in results:
    win = np.abs(preds[label] - y_test_raw).mean(0) < mae_pers_flow
    print(f"  {label:{W}s} {win.sum():4d}/144 " + " ".join(f"{win[quart == q].sum():5d}/{(quart == q).sum()}" for q in range(4)))

# %% 4. figures
report.section("4. Figures")
HUE = {"LSTM": COLORS["aqua"], "CNN": COLORS["violet"]}
hue = lambda label: HUE[label.split("-")[0]]
plain = lambda label: "residual" not in label

fig, axes = plt.subplots(2, 2, figsize=(11, 6.4), sharey=True)
for ax, (label, h) in zip(axes.ravel(), histories.items()):
    ep = np.arange(1, len(h["loss"]) + 1)
    ax.plot(ep, h["loss"], color=COLORS["ink2"], label="training")
    ax.plot(ep, h["val_loss"], color=hue(label), label="validation")
    best = int(np.argmin(h["val_loss"]))
    ax.plot([best + 1], [h["val_loss"][best]], marker="o", markersize=8, color=hue(label),
            linestyle="none", label=f"best (epoch {best + 1})")
    ax.axhline(pers_val_mse, color=COLORS["orange"], linewidth=1.2, linestyle="--",
               label="persistence (val)")
    ax.set_title(f"{label}: loss per epoch")
    ax.set_yscale("log")
    ax.legend(frameon=False, loc="upper right", fontsize=8.5)
for ax in axes[1]:
    ax.set_xlabel("epoch")
for ax in axes[:, 0]:
    ax.set_ylabel("MSE, scaled units (log axis)")
fig.tight_layout()
show(fig, "03_learning_curves.png", "Training and validation loss in the units the models are trained on. "
     "The dashed line is persistence's validation loss in the same units.")

labels = ["persistence"] + list(results)
test_mae = [pers["test"]["mae"] * MBIT] + [results[l]["test"]["mae"] * MBIT for l in results]
fig, ax = plt.subplots(figsize=(10, 3.4))
yy = np.arange(len(labels))[::-1]
for yi, l, v in zip(yy, labels, test_mae):
    is_plain = l != "persistence" and plain(l)
    ax.barh(yi, v, 0.62, color=COLORS["orange"] if l == "persistence" else hue(l),
            alpha=0.45 if is_plain else 1.0, hatch="//" if is_plain else None, edgecolor="white")
    ax.text(v, yi, f"  {v:.3f}", va="center", fontsize=9, color=COLORS["ink2"])
ax.set_yticks(yy, labels)
ax.axvline(test_mae[0], color=COLORS["orange"], linewidth=1, linestyle="--")
ax.set_xlim(0, max(test_mae) * 1.15)
ax.set_xlabel("test MAE, Mbit/s (all 144 flows; lower is better)")
ax.set_title("Test MAE: as planned (hatched) vs residual (solid)")
ax.grid(axis="y", visible=False)
fig.tight_layout()
show(fig, "03_test_mae.png", "Test MAE over all 144 flows. The dashed line marks persistence.")

k = top[0]
day = slice(288, 576)
hours = np.arange(288) / 12
zoom = (7.5, 10.5)
fig, ax = plt.subplots(figsize=(10, 3.8))
ax.plot(hours, y_test_raw[day, k] * MBIT, color=COLORS["ink2"], linewidth=2.6, label="actual")
ax.plot(hours, p_test_raw[day, k] * MBIT, color=COLORS["orange"], linewidth=1.4, label="persistence")
for label in results:
    ax.plot(hours, preds[label][day, k] * MBIT, color=hue(label), linewidth=1.3 if plain(label) else 1.8,
            linestyle="--" if plain(label) else "-", label=label)
ax.set_xlim(*zoom); ax.set_xticks(np.arange(zoom[0], zoom[1] + 0.01, 0.5))
ax.set_title(f"Same 3 test hours as Phase 2, {arrow(k)}: one-step forecasts")
ax.set_xlabel(f"hours from {step_label(test_steps[day.start])}"); ax.set_ylabel("Mbit/s")
ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize=9)
fig.tight_layout()
show(fig, "03_forecast_closeup.png", "The 3-hour window from Phase 2's close-up. Dashed: as planned; "
     "solid: residual. The residual models lie almost exactly on persistence (orange, hidden underneath): "
     "they learned only small corrections to the last value.")

bars = {"persistence": mae_pers_flow[top] * MBIT}
for label in results:
    if not plain(label):
        bars[label] = np.abs(preds[label][:, top] - y_test_raw[:, top]).mean(0) * MBIT
fig, ax = plt.subplots(figsize=(10.5, 3.9))
x = np.arange(len(top)); w = 0.27
for i, (label, v) in enumerate(bars.items()):
    xs = x + (i - 1) * (w + 0.02)
    ax.bar(xs, v, w, color=COLORS["orange"] if label == "persistence" else hue(label), label=label)
    for xi, vi in zip(xs, v):
        ax.text(xi, vi, f"{vi:.1f}", ha="center", va="bottom", fontsize=7.5, color=COLORS["ink2"])
ax.set_xticks(x, [arrow(k) for k in top], fontsize=9)
ax.set_ylabel("MAE, Mbit/s"); ax.grid(axis="x", visible=False)
ax.set_ylim(0, max(v.max() for v in bars.values()) * 1.15)
ax.set_title("Test MAE per flow, 5 busiest flows: persistence vs residual models (lower is better)")
ax.legend(frameon=False, loc="upper right", ncol=3)
fig.tight_layout()
show(fig, "03_mae_per_flow.png", "Test MAE for the 5 busiest flows.")

# %% write the results page and open it in the browser
report.finish()
