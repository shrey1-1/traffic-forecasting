"""03b Hybrid: Phase 4, the CNN-LSTM and its training loss.

Step 1 (this version): the planned CNN-LSTM in both forms (as planned and
residual), then the open question from Phase 3 (section 13.5): does training on
a loss closer to the reported metric help? The residual LSTM and CNN-LSTM are
trained with three losses: MSE on the scaled data (the plan), MAE on the scaled
data, and MAE in original units (traffic in Mbit/s, what the report scores).

Models are chosen on the VALIDATION split only. Test numbers are logged and
shown for the record, never used to pick anything.
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
report = Report("03b_hybrid", "Phase 4: CNN-LSTM")
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


# %% 1. data and the persistence reference
report.section("1. Data and the persistence reference")
X_train, y_train, X_val, y_val, X_test, y_test = load_splits()
scaler = FlowScaler.load()
top = np.argsort(scaler.inverse_transform(y_train).mean(0))[::-1][:5]     # same 5 flows as Phase 2
y_test_raw = scaler.inverse_transform(y_test)
pers = {"val": evaluate_scaled(y_val, persistence(X_val), scaler),
        "test": evaluate_scaled(y_test, persistence(X_test), scaler),
        "test_top5": evaluate(y_test_raw[:, top], scaler.inverse_transform(persistence(X_test))[:, top])}
print(f"train {len(y_train)}, val {len(y_val)}, test {len(y_test)} examples")
print(f"persistence val   {describe(pers['val'])}")
print(f"persistence test  {describe(pers['test'])}")

# mean training traffic: divides the original-units loss so it is of order 1
REF = float(scaler.inverse_transform(y_train).mean())
LOSSES = {"mse": make_loss("mse"), "mae": make_loss("mae"),
          "mae_raw": make_loss("mae_raw", scaler, REF)}
print(f"mean training traffic {REF:,.1f} (100-byte units per 5 min) = {REF * MBIT:.2f} Mbit/s")

# %% 2. train (or load) the models
RUNS = [  # (file name, label, builder, loss)
    ("cnn_lstm", "CNN-LSTM", lambda: build_cnn_lstm(), "mse"),
    ("cnn_lstm_res", "CNN-LSTM, residual", lambda: build_cnn_lstm(residual=True), "mse"),
    ("cnn_lstm_res_mae", "CNN-LSTM, residual, MAE loss",
     lambda: build_cnn_lstm(residual=True), "mae"),
    ("cnn_lstm_res_maeraw", "CNN-LSTM, residual, original-units MAE loss",
     lambda: build_cnn_lstm(residual=True), "mae_raw"),
    ("lstm_res_maeraw", "LSTM-only, residual, original-units MAE loss",
     lambda: build_lstm(residual=True), "mae_raw"),
]
results, preds, histories = {}, {}, {}
for name, label, build, loss in RUNS:
    report.section(f"2. {label}")
    model, hist, trained = fit_or_load(build, name, X_train, y_train, X_val, y_val,
                                       retrain=RETRAIN, loss=LOSSES[loss])
    print(f"  {model.count_params():,} parameters, loss: {loss}")
    histories[label] = hist
    p_val, p_test = predict(model, X_val), predict(model, X_test)
    preds[label] = scaler.inverse_transform(p_test)
    m_val = evaluate_scaled(y_val, p_val, scaler)
    m_test = evaluate_scaled(y_test, p_test, scaler)
    m_top = evaluate(y_test_raw[:, top], preds[label][:, top])
    results[label] = {"val": m_val, "test": m_test, "test_top5": m_top}
    note = f"loss {loss}, best epoch {int(np.argmin(hist['val_loss'])) + 1} of {len(hist['loss'])}"
    log_result(name, m_val, notes=f"val split, all 144 flows, {note}")
    log_result(name, m_test, notes=f"test split, all 144 flows, {note}")
    log_result(f"{name}_top5", m_top, notes="test split, 5 busiest flows (same as arima_top5)")
    print(f"val   {describe(m_val)}   vs persistence {vs(m_val, pers['val'])}")
    print(f"test  {describe(m_test)}   vs persistence {vs(m_test, pers['test'])}")

# %% 3. comparison (ranked on validation)
report.section("3. Comparison")
W = 46
print("All 144 flows, MAE in Mbit/s; change: positive = better than persistence")
print("Ranked on VALIDATION; test shown for the record only.")
print(f"  {'':{W}s} {'val MAE':>8s} {'change':>8s}   {'test MAE':>8s} {'change':>8s}")
print(f"  {'persistence':{W}s} {pers['val']['mae'] * MBIT:8.3f} {'':8s}   {pers['test']['mae'] * MBIT:8.3f}")
for label, r in sorted(results.items(), key=lambda kv: kv[1]["val"]["mae"]):
    print(f"  {label:{W}s} {r['val']['mae'] * MBIT:8.3f} {vs(r['val'], pers['val']):>8s}   "
          f"{r['test']['mae'] * MBIT:8.3f} {vs(r['test'], pers['test']):>8s}")

# %% 4. figures
report.section("4. Figures")
labels = ["persistence"] + list(results)
val_mae = [pers["val"]["mae"] * MBIT] + [results[l]["val"]["mae"] * MBIT for l in results]
fig, ax = plt.subplots(figsize=(10, 3.8))
yy = np.arange(len(labels))[::-1]
for yi, l, v in zip(yy, labels, val_mae):
    ax.barh(yi, v, 0.62, color=COLORS["orange"] if l == "persistence" else COLORS["aqua"], edgecolor="white")
    ax.text(v, yi, f"  {v:.3f}", va="center", fontsize=9, color=COLORS["ink2"])
ax.set_yticks(yy, labels, fontsize=8.5)
ax.axvline(val_mae[0], color=COLORS["orange"], linewidth=1, linestyle="--")
ax.set_xlim(0, max(val_mae) * 1.15)
ax.set_xlabel("validation MAE, Mbit/s (all 144 flows; lower is better)")
ax.set_title("Phase 4, step 1: validation MAE by model and training loss")
ax.grid(axis="y", visible=False)
fig.tight_layout()
show(fig, "03b_val_mae.png", "Validation MAE over all 144 flows. The dashed line marks persistence.")

# %% write the results page and open it in the browser
report.finish()
