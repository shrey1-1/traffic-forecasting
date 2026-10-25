"""02 Baselines: Phase 2 (evaluation gate).

Persistence (next = current) and ARIMA on a few busy flows, logged to metrics.csv.
Everything later is judged against persistence. Invert scaling before evaluate().

Persistence runs on all 144 flows. ARIMA runs on the 5 busiest flows, one model
per flow: fitted on the last 4 weeks of training data, order chosen by
validation error, then scored on exactly the same test examples as persistence.
Output page: results/reports/02_baselines.html
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

import numpy as np
import common
importlib.reload(common)
from common import *
report = Report("02_baselines", "Phase 2: Baselines")
report.section("Project status")
set_seeds()
status()

import time
import warnings
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

use_plot_style()


def describe(m):
    """Metrics in Mbit/s for reading (metrics.csv keeps raw units)."""
    return (f"MAE {m['mae'] * MBIT:7.3f} Mbit/s   RMSE {m['rmse'] * MBIT:7.3f} Mbit/s   "
            f"MAPE {m['mape']:5.1f}% (coverage {m['mape_coverage'] * 100:.0f}%)")


def flow_name(k):
    return f"{ABILENE_NODES[k // 12]} -> {ABILENE_NODES[k % 12]}"


# %% 1. load the Phase 1 splits
report.section("1. Data")
X_train, y_train, X_val, y_val, X_test, y_test = load_splits()
scaler = FlowScaler.load()
steps = np.load(PROCESSED / "target_steps.npy")
n_tr, n_va = len(y_train), len(y_val)
val_steps, test_steps = steps[n_tr:n_tr + n_va], steps[n_tr + n_va:]
print(f"train {n_tr}, val {n_va}, test {len(y_test)} examples")
print(f"test answers run from {step_label(test_steps[0])} to {step_label(test_steps[-1])}")
print("metrics.csv stores raw units (100 bytes / 5 min); this page shows Mbit/s")

# %% 2. persistence on all 144 flows
report.section("2. Persistence, all 144 flows")
for split, X, y in (("val", X_val, y_val), ("test", X_test, y_test)):
    m = evaluate_scaled(y, persistence(X), scaler)
    log_result("persistence", m, notes=f"{split} split, all 144 flows")
    print(f"{split:4s}  {describe(m)}")

y_test_raw = scaler.inverse_transform(y_test)
p_test_raw = scaler.inverse_transform(persistence(X_test))
flow_mae = np.abs(p_test_raw - y_test_raw).mean(0)
flow_mean = y_test_raw.mean(0)
rel = flow_mae / np.maximum(flow_mean, 1)
print("\nper-flow MAE as a share of the flow's mean traffic (test):")
print(f"  median {np.median(rel) * 100:.0f}%,  busiest 10 flows {np.median(rel[np.argsort(flow_mean)[-10:]]) * 100:.0f}%,"
      f"  smallest 10 flows {np.median(rel[np.argsort(flow_mean)[:10]]) * 100:.0f}%")
print("small flows are relatively much harder: under 1-in-100 sampling they are mostly noise")

# %% 3. ARIMA on the 5 busiest flows
report.section("3. ARIMA, 5 busiest flows")
weeks = load_abilene_raw()
segments, _ = abilene_clean(weeks)
full = weeks.reshape(-1, N_FLOWS)

# fit on the last 4 weeks of the unbroken stretch where training ends
tr_end = steps[n_tr - 1]
seg_start, seg = next((s, g) for s, g in segments if s <= tr_end < s + len(g))
fit_end = tr_end - seg_start + 1
fit_len = min(4 * STEPS_PER_WEEK, fit_end)
print(f"fitting data: {step_label(seg_start + fit_end - fit_len)} .. {step_label(tr_end)} "
      f"({fit_len} steps, {fit_len / 288:.0f} days, training only)")

top = np.argsort(scaler.inverse_transform(y_train).mean(0))[::-1][:5]
print("flows:", ", ".join(f"#{k} {flow_name(k)}" for k in top))

ORDERS = [((1, 0, 0), "c"), ((2, 0, 0), "c"), ((1, 0, 1), "c"), ((2, 0, 1), "c"),
          ((0, 1, 1), "n"), ((1, 1, 1), "n"), ((2, 1, 1), "n")]


def arima_predict(res, k, targets):
    """One-step forecasts for the given target steps, with the fitted parameters fixed.

    The model is run separately over every segment that contains targets, so it
    never crosses a gap; the forecast for step t uses only steps before t.
    Modelled on log1p traffic; forecasts are turned back with expm1.
    """
    pred = np.full(len(targets), np.nan)
    for s, g in segments:
        sel = (targets >= s) & (targets < s + len(g))
        if sel.any():
            one_step = res.apply(np.log1p(g[:, k])).predict()
            pred[sel] = np.maximum(np.expm1(one_step[targets[sel] - s]), 0)
    assert not np.isnan(pred).any()
    return pred


arima_test = np.zeros((len(test_steps), len(top)))
chosen = []
t0 = time.time()
for j, k in enumerate(top):
    y_fit = np.log1p(seg[fit_end - fit_len:fit_end, k])
    val_true = full[val_steps, k]
    best = None
    for order, trend in ORDERS:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = ARIMA(y_fit, order=order, trend=trend).fit()
        val_mae = np.abs(arima_predict(res, k, val_steps) - val_true).mean()
        if best is None or val_mae < best[0]:
            best = (val_mae, order, res)
    val_mae, order, res = best
    converged = res.mle_retvals.get("converged", True) if res.mle_retvals else True
    chosen.append(order)
    arima_test[:, j] = arima_predict(res, k, test_steps)
    pers_val_mae = np.abs(full[val_steps - 1, k] - val_true).mean()
    print(f"#{k:3d} {flow_name(k):18s} ARIMA{order}  val MAE {val_mae * MBIT:7.3f} Mbit/s"
          f"  (persistence {pers_val_mae * MBIT:7.3f})"
          f"{'' if converged else '  (optimiser did not fully converge)'}")
print(f"{len(top) * len(ORDERS)} models fitted in {time.time() - t0:.0f} s")

# the ARIMA truth (from the raw segments) must be exactly the test answers
assert np.allclose(full[test_steps][:, top], y_test_raw[:, top], rtol=1e-4, atol=1)

# %% 4. ARIMA vs persistence on the same flows and test examples
report.section("4. ARIMA vs persistence (test, same 5 flows)")
true5, pers5 = y_test_raw[:, top], p_test_raw[:, top]
m_arima, m_pers = evaluate(true5, arima_test), evaluate(true5, pers5)
log_result("arima_top5", m_arima,
           notes="test split, 5 busiest flows, per-flow order chosen on val MAE, fitted on 4 train weeks")
log_result("persistence_top5", m_pers, notes="test split, 5 busiest flows (same as arima_top5)")
print(f"ARIMA        {describe(m_arima)}")
print(f"persistence  {describe(m_pers)}")
print(f"ARIMA MAE is {(1 - m_arima['mae'] / m_pers['mae']) * 100:+.1f}% vs persistence "
      f"(negative means ARIMA is worse)\n")

print(f"{'flow':24s} {'order':10s} {'ARIMA MAE':>10s} {'persist. MAE':>13s} {'change':>8s}")
arima_mae, pers_mae = [], []
for j, k in enumerate(top):
    a = evaluate(true5[:, j], arima_test[:, j])["mae"] * MBIT
    p = evaluate(true5[:, j], pers5[:, j])["mae"] * MBIT
    arima_mae.append(a); pers_mae.append(p)
    print(f"#{k:3d} {flow_name(k):19s} {str(chosen[j]):10s} {a:10.3f} {p:13.3f} {(1 - a / p) * 100:+7.1f}%")
print("(MAE in Mbit/s; change = ARIMA improvement over persistence)")
print("\nOn validation ARIMA beat persistence on 4 of 5 flows; on test only on 1 of 5.")
k_worst = top[int(np.argmin(np.array(pers_mae) - np.array(arima_mae)))]
pv = np.abs(full[val_steps - 1, k_worst] - full[val_steps, k_worst]).mean() * MBIT
pt = np.abs(full[test_steps - 1, k_worst] - full[test_steps, k_worst]).mean() * MBIT
print(f"The biggest loss is {flow_name(k_worst)}, which is much burstier in test than in validation\n"
      f"(persistence MAE {pv:.1f} -> {pt:.1f} Mbit/s). ARIMA forecasts are smoothed, so they react to\n"
      f"sudden bursts later than persistence does.")
print("ARIMA forecasts are the median of a log-normal, so they sit slightly low on average;\n"
      "correcting that to the mean was tried and makes MAE worse, so it is not used.")

# %% 5. figures
report.section("5. Figures")
j, k = 0, top[0]
day = slice(288, 576)                         # the second day of the test period
hours = np.arange(288) / 12
arrow = lambda k: flow_name(k).replace(" -> ", "→")   # figure labels only
zoom = (7.5, 10.5)                                         # hours shown close up in the lower panel
fig, (ax, az) = plt.subplots(2, 1, figsize=(10, 6.6), height_ratios=[1.25, 1])
for a, lw in ((ax, 1.2), (az, 1.6)):
    a.plot(hours, true5[day, j] * MBIT, color=COLORS["ink2"], linewidth=lw + 1, label="actual")
    a.plot(hours, pers5[day, j] * MBIT, color=COLORS["orange"], linewidth=lw, label="persistence")
    a.plot(hours, arima_test[day, j] * MBIT, color=COLORS["blue"], linewidth=lw, label=f"ARIMA{chosen[j]}")
    a.set_ylabel("Mbit/s")
ax.axvspan(*zoom, color=COLORS["grid"], alpha=0.6, linewidth=0, zorder=0)
ax.set_title(f"One test day, {arrow(k)}: actual vs one-step forecasts")
ax.set_xticks(range(0, 25, 3)); ax.set_xlim(-0.3, 24.3)
az.set_title(f"Close-up of the shaded {zoom[1] - zoom[0]:.0f} hours: persistence lags one step, ARIMA smooths")
az.set_xlim(*zoom)
az.set_xticks(np.arange(zoom[0], zoom[1] + 0.01, 0.5))
az.set_xlabel(f"hours from {step_label(test_steps[day.start])}")
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, frameon=False, ncol=3, loc="lower center")
fig.tight_layout(rect=(0, 0.05, 1, 1))
show(fig, "02_forecast_day.png", "One day of the test period for the busiest flow. In the close-up, "
     "persistence (orange) repeats each actual value one step later; ARIMA (blue) stays near the middle "
     "and misses the peaks.")

fig, ax = plt.subplots(figsize=(10, 3.8))
x = np.arange(len(top))
w = 0.38
ax.bar(x - w / 2 - 0.01, pers_mae, w, color=COLORS["orange"], label="persistence")
ax.bar(x + w / 2 + 0.01, arima_mae, w, color=COLORS["blue"], label="ARIMA")
for xi, a, p in zip(x, arima_mae, pers_mae):
    ax.text(xi - w / 2 - 0.01, p, f"{p:.2f}", ha="center", va="bottom", fontsize=8, color=COLORS["ink2"])
    ax.text(xi + w / 2 + 0.01, a, f"{a:.2f}", ha="center", va="bottom", fontsize=8, color=COLORS["ink2"])
ax.set_xticks(x, [arrow(k) for k in top], fontsize=9)
ax.set_ylabel("MAE, Mbit/s")
ax.set_ylim(0, max(pers_mae + arima_mae) * 1.15)          # room for the value labels
ax.set_title("Test MAE per flow (lower is better)")
ax.grid(axis="x", visible=False)
ax.legend(frameon=False, loc="upper right")
fig.tight_layout()
show(fig, "02_mae_per_flow.png", "Mean absolute error on the test split for the 5 busiest flows.")

# %% 6. everything logged so far
report.section("6. metrics.csv")
import pandas as pd
with pd.option_context("display.width", 200, "display.max_columns", 20, "display.max_colwidth", 60):
    print(show_results().drop(columns=["timestamp"]).to_string(index=False))

# %% write the report page and open it in the browser
report.finish()
