"""03e GÉANT: Phase 6, does it generalise to a second network?

GÉANT (European research backbone, TOTEM dataset): 23 nodes = 529 flows, one
matrix every 15 minutes, 1 Jan to 29 Apr 2005. The models are the Phase 4/5
ones with exactly the same settings (residual, original-units MAE loss, Phase 4
hyper-parameters, window 12), retrained on GÉANT with no retuning. They are
NOT applied to GÉANT without retraining: the flow counts differ.
Horizons: 1 step (15 min) and 4 steps (60 min, the same wall-clock time as
Abilene's 12 steps).

Link level (section 5) is approximate: the anonymised topology has no IGP
weights and no capacities, so flows are routed over minimum-hop paths with equal
splitting, and "high load" means a link's load is in the top 10% of its own
training-period loads. This is NOT congestion against capacity.

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
report = Report("03e_geant", "Phase 6: GÉANT")
report.section("Project status")
set_seeds()
status()

import matplotlib.pyplot as plt
use_plot_style()
RETRAIN = False          # True: train again even if a saved model exists
EPOCHS, PATIENCE = 200, 15
HORIZONS = (1, 4)
SEEDS = (42, 43, 44)
LR = 1e-3
CAP = GEANT_MAX_FLOW_KBPS
mb = lambda m: m["mae"] / 1000          # kbit/s -> Mbit/s
STEP_MIN = GEANT_STEP_S // 60

VARIANTS = {   # key: (label, builder)
    "cnn_lstm": ("CNN-LSTM", lambda: build_cnn_lstm(n_flows=GEANT_FLOWS, residual=True)),
    "lstm": ("LSTM-only", lambda: build_lstm(n_flows=GEANT_FLOWS, residual=True)),
    "cnn": ("CNN-only", lambda: build_cnn(n_flows=GEANT_FLOWS, residual=True)),
}


def describe(m):
    return (f"MAE {mb(m):8.3f} Mbit/s   RMSE {m['rmse'] / 1000:8.3f} Mbit/s   "
            f"MAPE {m['mape']:5.1f}% (coverage {m['mape_coverage'] * 100:.0f}%)")


# %% 1. the data and its cleaning
report.section("1. The data and its cleaning")
raw = load_geant_raw()
segments, info = geant_clean(raw)
print(f"{info['total']} slots of 15 minutes (1 Jan to 29 Apr 2005); kept {info['kept']} "
      f"({info['kept'] / info['total'] * 100:.1f}%)")
print("removed:", info["removed"])
print(f"  no_file    no matrix for that slot (8 gaps; the longest is 456 slots = 4.75 days)\n"
      f"  incomplete more than {GEANT_MAX_ABSENT:.0%} of the usual flows absent (collector failure)\n"
      f"  impossible a flow above {CAP / 1e6:.0f} Gbit/s (the core link speed); e.g. one slot has flows of hundreds of Tbit/s\n"
      f"  outage     network total below {GEANT_MIN_TOTAL:.0%} of the median")
lens = sorted([len(s) for _, s in segments], reverse=True)
print(f"{len(segments)} segments of consecutive steps, longest {lens[0]} (= {lens[0] / 96:.1f} days), shortest {lens[-1]}")
live = ~np.isnan(raw[~np.isnan(raw).all(1)]).all(0)
print(f"{(~live).sum()} of 529 flows never appear in any file (all zero); pairs a file leaves out are set to 0:\n"
      f"  they are mostly small flows and no listed value is ever exactly 0, so absence = traffic below the measurement floor")

x = np.nan_to_num(raw)
total_gbps = x.sum(1) / 1e6
reason = info["reason"]
fig, ax = plt.subplots(figsize=(10.5, 3.6))
okm = reason == ""
ax.plot(np.arange(len(raw))[okm] / 96, np.where(okm, total_gbps, np.nan)[okm], ".", markersize=1.2, color=COLORS["aqua"])
for r, col in (("no_file", COLORS["ink2"]), ("incomplete", COLORS["orange"]), ("impossible", "red"), ("outage", COLORS["violet"])):
    m = reason == r
    if m.any():
        ax.vlines(np.where(m)[0] / 96, 0, 20, color=col, alpha=0.35 if r == "no_file" else 0.9, linewidth=1.2,
                  label=f"{r} ({m.sum()})")
ax.set_ylim(0, 20); ax.set_xlabel("days from 1 Jan 2005"); ax.set_ylabel("total traffic, Gbit/s")
ax.set_title("GÉANT: total traffic per 15-minute slot, with the removed slots marked")
ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.22), fontsize=8.5)
fig.tight_layout()
show(fig, "03e_data.png", "Kept slots (dots) and removed slots (vertical lines, by rule). The y axis is cut at "
     "20 Gbit/s; the removed 'impossible' slots are far above it.")

# daily cycle exit check
hours = (np.arange(len(raw)) % 96) / 4
tod = np.array([total_gbps[okm & (np.floor(hours) == h)].mean() for h in range(24)])
print(f"\ndaily cycle (whole-network mean by hour, UTC assumed): lowest {tod.min():.2f} Gbit/s at {tod.argmin():02d}:00, "
      f"highest {tod.max():.2f} at {tod.argmax():02d}:00 ({tod.max() / tod.min():.2f}x)")

# %% 2. train (or load) everything, one horizon at a time
res, ctx = {}, {}
for h in HORIZONS:
    report.section(f"2. Horizon {h} ({h * STEP_MIN} minutes ahead)")
    d = PROCESSED / "geant" / f"h{h}"
    if not (d / "scaler.npz").exists():
        prepare_geant(horizon=h, directory=d)
    X_train, y_train, X_val, y_val, X_test, y_test = load_splits(d)
    scaler = FlowScaler.load(d / "scaler.npz")
    steps = np.load(d / "target_steps.npy")
    loss = make_loss("mae_raw", scaler, float(scaler.inverse_transform(y_train).mean()))
    res[h] = {"persistence": {"val": evaluate_scaled(y_val, persistence(X_val), scaler),
                              "test": evaluate_scaled(y_test, persistence(X_test), scaler)}}
    pv, pt = res[h]["persistence"]["val"], res[h]["persistence"]["test"]
    print(f"train {len(y_train)}, val {len(y_val)}, test {len(y_test)} examples")
    print(f"test period: {step_label(steps[len(y_train) + len(y_val)]).replace('X', 'slot ')} to the end "
          f"(slots {steps[len(y_train) + len(y_val)]} to {steps[-1]} of 11396)")
    print(f"persistence val   {describe(pv)}")
    print(f"persistence test  {describe(pt)}")
    test_preds = {k: [] for k in VARIANTS}
    for key, (label, build) in VARIANTS.items():
        res[h][key] = []
        for seed in SEEDS:
            name = f"g_h{h}_{key}_s{seed}"
            model, hist, _ = fit_or_load(build, name, X_train, y_train, X_val, y_val, retrain=RETRAIN,
                                         seed=seed, epochs=EPOCHS, patience=PATIENCE, lr=LR, loss=loss)
            p_val = predict(model, X_val, scaler=scaler, max_raw=CAP)
            p_test = predict(model, X_test, scaler=scaler, max_raw=CAP)
            mv, mt = evaluate_scaled(y_val, p_val, scaler), evaluate_scaled(y_test, p_test, scaler)
            test_preds[key].append(scaler.inverse_transform(p_test))
            note = f"GEANT, loss mae_raw, best epoch {int(np.argmin(hist['val_loss'])) + 1} of {len(hist['loss'])}, seed {seed}"
            log_result(name, mv, horizon=h, dataset="geant", notes=f"val split, all 529 flows, {note}")
            log_result(name, mt, horizon=h, dataset="geant", notes=f"test split, all 529 flows, {note}")
            res[h][key].append((seed, mv, mt))
            print(f"  {label:10s} seed {seed}  val {mb(mv):8.3f}  test {mb(mt):8.3f} Mbit/s  "
                  f"(test {(1 - mt['mae'] / pt['mae']) * 100:+.1f}% vs persistence)")
    ctx[h] = {"scaler": scaler, "steps": steps[len(y_train) + len(y_val):], "y_test": scaler.inverse_transform(y_test),
              "pers": scaler.inverse_transform(persistence(X_test)),
              "model": np.mean(test_preds["cnn_lstm"], axis=0), "y_train": scaler.inverse_transform(y_train)}

# %% 3. results
report.section("3. Results")
for h in HORIZONS:
    pv, pt = res[h]["persistence"]["val"], res[h]["persistence"]["test"]
    print(f"\nHorizon {h} ({h * STEP_MIN} min). MAE in Mbit/s; change: positive = better than persistence")
    print(f"  {'':12s} {'val MAE':>16s} {'change':>8s}   {'test MAE':>16s} {'change':>8s}   seeds beating pers. (test)")
    print(f"  {'persistence':12s} {mb(pv):16.3f} {'':8s}   {mb(pt):16.3f}")
    for key, (label, _) in VARIANTS.items():
        v = np.array([mb(a) for _, a, _ in res[h][key]]); t = np.array([mb(b) for _, _, b in res[h][key]])
        print(f"  {label:12s} {v.mean():8.3f} ± {v.std():5.3f} {(1 - v.mean() / mb(pv)) * 100:+7.1f}%   "
              f"{t.mean():8.3f} ± {t.std():5.3f} {(1 - t.mean() / mb(pt)) * 100:+7.1f}%   {(t < mb(pt)).sum()}/{len(t)}")

# %% 4. where the gains are
report.section("4. Where the models differ from persistence")
for h in HORIZONS:
    c = ctx[h]
    e_p = np.abs(c["pers"] - c["y_test"]).mean(0)
    e_m = np.abs(c["model"] - c["y_test"]).mean(0)
    size = c["y_test"].mean(0)
    order = np.argsort(size)[::-1]
    quart = np.digitize(size, np.quantile(size, [0.25, 0.5, 0.75]))
    win = e_m < e_p
    print(f"\nhorizon {h}: CNN-LSTM (3-seed mean forecast) beats persistence on {win.sum()} of 529 flows "
          f"(smallest quarter {win[quart == 0].sum()}/{(quart == 0).sum()}, largest quarter {win[quart == 3].sum()}/{(quart == 3).sum()})")
    big = order[:10]
    print(f"  10 busiest flows: persistence MAE {e_p[big].mean() / 1000:.2f}, CNN-LSTM {e_m[big].mean() / 1000:.2f} Mbit/s "
          f"({(1 - e_m[big].mean() / e_p[big].mean()) * 100:+.1f}%)")
    worst = np.argsort(e_m - e_p)[::-1][:3]
    print("  flows where the model loses most to persistence (src>dst index, MAE gap Mbit/s):",
          [(f"{k // 23}>{k % 23}", round(float((e_m - e_p)[k]) / 1000, 2)) for k in worst])

# %% 5. link level (approximate routing, relative threshold)
report.section("5. Link level (approximate)")
R = load_geant_routing()
A = R["A"]
print(f"{len(R['links'])} directed links; minimum-hop routing with equal splitting (the real IGP weights are not in the data).")
print("High load = link load in the top 10% of that link's own training-period loads. Capacities are unknown,")
print("so this is NOT congestion against capacity. Test split, mean of 3 seeds for the CNN-LSTM.\n")
print(f"  {'':14s} {'load MAE':>14s} {'(% of mean)':>11s} | {'':12s} {'precision':>9s} {'recall':>7s} {'F1':>6s} {'onsets warned':>14s}")
link = {}
for h in HORIZONS:
    c = ctx[h]
    load = lambda f: f.astype(np.float64) @ A.T
    thr = np.quantile(load(c["y_train"]), 0.9, axis=0)
    true_l, pers_l, mod_l = load(c["y_test"]), load(c["pers"]), load(c["model"])
    t_ev = true_l >= thr
    on = warn_onsets(t_ev, c["steps"])
    link[h] = {}
    for name, label, ld in (("pers", "persistence", pers_l), ("model", "CNN-LSTM", mod_l)):
        mae = np.abs(ld - true_l).mean()
        p_ev = ld >= thr
        p, r, f, tp, fp, fn = warn_prf(t_ev, p_ev)
        link[h][name] = (mae, p, r, f, int((on & p_ev).sum()), int(on.sum()))
        first = f"h={h} ({h * STEP_MIN} min)" if name == "pers" else ""
        print(f"  {first:14s} {mae / 1000:10.2f} Mbit/s {mae / true_l.mean() * 100:9.1f}%  | {label:12s} {p:9.2f} {r:7.2f} {f:6.2f} "
              f"{link[h][name][4]:>8d}/{link[h][name][5]}")
    print(f"  {'':14s} ({t_ev.sum()} high-load link-steps in the test split, {int(on.sum())} onsets)")

# %% 6. figures
report.section("6. Figures")
HUE = {"cnn_lstm": COLORS["violet"], "lstm": COLORS["aqua"], "cnn": COLORS["orange"]}
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
for ax, split, title in zip(axes, (1, 2), ("validation", "test")):
    for i, key in enumerate(VARIANTS):
        mean, sd = [], []
        for h in HORIZONS:
            base = mb(res[h]["persistence"]["val" if split == 1 else "test"])
            g = np.array([(1 - mb(r[split]) / base) * 100 for r in res[h][key]])
            mean.append(g.mean()); sd.append(g.std())
        ax.bar(np.arange(len(HORIZONS)) + (i - 1) * 0.26, mean, 0.24, yerr=sd, color=HUE[key],
               label=VARIANTS[key][0], capsize=3)
    ax.axhline(0, color=COLORS["ink2"], linewidth=1)
    ax.set_xticks(range(len(HORIZONS)), [f"{h} step{'s' if h > 1 else ''}\n({h * STEP_MIN} min)" for h in HORIZONS])
    ax.set_title(f"GÉANT {title}: MAE change vs persistence"); ax.grid(axis="x", visible=False)
axes[0].set_ylabel("% better than persistence (higher is better)")
axes[0].legend(frameon=False, loc="best")
fig.tight_layout()
show(fig, "03e_models.png", "Change in MAE relative to persistence at the same horizon, mean of 3 seeds with one "
     "standard deviation. Same architecture and settings as Abilene, retrained on GÉANT, no retuning.")

# %% write the results page and open it in the browser
report.finish()
