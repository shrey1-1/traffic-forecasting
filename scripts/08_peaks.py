"""08 Peaks and bursts: Phase 10.

Phase 9's seasonal inputs gave large gains on ordinary flows but lost badly on LOSA->CHIN
(flow 86) during its bursts. A diagnosis (docs/PROJECT_LOG.md section 21.1) showed the loss
comes from steps where a flow is far from its usual level for that time of day: the seasonal
term then pulls the forecast back to normal, which is wrong in a burst. Out-of-range inputs
explain only about 10% of it. NOTE: that diagnosis looked at test results, so nothing here is
tuned on them; every choice below is fixed in advance or made on validation.

Compared (all residual CNN-LSTM, seasonal inputs, original-units MAE loss, 3 seeds, GPU):
 base      Phase 9 base, no seasonal inputs (p9g_*_base)
 seas      Phase 9 seasonal model (p9g_*_seas)
 rule      seas, but persistence for a flow-step when |seasonal - last| > k training sd of that
           flow; k from (3, 4, 5, 6, 8) chosen on validation (largest gain; within 0.05% of
           the best, the smaller k, i.e. the more cautious rule, wins). No training.
 gate      seas with a learned gate (common._seasonal_gate_class): the seasonal term is
           multiplied by sigmoid(a - b z), z = |seasonal - last| / sd, with a, b learned from
           the training data (2 numbers shared by all flows), starting almost open. New models.
Scored on all flows, without flow 86, on flow 86 alone, and on peaks (targets above the
flow's training 95th percentile; bias = mean (forecast - true) / mean true, as Phase 7).

Run `python scripts/08_peaks.py --only abilene:6` (etc.) to train without writing the page.
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
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import tensorflow as tf
import common
importlib.reload(common)
from common import *

ONLY = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None
GPU = bool(tf.config.list_physical_devices("GPU"))
if not GPU:
    raise SystemExit("Phase 10 compares against the Phase 9 GPU models (p9g_*): run it on the GPU (scripts/wsl/run.sh)")
report = None if ONLY else Report("08_peaks", "Phase 10: Peaks and bursts")
section = (lambda t: print(f"\n== {t} ==")) if report is None else report.section
section("Project status")
set_seeds()
status()

import matplotlib.pyplot as plt
use_plot_style()
RETRAIN = False
EPOCHS, PATIENCE, LR = 200, 15, 1e-3
SEEDS = (42, 43, 44)
CASES = [("abilene", 1), ("abilene", 6), ("abilene", 12), ("geant", 1), ("geant", 4)]
if ONLY:
    CASES = [c for c in CASES if f"{c[0]}:{c[1]}" == ONLY or c[0] == ONLY]
DS_NAME = {"abilene": "Abilene", "geant": "GÉANT"}
STEP_MIN = {"abilene": 5, "geant": 15}
N_F = {"abilene": N_FLOWS, "geant": GEANT_FLOWS}
CAPS = {"abilene": None, "geant": GEANT_MAX_FLOW_KBPS}
KS = (3, 4, 5, 6, 8)
TIE = 0.05
F86 = 86
case_label = lambda ds, h: f"{DS_NAME[ds]} {h * STEP_MIN[ds]} min"
mae = lambda y, p: float(np.mean(np.abs(p - y)))
gain = lambda m, b: (1 - m / b) * 100
LABEL = {"base": "base (no seasonal)", "seas": "seasonal (Phase 9)", "rule": "seasonal + fallback rule",
         "gate": "seasonal + learned gate"}


def ens(names, d, i):
    out = []
    for n in names:
        m = load_saved(n)
        out.append(d["scaler"].inverse_transform(predict(m, model_inputs(d, i), scaler=d["scaler"],
                                                         max_raw=CAPS[d["ds"]])).astype(np.float64))
        tf.keras.backend.clear_session()
    return out


# %% 1. train the gated models (or load them) and collect forecasts
section("1. Models")
R, GATES = {}, {}
for ds, h in CASES:
    case = (ds, h)
    d = context_dataset(ds, h, seasonal="seasonal_mean7")
    d["ds"] = ds
    b = context_dataset(ds, h)
    b["ds"] = ds
    sd = d["S"][0][:, 0].std(0)                     # per-flow training spread of seasonal(t) - last
    loss = make_loss("mae_raw", d["scaler"], float(b["y_raw"][0].mean()))
    gate_names = [f"p10g_{ds}_h{h}_gate_s{s}" for s in SEEDS]
    GATES[case] = []
    for seed, name in zip(SEEDS, gate_names):
        print(f"  {case_label(ds, h)}, learned gate, seed {seed}")
        model, hist, _ = fit_or_load(
            lambda: build_cnn_lstm(n_flows=N_F[ds], residual=True, seasonal=True, gate_sd=sd),
            name, model_inputs(d, 0), d["y"][0], model_inputs(d, 1), d["y"][1],
            retrain=RETRAIN, seed=seed, epochs=EPOCHS, patience=PATIENCE, lr=LR, loss=loss)
        gl = model.get_layer("seasonal_gate")
        a, bb = float(gl.a.numpy()), float(tf.nn.softplus(gl.b_raw).numpy())
        GATES[case].append((a, bb))
        print(f"    learned gate: a = {a:.2f}, b = {bb:.3f} -> half closed at {a / bb:.1f} sd" if bb > 0 else "")
        tf.keras.backend.clear_session()
    if ONLY:
        continue
    y = {i: d["y_raw"][i] for i in (1, 2)}
    last = {i: d["last_raw"][i] for i in (1, 2)}
    z = {i: np.abs(d["S"][i][:, 0]) / np.where(sd > 0, sd, 1) for i in (1, 2)}
    P = {}
    for key, names, dd in (("base", [f"p9g_{ds}_h{h}_base_s{s}" for s in SEEDS], b),
                           ("seas", [f"p9g_{ds}_h{h}_seas_s{s}" for s in SEEDS], d),
                           ("gate", gate_names, d)):
        P[key] = {i: ens(names, dd, i) for i in (1, 2)}
    # fallback rule: k chosen on validation, fixed in advance how
    seas_mean = {i: np.mean(P["seas"][i], 0) for i in (1, 2)}
    pv = mae(y[1], last[1])
    scores = {k: gain(mae(y[1], np.where(z[1] > k, last[1], seas_mean[1])), pv) for k in KS}
    best = max(scores.values())
    k_rule = min(k for k in KS if scores[k] >= best - TIE)
    print(f"  {case_label(ds, h)}: fallback rule validation gains " + ", ".join(f"k={k}: {v:+.2f}%" for k, v in scores.items())
          + f" -> k = {k_rule}")
    P["rule"] = {i: [np.where(z[i] > k_rule, last[i], p) for p in P["seas"][i]] for i in (1, 2)}
    R[case] = {"y": y, "last": last, "P": P, "k": k_rule, "y_train": b["y_raw"][0], "share_rule": float((z[2] > k_rule).mean())}
    del d, b

if ONLY:
    print(f"\ntraining for {ONLY} done; run the script without --only to write the page")
    raise SystemExit

# %% 2. results
section("2. Results (test unless stated)")
print("MAE change vs persistence, positive = better; forecasts are the mean of 3 seeds (single seeds in brackets).\n"
      "'peak bias' = mean (forecast - true) / mean true on targets above the flow's training 95th percentile\n"
      "(persistence's own peak bias in the header; closer to 0 is better).")
KEYS = ("base", "seas", "rule", "gate")
SUM = {}
for ds, h in CASES:
    case = (ds, h)
    r = R[case]
    y, last = r["y"], r["last"]
    pv, pt = mae(y[1], last[1]), mae(y[2], last[2])
    nf = y[2].shape[1]
    others = np.arange(nf) != F86 if ds == "abilene" else slice(None)
    p95 = np.percentile(r["y_train"], 95, axis=0)
    peak = y[2] > p95
    pbias = lambda p: float((p - y[2])[peak].mean() / y[2][peak].mean() * 100)
    print(f"\n{case_label(ds, h)}  (rule threshold k = {r['k']} sd, applied to {r['share_rule']:.2%} of test flow-steps;"
          f" persistence peak bias {pbias(last[2]):+.1f}%)")
    hdr = f"  {'':28s} {'val':>7s} {'test':>7s}  {'single seeds (test)':22s}"
    if ds == "abilene":
        hdr += f" {'w/o 86':>7s} {'flow 86':>8s}"
    print(hdr + f" {'peak bias':>9s}")
    for key in KEYS:
        pm = {i: np.mean(r["P"][key][i], 0) for i in (1, 2)}
        mv, mt = mae(y[1], pm[1]), mae(y[2], pm[2])
        seeds = ", ".join(f"{gain(mae(y[2], p), pt):+.1f}" for p in r["P"][key][2])
        line = f"  {LABEL[key]:28s} {gain(mv, pv):+6.2f}% {gain(mt, pt):+6.2f}%  {seeds:22s}"
        if ds == "abilene":
            line += (f" {gain(mae(y[2][:, others], pm[2][:, others]), mae(y[2][:, others], last[2][:, others])):+6.1f}%"
                     f" {gain(mae(y[2][:, F86], pm[2][:, F86]), mae(y[2][:, F86], last[2][:, F86])):+7.1f}%")
        print(line + f" {pbias(pm[2]):+8.1f}%")
        SUM[(case, key)] = (gain(mv, pv), gain(mt, pt))
        for split, i in (("val", 1), ("test", 2)):
            log_result(f"p10g_{key}_mean3", evaluate(y[i], pm[i]), horizon=h, dataset=ds,
                       notes=f"Phase 10, {split} split, all flows")
    if case in GATES:
        print("  learned gates (a, b; half closed at a/b sd): " + "; ".join(f"{a:.2f}, {b_:.3f} ({a / b_:.0f} sd)" for a, b_ in GATES[case]))

# %% 3. significance
section("3. Significance (test)")
print("Diebold-Mariano (Newey-West, one day of lags) and 95% moving-block bootstrap (one-day blocks);\n"
      "positive = the first is better.")
SIG = {}
for ds, h in CASES:
    case = (ds, h)
    r = R[case]
    day = STEPS_PER_DAY[ds]
    err = lambda key: np.abs(np.mean(r["P"][key][2], 0) - r["y"][2]).mean(1)
    print(f"\n{case_label(ds, h)}")
    for a_, b_ in (("gate", "seas"), ("gate", "base"), ("rule", "seas"), ("rule", "base"), ("gate", "rule")):
        ea, eb = err(a_), err(b_)
        _, _, p = dm_test(ea, eb, day)
        g_, lo, hi = block_bootstrap_gain(ea, eb, day)
        SIG[(case, a_, b_)] = (g_, lo, hi, p)
        verdict = "significant" if p < 0.05 and lo * hi > 0 else "not significant"
        print(f"  {LABEL[a_]:26s} vs {LABEL[b_]:26s} {g_:+6.2f}%  [{lo:+6.2f}, {hi:+6.2f}]  p = {p:.3f}  {verdict}")

# %% 4. figure
section("4. Figure")
HUE = {"base": COLORS["ink2"], "seas": COLORS["aqua"], "rule": COLORS["orange"], "gate": COLORS["violet"]}
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
for ax, j, title in zip(axes, (0, 1), ("validation", "test")):
    for n, key in enumerate(KEYS):
        xs = np.arange(len(CASES)) + (n - 1.5) * 0.2
        ax.bar(xs, [SUM[(c, key)][j] for c in CASES], 0.19, color=HUE[key], label=LABEL[key])
    ax.axhline(0, color=COLORS["ink2"], linewidth=1)
    ax.set_xticks(range(len(CASES)), [case_label(*c).replace(" ", "\n", 1) for c in CASES], fontsize=8.5)
    ax.set_title(f"{title}: MAE change vs persistence"); ax.grid(axis="x", visible=False)
axes[0].set_ylabel("% better than persistence")
axes[0].legend(frameon=False, fontsize=8, loc="upper left")
fig.tight_layout()
show(fig, "08_peaks.png", "MAE change against persistence (mean of 3 seeds' forecasts). The fallback rule and the "
     "learned gate switch the seasonal term off when a flow is far from its usual level.")

report.finish(open_browser="--no-browser" not in sys.argv)
