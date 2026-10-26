"""09 Early warning, redone: Phase 11.

Phase 6 found that CNN-LSTM congestion warnings were no better than reacting to the
current load (persistence). Since then the forecasts improved (Phase 10). This phase
asks whether better forecasts, and warning on a high estimate instead of the central
forecast, give better warnings.

Forecasters (test; models are the GPU ones, mean of 3 seeds):
 persistence     the link will look as it looks now (the reactive baseline)
 base            CNN-LSTM without seasonal inputs (p9g_*_base)
 seasonal+rule   Phase 10's best: seasonal CNN-LSTM, persistence for a flow more than k sd
                 from its seasonal level (k chosen on validation exactly as in Phase 10)
Each is used in two ways:
 point           warn when the forecast link load is at or above the threshold
 high (q90)      warn when forecast + margin is, where the margin of each link is the 90th
                 percentile of (true - forecast) for that link on VALIDATION (split-conformal
                 calibration). Fixed at 90% in advance; other levels only as a sensitivity check.
If the margins are calibrated, the true load is below forecast + margin about 90% of the
time on test; this is printed as a check.

Events: Abilene, link utilisation >= 50% (40% and 60% as sensitivity), as Phase 6.
GÉANT, link load in the top 10% of that link's training loads (no capacities; approximate
minimum-hop routing), as Phase 6. Scores: precision, recall, F1 per link-step, onsets warned
in advance, and a 95% day-block bootstrap interval for F1 differences.
Run on the GPU environment (it loads the p9g models). Output: results/report.html
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
report = Report("09_warnings", "Phase 11: Early warning, redone")
report.section("Project status")
set_seeds()
status()

import matplotlib.pyplot as plt
use_plot_style()
SEEDS = (42, 43, 44)
CASES = [("abilene", 1), ("abilene", 6), ("abilene", 12), ("geant", 1), ("geant", 4)]
DS_NAME = {"abilene": "Abilene", "geant": "GÉANT"}
STEP_MIN = {"abilene": 5, "geant": 15}
CAPS = {"abilene": None, "geant": GEANT_MAX_FLOW_KBPS}
THRESHOLDS = (0.4, 0.5, 0.6)
PRIMARY = 0.5
Q = 0.9
Q_SENS = (0.5, 0.75, 0.9, 0.95, 0.99)
KS, TIE = (3, 4, 5, 6, 8), 0.05                   # Phase 10's rule selection, unchanged
METHODS = {"pers": "persistence", "base": "CNN-LSTM base", "rule": "seasonal + rule"}
case_label = lambda ds, h: f"{DS_NAME[ds]} {h * STEP_MIN[ds]} min"
mae = lambda y, p: float(np.mean(np.abs(p - y)))
gain = lambda m, b: (1 - m / b) * 100


def ens(prefix, d, i):
    out = []
    for s in SEEDS:
        m = load_saved(f"{prefix}_s{s}")
        out.append(d["scaler"].inverse_transform(predict(m, model_inputs(d, i), scaler=d["scaler"],
                                                         max_raw=CAPS[d["ds"]])).astype(np.float64))
        tf.keras.backend.clear_session()
    return np.mean(out, 0)


def f1_boot(t_ev, a_ev, b_ev, block, n_boot=2000, seed=SEED):
    """95% interval of F1(a) - F1(b), resampling blocks of `block` consecutive rows."""
    rng = np.random.default_rng(seed)
    n = len(t_ev)
    k = int(np.ceil(n / block))
    tp = lambda ev: (t_ev & ev).sum(1); fp = lambda ev: (~t_ev & ev).sum(1); fn = lambda ev: (t_ev & ~ev).sum(1)
    rows = {nm: (tp(ev), fp(ev), fn(ev)) for nm, ev in (("a", a_ev), ("b", b_ev))}
    out = []
    for _ in range(n_boot):
        idx = (rng.integers(0, n - block + 1, size=k)[:, None] + np.arange(block)).ravel()[:n]
        f = []
        for nm in ("a", "b"):
            TP, FP, FN = (x[idx].sum() for x in rows[nm])
            f.append(2 * TP / (2 * TP + FP + FN) if TP else 0.0)
        out.append(f[0] - f[1])
    return np.percentile(out, [2.5, 97.5])


# %% 1. forecasts, mapped onto links
report.section("1. Forecasts mapped onto links")
RA = load_abilene_routing()
bb = np.array(RA["link_type"]) == 0
A_ab, cap_ab = RA["A"][bb], RA["capacity_kbps"][bb]
A_ge = load_geant_routing()["A"]
to_link = {"abilene": lambda f: f @ A_ab.T * ABILENE_UNIT_BYTES * 8 / 300 / 1000 / cap_ab,   # utilisation 0..1
           "geant": lambda f: f @ A_ge.T}                                                      # kbit/s
C = {}
for ds, h in CASES:
    case = (ds, h)
    ds_ = context_dataset(ds, h, seasonal="seasonal_mean7"); ds_["ds"] = ds
    db = context_dataset(ds, h); db["ds"] = ds
    sd = ds_["S"][0][:, 0].std(0)
    sd = np.where(sd > 0, sd, 1)
    fc = {}
    for i in (1, 2):
        seas = ens(f"p9g_{ds}_h{h}_seas", ds_, i)
        z = np.abs(ds_["S"][i][:, 0]) / sd
        fc[i] = {"pers": ds_["last_raw"][i], "base": ens(f"p9g_{ds}_h{h}_base", db, i), "seas": seas, "z": z}
    y = {i: ds_["y_raw"][i] for i in (1, 2)}
    pv = mae(y[1], fc[1]["pers"])
    scores = {k: gain(mae(y[1], np.where(fc[1]["z"] > k, fc[1]["pers"], fc[1]["seas"])), pv) for k in KS}
    k = min(kk for kk in KS if scores[kk] >= max(scores.values()) - TIE)
    for i in (1, 2):
        fc[i]["rule"] = np.where(fc[i]["z"] > k, fc[i]["pers"], fc[i]["seas"])
    L = {i: {m: to_link[ds](fc[i][m]) for m in METHODS} for i in (1, 2)}
    T = {i: to_link[ds](y[i]) for i in (1, 2)}
    margin = {m: np.quantile(T[1] - L[1][m], Q, axis=0) for m in METHODS}
    C[case] = {"L": L, "T": T, "margin": margin, "k": k, "times": ds_["times"][2],
               "thr_ge": np.quantile(to_link[ds](ds_["y_raw"][0]), 0.9, axis=0) if ds == "geant" else None,
               "val_L": L[1], "val_T": T[1]}
    unit = "pp of capacity" if ds == "abilene" else "Mbit/s"
    s_ = 100 if ds == "abilene" else 1 / 1000
    print(f"{case_label(ds, h)} (rule k = {k}):  link-load MAE "
          + ", ".join(f"{METHODS[m]} {np.abs(L[2][m] - T[2]).mean() * s_:.3f}" for m in METHODS) + f" {unit};"
          + "  share of test link-steps below forecast + q90 margin: "
          + ", ".join(f"{METHODS[m]} {(T[2] <= L[2][m] + margin[m]).mean():.1%}" for m in METHODS))
    del ds_, db
print("\n(the last numbers should be near 90% if the validation margins carry over to test)")

# %% 2. warnings
report.section("2. Warnings (test)")
print("P = precision (share of warnings that were right), R = recall (share of events warned), F1 combines both.\n"
      "'onsets' = congestion starts that the forecast made that far ahead already flagged.")
RES = {}
for ds, h in CASES:
    case = (ds, h)
    c = C[case]
    thrs = THRESHOLDS if ds == "abilene" else ("top 10%",)
    for thr in thrs:
        tv = c["thr_ge"] if ds == "geant" else thr
        t_ev = c["T"][2] >= tv
        on = warn_onsets(t_ev, c["times"])
        primary = ds == "geant" or thr == PRIMARY
        if primary:
            print(f"\n{case_label(ds, h)}, threshold {thr if isinstance(thr, str) else f'{thr:.0%}'}: "
                  f"{int(t_ev.sum())} event link-steps, {int(on.sum())} onsets")
            print(f"  {'':28s} {'P':>5s} {'R':>5s} {'F1':>5s} {'onsets':>9s}")
        for m in METHODS:
            for form in ("point", "high"):
                p_ev = (c["L"][2][m] + (c["margin"][m] if form == "high" else 0)) >= tv
                p, r, f, *_ = warn_prf(t_ev, p_ev)
                RES[(case, thr, m, form)] = (p, r, f, int((on & p_ev).sum()), int(on.sum()), p_ev, t_ev)
                if primary:
                    print(f"  {METHODS[m] + (' (q90 high)' if form == 'high' else ''):28s} {p:5.2f} {r:5.2f} {f:5.2f} "
                          f"{int((on & p_ev).sum()):>4d}/{int(on.sum())}")
print("\nSensitivity, Abilene thresholds 40% / 60% (F1, point -> q90 high):")
for h in (1, 6, 12):
    for thr in (0.4, 0.6):
        print(f"  {h * 5:2d} min, {thr:.0%}: " + "; ".join(
            f"{METHODS[m]} {RES[(('abilene', h), thr, m, 'point')][2]:.2f} -> {RES[(('abilene', h), thr, m, 'high')][2]:.2f}"
            for m in METHODS))

# %% 3. is any difference real?
report.section("3. Are the F1 differences real? (test, primary threshold)")
print("95% interval of the F1 difference, day-block bootstrap. An interval containing 0 = no clear difference.")
for ds, h in CASES:
    case = (ds, h)
    thr = "top 10%" if ds == "geant" else PRIMARY
    block = STEPS_PER_DAY[ds]
    t_ev = RES[(case, thr, "pers", "point")][6]
    print(f"\n{case_label(ds, h)}")
    for a, b in ((("rule", "high"), ("pers", "point")), (("rule", "high"), ("pers", "high")),
                 (("rule", "point"), ("pers", "point")), (("base", "high"), ("pers", "high"))):
        fa, fb = RES[(case, thr) + a][2], RES[(case, thr) + b][2]
        lo, hi = f1_boot(t_ev, RES[(case, thr) + a][5], RES[(case, thr) + b][5], block)
        name = lambda x: METHODS[x[0]] + (" q90" if x[1] == "high" else "")
        print(f"  {name(a):24s} vs {name(b):24s} F1 {fa:.3f} vs {fb:.3f}  difference [{lo:+.3f}, {hi:+.3f}]"
              f"  {'clear' if lo * hi > 0 else 'not clear'}")

# %% 4. how the margin level trades precision for recall
report.section("4. Margin level (sensitivity, Abilene 30 min, 50%)")
case = ("abilene", 6)
c = C[case]
t_ev = c["T"][2] >= PRIMARY
print(f"  {'level':>6s} | " + " | ".join(f"{METHODS[m]:>22s}" for m in METHODS))
print(f"  {'':6s} | " + " | ".join(f"{'P':>6s} {'R':>6s} {'F1':>6s}  " for m in METHODS))
CURVE = {m: [] for m in METHODS}
for q in Q_SENS:
    row = []
    for m in METHODS:
        mg = np.quantile(c["val_T"] - c["val_L"][m], q, axis=0)
        p, r, f, *_ = warn_prf(t_ev, c["L"][2][m] + mg >= PRIMARY)
        CURVE[m].append((p, r, f))
        row.append(f"{p:6.2f} {r:6.2f} {f:6.2f}  ")
    print(f"  {q:6.2f} | " + " | ".join(row))

# %% 5. figure
report.section("5. Figure")
HUE = {"pers": COLORS["orange"], "base": COLORS["ink2"], "rule": COLORS["violet"]}
fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.9))
for n, m in enumerate(METHODS):
    for k_, form in enumerate(("point", "high")):
        xs = np.arange(len(CASES)) + (n * 2 + k_ - 2.5) * 0.13
        vals = [RES[(cs, "top 10%" if cs[0] == "geant" else PRIMARY, m, form)][2] for cs in CASES]
        axes[0].bar(xs, vals, 0.12, color=HUE[m], alpha=1 if form == "high" else 0.45,
                    label=METHODS[m] + (" q90" if form == "high" else ""))
axes[0].set_xticks(range(len(CASES)), [case_label(*cs).replace(" ", "\n", 1) for cs in CASES], fontsize=8.5)
axes[0].set_ylim(0, 1.05); axes[0].set_title("F1 of warnings (test; faded = point, solid = q90 high)")
axes[0].legend(frameon=False, fontsize=7.5, ncol=3, loc="lower center"); axes[0].grid(axis="x", visible=False)
for m in METHODS:
    pr = np.array(CURVE[m])
    axes[1].plot(pr[:, 1], pr[:, 0], "o-", color=HUE[m], label=METHODS[m])
    for q, (p, r, _) in zip(Q_SENS, CURVE[m]):
        axes[1].annotate(f"{q:g}", (r, p), fontsize=7, color=HUE[m], xytext=(3, 3), textcoords="offset points")
axes[1].set_xlabel("recall"); axes[1].set_ylabel("precision")
axes[1].set_title("Abilene 30 min, 50%: margin level 0.5 to 0.99")
axes[1].legend(frameon=False, fontsize=8)
fig.tight_layout()
show(fig, "09_warnings.png", "Left: F1 of warnings at the primary threshold. Right: how a larger safety margin trades "
     "precision for recall (labels are the margin's quantile level, calibrated on validation).")

report.finish(open_browser="--no-browser" not in sys.argv)
