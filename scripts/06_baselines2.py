"""06 Stronger baselines and significance: Phase 8.

Raises the bar the models are judged against, and tests whether the gains reported
in Phases 4-6 are larger than chance. Abilene at 1, 6 and 12 steps (5, 30, 60 min)
and GÉANT at 1 and 4 steps (15, 60 min), the same data and splits as before.

Baselines (all in original units, on all flows unless stated):
 seasonal_day    the value at the same time yesterday
 seasonal_week   the value at the same time last week
 seasonal_mean7  the mean of the same time on the last 7 days
 blend           last value + w * (seasonal reference - last value); the reference
                 and w are chosen on validation
 yday_change     last value + v * (yesterday's change over the same interval); v on validation
 ridge           linear model on the whole 12-step window of every flow, predicting
                 the change from the last value; penalty and scale (scaled or original
                 units) chosen on validation
 arima_top5      ARIMA on the 5 busiest Abilene flows (as Phase 2), now also at 6 and 12
                 steps; order chosen per flow and horizon on validation
Where a seasonal value is missing (a data gap a day or week earlier), the last
value is used instead; the share of forecasts where this happens is printed.

Significance: the CNN-LSTM, LSTM-only and CNN-only forecasts (mean of the 3 saved
seeds, as in Phase 7) against persistence and against the best baseline (the
lowest validation MAE). Per-step loss = absolute error averaged over flows.
Diebold-Mariano test with Newey-West variance over one day of lags, and a
moving-block bootstrap (blocks of one day) for a 95% interval of the MAE change.
Both allow for errors on nearby steps being correlated; one day is a cautious choice.

Nothing is chosen on test. Output: results/report.html
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
report = Report("06_baselines2", "Phase 8: Stronger baselines and significance")
report.section("Project status")
set_seeds()
status()

import time
import warnings
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
use_plot_style()

CASES = [("abilene", 1), ("abilene", 6), ("abilene", 12), ("geant", 1), ("geant", 4)]
DS_NAME = {"abilene": "Abilene", "geant": "GÉANT"}
UNIT = {"abilene": MBIT, "geant": 1 / 1000}                   # raw units -> Mbit/s
STEP_MIN = {"abilene": 5, "geant": 15}
CAP = {"abilene": MAX_FLOW_RAW, "geant": GEANT_MAX_FLOW_KBPS}  # physical limit on a forecast
SEEDS = (42, 43, 44)
NN = {"cnn_lstm": "CNN-LSTM", "lstm": "LSTM-only", "cnn": "CNN-only"}
GRID = np.linspace(0, 1, 101)                                   # blend weights
ALPHAS = 10.0 ** np.arange(-2, 5)                               # ridge penalties (times n)
ARIMA_ORDERS = [((1, 0, 0), "c"), ((2, 0, 0), "c"), ((1, 0, 1), "c"), ((2, 0, 1), "c"),
                ((0, 1, 1), "n"), ((1, 1, 1), "n"), ((2, 1, 1), "n")]   # as Phase 2
LABEL = {"persistence": "persistence", "seasonal_day": "same time yesterday",
         "seasonal_week": "same time last week", "seasonal_mean7": "mean of last 7 days",
         "blend": "blend (last + seasonal)", "yday_change": "last + yesterday's change",
         "ridge": "ridge (linear)"}
BASELINES = list(LABEL)
case_label = lambda ds, h: f"{DS_NAME[ds]} {h * STEP_MIN[ds]} min"
mae = lambda y, p: float(np.mean(np.abs(p - y)))
gain = lambda m, base: (1 - m / base) * 100                     # % better than base


def load_case(ds, h):
    d = data_dir(ds, h)
    X_tr, y_tr, X_va, y_va, X_te, y_te = load_splits(d)
    sc = FlowScaler.load(d / "scaler.npz")
    steps = np.load(d / "target_steps.npy")
    n_tr, n_va = len(y_tr), len(y_va)
    f64 = lambda a: sc.inverse_transform(a).astype(np.float64)
    return {"X": (X_tr, X_va, X_te), "sc": sc, "n_tr": n_tr,
            "y": (f64(y_tr), f64(y_va), f64(y_te)),
            "last": (f64(X_tr[:, -1]), f64(X_va[:, -1]), f64(X_te[:, -1])),
            "steps": (steps[:n_tr], steps[n_tr:n_tr + n_va], steps[n_tr + n_va:])}


# %% 1. seasonal, blend and yesterday's-change baselines
report.section("1. Seasonal baselines")
TL = {ds: clean_timeline(ds) for ds in ("abilene", "geant")}
R = {}           # R[case][method] = {"val": pred, "test": pred} in raw units
C = {}           # loaded data per case
for ds, h in CASES:
    case = (ds, h)
    c = C[case] = load_case(ds, h)
    values, known, to_time = TL[ds]
    day = STEPS_PER_DAY[ds]
    R[case] = {"persistence": {"val": c["last"][1], "test": c["last"][2]}}
    print(f"\n{case_label(ds, h)}  (train {c['n_tr']}, val {len(c['y'][1])}, test {len(c['y'][2])})")
    refs = {}
    for split, i in (("val", 1), ("test", 2)):
        t = to_time[c["steps"][i]]
        last = c["last"][i]
        refs.setdefault("seasonal_day", {})[split] = lookback(values, known, t, day)
        refs.setdefault("seasonal_week", {})[split] = lookback(values, known, t, 7 * day)
        refs.setdefault("seasonal_mean7", {})[split] = seasonal_mean(values, known, t, day, 7)
        ych = lookback(values, known, t, day) - lookback(values, known, t - h, day)
        refs.setdefault("yday_change", {})[split] = ych
    for name in ("seasonal_day", "seasonal_week", "seasonal_mean7"):
        miss = np.isnan(refs[name]["test"][:, 0]).mean()
        R[case][name] = {s: np.where(np.isnan(refs[name][s]), c["last"][i], refs[name][s])
                         for s, i in (("val", 1), ("test", 2))}
        print(f"  {LABEL[name]:26s} missing (last value used instead) on {miss:5.1%} of test forecasts")

    # blend: last + w * (ref - last), the reference and w chosen on validation
    best = None
    for name in ("seasonal_day", "seasonal_week", "seasonal_mean7"):
        diff_va = R[case][name]["val"] - c["last"][1]
        errs = [mae(c["y"][1], c["last"][1] + w * diff_va) for w in GRID]
        j = int(np.argmin(errs))
        if best is None or errs[j] < best[0]:
            best = (errs[j], name, GRID[j])
    _, ref, w = best
    R[case]["blend"] = {s: c["last"][i] + w * (R[case][ref][s] - c["last"][i]) for s, i in (("val", 1), ("test", 2))}
    print(f"  blend: reference '{LABEL[ref]}', weight w = {w:.2f} (chosen on validation)")
    C[case]["blend_cfg"] = (ref, w)

    # last value + v * yesterday's change over the same h steps
    ch_va = np.nan_to_num(refs["yday_change"]["val"])
    errs = [mae(c["y"][1], c["last"][1] + v * ch_va) for v in GRID]
    v = GRID[int(np.argmin(errs))]
    R[case]["yday_change"] = {s: np.clip(c["last"][i] + v * np.nan_to_num(refs["yday_change"][s]), 0, CAP[ds])
                              for s, i in (("val", 1), ("test", 2))}
    print(f"  yesterday's change: v = {v:.2f} (chosen on validation)")
    C[case]["ych_v"] = v

# %% 2. ridge regression on the whole window
report.section("2. Ridge regression")
print("Inputs: all 12 steps of every flow (flattened). Output: the change from the last value for every flow.\n"
      "Two versions: in scaled units (log1p + min-max, like the neural networks' inputs) and in original units.\n"
      "Both are fitted by least squares with a ridge penalty; the penalty and the version are chosen on validation.")
for ds, h in CASES:
    case = (ds, h)
    c = C[case]
    sc = c["sc"]
    X_tr, X_va, X_te = c["X"]
    t0 = time.time()
    lo = (np.log1p(0.0) - sc.min_) / sc.range_
    hi = (np.log1p(CAP[ds]) - sc.min_) / sc.range_
    cands = []
    # scaled units
    flat = lambda X: X.reshape(len(X), -1).astype(np.float64)
    path = ridge_path(flat(X_tr), (sc.transform(c["y"][0]) - X_tr[:, -1]).astype(np.float64), ALPHAS)
    for a, (W, b) in zip(ALPHAS, path):
        f = lambda X, W=W, b=b: sc.inverse_transform(np.clip(X[:, -1] + flat(X) @ W + b, lo, hi)).astype(np.float64)
        cands.append((mae(c["y"][1], f(X_va)), "scaled", a, f))
    del path
    # original units
    raw = lambda X: sc.inverse_transform(X).reshape(len(X), -1).astype(np.float64)
    path = ridge_path(raw(X_tr), c["y"][0] - c["last"][0], ALPHAS)
    for a, (W, b) in zip(ALPHAS, path):
        f = lambda X, W=W, b=b: np.clip(sc.inverse_transform(X[:, -1]).astype(np.float64) + raw(X) @ W + b,
                                                0, CAP[ds])
        cands.append((mae(c["y"][1], f(X_va)), "original", a, f))
    del path
    best = min(cands, key=lambda r: r[0])
    pv = mae(c["y"][1], c["last"][1])
    print(f"\n{case_label(ds, h)}: {len(cands)} fits in {time.time() - t0:.0f} s; validation MAE change vs persistence:")
    for space in ("scaled", "original"):
        row = [f"{gain(m, pv):+6.1f}%" for m, s, a, _ in cands if s == space]
        print(f"  {space:9s} alpha " + "  ".join(f"{a:g}:{r}" for a, r in zip(ALPHAS, row)))
    _, space, a, f = best
    print(f"  chosen: {space} units, alpha {a:g}")
    R[case]["ridge"] = {"val": f(X_va), "test": f(X_te)}
    C[case]["ridge_cfg"] = (space, a)
    del cands

# %% 3. ARIMA on the 5 busiest Abilene flows at 1, 6 and 12 steps
report.section("3. ARIMA at 1, 6 and 12 steps (Abilene, 5 busiest flows)")
weeks = load_abilene_raw()
segments, _ = abilene_clean(weeks)
ab_cases = [cs for cs in CASES if cs[0] == "abilene"]
tr_ends = {h: int(C[("abilene", h)]["steps"][0][-1]) for _, h in ab_cases}
tr_end = min(tr_ends.values())        # fit once, on data that is training data at every horizon
print("last training target per horizon:", tr_ends, f"-> fitting ends at {step_label(tr_end)}")
seg_start, seg = next((s, g) for s, g in segments if s <= tr_end < s + len(g))
fit_end = tr_end - seg_start + 1
fit_len = min(4 * STEPS_PER_WEEK, fit_end)
print(f"fitting data: {step_label(seg_start + fit_end - fit_len)} .. {step_label(tr_end)} ({fit_len} steps, training only)")
c1 = C[("abilene", 1)]
top = np.argsort(c1["y"][0].mean(0))[::-1][:5]
print("flows:", ", ".join(f"#{k} {ABILENE_NODES[k // 12]}->{ABILENE_NODES[k % 12]}" for k in top))


def arima_forecast(res, k, targets, h):
    """h-step forecasts of flow k for the given target steps, parameters fixed.

    The fitted model is run over every segment holding targets (never across a gap).
    The forecast for step t uses only steps up to t - h: the Kalman filter's predicted
    state for step t - h + 1 is pushed h - 1 steps ahead with the transition matrix.
    (Checked to equal statsmodels' own forecast() and, at h = 1, predict().)
    """
    pred = np.full(len(targets), np.nan)
    for s, g in segments:
        sel = (targets >= s) & (targets < s + len(g))
        if not sel.any():
            continue
        y = np.log1p(g[:, k])
        fr = res.apply(y).filter_results
        Z, T, cst = fr.design[:, :, 0], fr.transition[:, :, 0], fr.state_intercept[:, 0]
        d = fr.obs_intercept[0] if fr.obs_intercept.shape[1] > 1 else np.full(len(y), fr.obs_intercept[0, 0])
        st = fr.predicted_state[:, :len(y)].copy()
        for _ in range(h - 1):
            st = T @ st + cst[:, None]
        f = np.full(len(y), np.nan)
        f[h - 1:] = (Z @ st)[0, :len(y) - h + 1] + d[h - 1:]
        pred[sel] = np.maximum(np.expm1(f[targets[sel] - s]), 0)
    assert not np.isnan(pred).any()
    return pred


t0 = time.time()
fits = {}
for k in top:
    y_fit = np.log1p(seg[fit_end - fit_len:fit_end, k])
    for order, trend in ARIMA_ORDERS:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fits[(k, order)] = ARIMA(y_fit, order=order, trend=trend).fit()
print(f"{len(fits)} ARIMA models fitted in {time.time() - t0:.0f} s")

ARIMA_RES = {}
for _, h in ab_cases:
    c = C[("abilene", h)]
    val_t, test_t = c["steps"][1], c["steps"][2]
    pa, chosen = np.zeros((len(test_t), len(top))), []
    for j, k in enumerate(top):
        scores = [(mae(c["y"][1][:, k], arima_forecast(fits[(k, o)], k, val_t, h)), o) for o, _ in ARIMA_ORDERS]
        m, o = min(scores)
        chosen.append(o)
        pa[:, j] = arima_forecast(fits[(k, o)], k, test_t, h)
    yt, lt = c["y"][2][:, top], c["last"][2][:, top]
    ARIMA_RES[h] = (mae(yt, pa), mae(yt, lt), chosen, pa)
    log_result("arima_top5", evaluate(yt, pa), horizon=h,
               notes="Phase 8, test split, 5 busiest flows, order per flow and horizon chosen on val")
    log_result("persistence_top5", evaluate(yt, lt), horizon=h, notes="Phase 8, test split, 5 busiest flows")
print(f"\n{'horizon':10s} {'ARIMA MAE':>10s} {'persistence':>12s} {'change':>8s}   orders")
for _, h in ab_cases:
    ma, mp, chosen, _ = ARIMA_RES[h]
    print(f"{h * 5:3d} min    {ma * MBIT:10.3f} {mp * MBIT:12.3f} {gain(ma, mp):+7.1f}%   {chosen}")
print("(test MAE in Mbit/s over the 5 busiest flows; positive change = ARIMA better than persistence)")

# %% 4. the neural networks' forecasts (Phases 4-6, mean of 3 seeds)
report.section("4. Neural network forecasts")
for ds, h in CASES:
    case = (ds, h)
    c = C[case]
    cap = None if ds == "abilene" else GEANT_MAX_FLOW_KBPS
    for key in NN:
        ps = {"val": [], "test": []}
        for n in saved_model_names(ds, h, key, SEEDS):
            m = tf.keras.models.load_model(MODELS / f"{n}.keras", compile=False)
            for s, i in (("val", 1), ("test", 2)):
                ps[s].append(c["sc"].inverse_transform(predict(m, c["X"][i], scaler=c["sc"], max_raw=cap)).astype(np.float64))
        R[case][key] = {s: np.mean(ps[s], 0) for s in ps}
        R[case][key]["seeds_test"] = [mae(c["y"][2], p) for p in ps["test"]]
    tf.keras.backend.clear_session()
print("loaded the 3 saved seeds of each model for every case; the forecast is their mean")

# %% 5. results table
report.section("5. All methods, MAE change vs persistence")
print("Positive = better than persistence. 'best' marks the baseline with the lowest VALIDATION MAE;\n"
      "it is the one the neural networks are compared with in section 6.\n")
BEST = {}
for ds, h in CASES:
    case = (ds, h)
    c, u = C[case], UNIT[ds]
    pv, pt = mae(c["y"][1], R[case]["persistence"]["val"]), mae(c["y"][2], R[case]["persistence"]["test"])
    BEST[case] = min((b for b in BASELINES if b != "persistence"), key=lambda b: mae(c["y"][1], R[case][b]["val"]))
    print(f"{case_label(ds, h)}   persistence MAE: val {pv * u:.3f}, test {pt * u:.3f} Mbit/s")
    print(f"  {'method':30s} {'val':>8s} {'test':>8s}")
    for name in BASELINES[1:] + list(NN):
        mv, mt = mae(c["y"][1], R[case][name]["val"]), mae(c["y"][2], R[case][name]["test"])
        tag = "  best" if name == BEST[case] else ""
        extra = ""
        if name in NN:
            seeds = [gain(m, pt) for m in R[case][name]["seeds_test"]]
            extra = f"   (single seeds: {', '.join(f'{g:+.1f}' for g in seeds)})"
        label = LABEL[name] if name in LABEL else NN[name] + ", 3-seed mean"
        print(f"  {label:30s} {gain(mv, pv):+7.1f}% {gain(mt, pt):+7.1f}%{tag}{extra}")
        for split, i in (("val", 1), ("test", 2)):
            log_result(name if name not in NN else f"{name}_mean3", evaluate(c["y"][i], R[case][name][split]),
                       horizon=h, dataset=ds, notes=f"Phase 8, {split} split, all flows")
    print()

# save the baseline forecasts so later phases compare against exactly these
out = PROCESSED / "phase8"
out.mkdir(exist_ok=True)
for ds, h in CASES:
    case = (ds, h)
    np.savez_compressed(out / f"{ds}_h{h}.npz", best=np.array(BEST[case]),
                        **{f"{n}_{s}": R[case][n][s].astype(np.float32) for n in BASELINES for s in ("val", "test")})
print(f"baseline forecasts saved to {out}")

# %% 6. significance
report.section("6. Are the gains real? (test split)")
print("Per-step loss = absolute error averaged over all flows. DM: Diebold-Mariano, Newey-West variance\n"
      "over one day of lags; p < 0.05 means the difference is unlikely to be chance. Interval: 95%\n"
      "moving-block bootstrap of the MAE change (blocks of one day). Positive = the first model is better.\n")
SIG = {}
for ds, h in CASES:
    case = (ds, h)
    c = C[case]
    day = STEPS_PER_DAY[ds]
    err = lambda name: np.abs(R[case][name]["test"] - c["y"][2]).mean(1)
    print(f"{case_label(ds, h)}  (test {len(c['y'][2])} steps = {len(c['y'][2]) / day:.0f} days; best baseline: {LABEL[BEST[case]]})")
    pairs = [(k, "persistence") for k in NN] + [(k, BEST[case]) for k in NN] + [(BEST[case], "persistence")]
    for a, b in pairs:
        ea, eb = err(a), err(b)
        _, stat, p = dm_test(ea, eb, day)
        g, lo, hi = block_bootstrap_gain(ea, eb, day)
        SIG[(case, a, b)] = (g, lo, hi, p)
        la, lb = LABEL.get(a, NN.get(a)), LABEL.get(b, NN.get(b))
        verdict = "significant" if p < 0.05 and lo * hi > 0 else "not significant"
        print(f"  {la:24s} vs {lb:24s} {g:+6.1f}%  [{lo:+5.1f}, {hi:+5.1f}]  DM p = {p:.3f}  {verdict}")
    print()

# %% 7. figures
report.section("7. Figures")
HUE = {"persistence": COLORS["ink2"], "seasonal_day": COLORS["yellow"], "seasonal_week": COLORS["yellow"],
       "seasonal_mean7": COLORS["yellow"], "blend": COLORS["orange"], "yday_change": COLORS["orange"],
       "ridge": COLORS["blue"], "cnn_lstm": COLORS["violet"], "lstm": COLORS["aqua"], "cnn": COLORS["red"]}
order = ["blend", "yday_change", "ridge", "cnn_lstm", "lstm", "cnn"]
fig, axes = plt.subplots(1, len(CASES), figsize=(13, 3.9), sharey=True)
for ax, (ds, h) in zip(axes, CASES):
    case, c = (ds, h), C[(ds, h)]
    pt = mae(c["y"][2], R[case]["persistence"]["test"])
    g = [gain(mae(c["y"][2], R[case][n]["test"]), pt) for n in order]
    ax.barh(range(len(order)), g, color=[HUE[n] for n in order])
    ax.axvline(0, color=COLORS["ink2"], linewidth=1)
    ax.set_title(case_label(ds, h), fontsize=10)
    ax.grid(axis="y", visible=False)
    lim = max(3, max(abs(x) for x in g) * 1.15)
    ax.set_xlim(-lim, lim)
axes[0].set_yticks(range(len(order)), [LABEL.get(n, NN.get(n)) for n in order], fontsize=9)
axes[0].invert_yaxis()
fig.supxlabel("test MAE, % better than persistence", fontsize=10, color=COLORS["ink2"])
fig.tight_layout()
show(fig, "06_baselines.png", "Test MAE change against persistence for the new baselines and the three neural "
     "networks (mean of 3 seeds). Each panel has its own scale. Pure seasonal forecasts are left out: they are "
     "far worse than persistence (see the table).")

fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
for ax, ref, title in zip(axes, ("persistence", "best"), ("against persistence", "against the best baseline")):
    for i, key in enumerate(NN):
        xs, gs, los, his = [], [], [], []
        for j, case in enumerate(CASES):
            b = "persistence" if ref == "persistence" else BEST[case]
            g, lo, hi, _ = SIG[(case, key, b)]
            xs.append(j + (i - 1) * 0.22); gs.append(g); los.append(g - lo); his.append(hi - g)
        ax.errorbar(xs, gs, yerr=[los, his], fmt="o", color=HUE[key], label=NN[key], capsize=3, markersize=5)
    ax.axhline(0, color=COLORS["ink2"], linewidth=1)
    ax.set_xticks(range(len(CASES)), [case_label(*cs).replace(" ", "\n", 1) for cs in CASES], fontsize=8.5)
    ax.set_title(f"Test MAE change {title}"); ax.grid(axis="x", visible=False)
axes[0].set_ylabel("% better (95% block-bootstrap interval)")
axes[0].legend(frameon=False, fontsize=8.5)
fig.tight_layout()
show(fig, "06_significance.png", "Each point is a 3-seed mean forecast; the bars are 95% intervals from a "
     "moving-block bootstrap with one-day blocks. An interval that crosses zero means the difference could be chance.")

# %% write the results page and open it in the browser
report.finish(open_browser="--no-browser" not in sys.argv)
