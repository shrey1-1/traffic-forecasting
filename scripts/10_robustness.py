"""10 Robustness: Phase 12.

1. Rolling origin. Phases 9-10 rest on one test period (the last 20% of the data), and Phase 10's
   fallback rule on one flow's bursts in it. Here the comparison is repeated on three EARLIER
   periods, each with its own chronological split (train = everything before validation), its own
   scaler, its own rule threshold (chosen on that fold's validation exactly as in Phase 10) and
   3 seeds per model. Test periods: 40-55%, 55-70% and 70-80% of the samples (the original test is
   80-100%). At 60 minutes on both networks, where the Phase 9-10 effects were largest.
2. Ablations with 3 seeds. Phase 5's ablations (one conv layer, no pooling, one LSTM layer) had one
   seed. Retrained here with 3 seeds on the GPU and compared with the GPU base (p9g_*_base).
3. Error analysis rerun (Phase 7's move-size bins and damping) for the seasonal + rule model.
Nothing is chosen on test.

Run on the GPU environment. `--only folds:abilene`, `--only folds:geant` or `--only ablations`
trains without writing the page. Output: results/report.html
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
if not tf.config.list_physical_devices("GPU"):
    raise SystemExit("Phase 12 compares against the GPU models (p9g_*): run it on the GPU (scripts/wsl/run.sh)")
report = None if ONLY else Report("10_robustness", "Phase 12: Robustness")
section = (lambda t: print(f"\n== {t} ==")) if report is None else report.section
section("Project status")
set_seeds()
status()

import matplotlib.pyplot as plt
use_plot_style()
RETRAIN = False
EPOCHS, PATIENCE, LR = 200, 15, 1e-3
SEEDS = (42, 43, 44)
FOLDS = {"A": (0.30, 0.40, 0.55), "B": (0.45, 0.55, 0.70), "C": (0.60, 0.70, 0.80)}
FOLD_CASES = [("abilene", 12), ("geant", 4)]
ABL = {"1conv": ("1 conv layer", dict(n_conv=1)), "nopool": ("no pooling", dict(pool=False)),
       "1lstm": ("1 LSTM layer", dict(units=(100,)))}
KS, TIE = (3, 4, 5, 6, 8), 0.05
F86 = 86
DS_NAME = {"abilene": "Abilene", "geant": "GÉANT"}
STEP_MIN = {"abilene": 5, "geant": 15}
N_F = {"abilene": N_FLOWS, "geant": GEANT_FLOWS}
CAPS = {"abilene": None, "geant": GEANT_MAX_FLOW_KBPS}
case_label = lambda ds, h: f"{DS_NAME[ds]} {h * STEP_MIN[ds]} min"
mae = lambda y, p: float(np.mean(np.abs(p - y)))
gain = lambda m, b: (1 - m / b) * 100


def train_or_load(name, build, d, ref):
    loss = make_loss("mae_raw", d["scaler"], ref)
    model, _, _ = fit_or_load(build, name, model_inputs(d, 0), d["y"][0], model_inputs(d, 1), d["y"][1],
                              retrain=RETRAIN, seed=int(name[-2:]), epochs=EPOCHS, patience=PATIENCE, lr=LR, loss=loss)
    return model


def forecasts(model, d, ds):
    return {i: d["scaler"].inverse_transform(predict(model, model_inputs(d, i), scaler=d["scaler"],
                                                     max_raw=CAPS[ds])).astype(np.float64) for i in (1, 2)}


def with_rule(d, seas_mean, ds):
    """Phase 10's rule: persistence where |seasonal - last| > k training sd; k chosen on validation."""
    sd = d["S"][0][:, 0].std(0)
    sd = np.where(sd > 0, sd, 1)
    z = {i: np.abs(d["S"][i][:, 0]) / sd for i in (1, 2)}
    pv = mae(d["y_raw"][1], d["last_raw"][1])
    sc = {k: gain(mae(d["y_raw"][1], np.where(z[1] > k, d["last_raw"][1], seas_mean[1])), pv) for k in KS}
    k = min(kk for kk in KS if sc[kk] >= max(sc.values()) - TIE)
    return {i: np.where(z[i] > k, d["last_raw"][i], seas_mean[i]) for i in (1, 2)}, k


def date_of(ds, t):
    """Approximate calendar day of a real-time index (Abilene: days from 1 Mar 2004; GÉANT: from 1 Jan 2005)."""
    from datetime import datetime, timedelta
    d0 = datetime.fromisoformat(ABILENE_WEEK_STARTS[0]) if ds == "abilene" else datetime(*common.GEANT_START)
    return (d0 + timedelta(days=int(t // STEPS_PER_DAY[ds]))).strftime("%d %b")


# %% 1. rolling origin
section("1. Rolling origin: three earlier test periods")
FOLD_RES = {}
if ONLY is None or ONLY.startswith("folds"):
    for ds, h in FOLD_CASES:
        if ONLY and ONLY != f"folds:{ds}":
            continue
        for fname, split in FOLDS.items():
            print(f"\n{case_label(ds, h)}, fold {fname} (train 0-{split[0]:.0%}, val to {split[1]:.0%}, test to {split[2]:.0%})")
            d = fold_dataset(ds, h, split, seasonal="seasonal_mean7")
            ref = float(d["y_raw"][0].mean())
            P = {"base": [], "seas": []}
            for seed in SEEDS:
                for key in ("base", "seas"):
                    name = f"p12g_{ds}_h{h}_f{fname}_{key}_s{seed}"
                    print(f"  {key}, seed {seed}")
                    seasonal = key == "seas"
                    m = train_or_load(name, lambda: build_cnn_lstm(n_flows=N_F[ds], residual=True, seasonal=seasonal),
                                      {**d, "S": d["S"] if seasonal else [None] * 3}, ref)
                    P[key].append(forecasts(m, {**d, "S": d["S"] if seasonal else [None] * 3}, ds))
                    tf.keras.backend.clear_session()
            mean = {key: {i: np.mean([p[i] for p in P[key]], 0) for i in (1, 2)} for key in P}
            rule, k = with_rule(d, mean["seas"], ds)
            FOLD_RES[(ds, fname)] = {"y": d["y_raw"], "last": d["last_raw"], "base": mean["base"], "seas": mean["seas"],
                                     "rule": rule, "k": k, "times": d["times"][2],
                                     "seeds": {key: [mae(d["y_raw"][2], p[2]) for p in P[key]] for key in P}}
            del d
if ONLY and ONLY.startswith("folds"):
    print(f"\ntraining for {ONLY} done")
    raise SystemExit

# %% 2. ablations with 3 seeds
section("2. Ablations, 3 seeds (Abilene)")
ABL_RES = {}
if ONLY is None or ONLY.startswith("ablations"):
    for h in (1, 6, 12):
        if ONLY and ":" in ONLY and int(ONLY.split(":")[1]) != h:      # e.g. --only ablations:12
            continue
        d = context_dataset("abilene", h)
        ref = float(d["y_raw"][0].mean())
        for key, (lab, kw) in ABL.items():
            ps = []
            for seed in SEEDS:
                print(f"  Abilene {h * 5} min, {lab}, seed {seed}")
                m = train_or_load(f"p12g_abilene_h{h}_{key}_s{seed}",
                                  lambda: build_cnn_lstm(residual=True, **kw), d, ref)
                ps.append(forecasts(m, d, "abilene"))
                tf.keras.backend.clear_session()
            ABL_RES[(h, key)] = ps
        ABL_RES[(h, "full")] = [forecasts(load_saved(f"p9g_abilene_h{h}_base_s{s}"), d, "abilene") for s in SEEDS]
        ABL_RES[(h, "data")] = (d["y_raw"], d["last_raw"])
        del d
if ONLY and ONLY.startswith("ablations"):
    print(f"\ntraining for {ONLY} done")
    raise SystemExit

# %% 3. rolling-origin results
section("3. Rolling-origin results (test of each fold)")
print("MAE change vs persistence (positive = better), mean of 3 seeds' forecasts. 'original' = the Phase 10 split\n"
      "(test = last 20%). Significance of seasonal + rule vs base: Diebold-Mariano with one day of lags and a 95%\n"
      "day-block bootstrap interval.")
ROWS = []
for ds, h in FOLD_CASES:
    print(f"\n{case_label(ds, h)}")
    hdr = f"  {'fold':8s} {'test period':16s} {'pers. MAE':>9s} {'base':>7s} {'seasonal':>9s} {'+ rule':>7s} {'k':>2s}"
    if ds == "abilene":
        hdr += f" {'rule w/o 86':>11s} {'rule, 86':>9s}"
    print(hdr + "   rule vs base")
    # the original split, from the Phase 9 models (same data and procedure as Phase 10)
    d = context_dataset(ds, h, seasonal="seasonal_mean7")
    b = context_dataset(ds, h)
    ens = lambda pref, dd: {i: np.mean([forecasts(load_saved(f"{pref}_s{s}"), dd, ds)[i] for s in SEEDS], 0) for i in (1, 2)}
    orig_seas = ens(f"p9g_{ds}_h{h}_seas", d)
    orig_rule, k0 = with_rule(d, orig_seas, ds)
    folds = [("original", {"y": d["y_raw"], "last": d["last_raw"], "base": ens(f"p9g_{ds}_h{h}_base", b),
                           "seas": orig_seas, "rule": orig_rule, "k": k0, "times": d["times"][2]})]
    folds += [(f, FOLD_RES[(ds, f)]) for f in FOLDS]
    tf.keras.backend.clear_session()
    del d, b
    for fname, r in folds:
        y, last = r["y"][2], r["last"][2]
        pt = mae(y, last)
        unit = MBIT if ds == "abilene" else 1 / 1000
        gb, gs, gr = (gain(mae(y, r[key][2]), pt) for key in ("base", "seas", "rule"))
        period = f"{date_of(ds, r['times'][0])}-{date_of(ds, r['times'][-1])}"
        line = f"  {fname:8s} {period:16s} {pt * unit:9.3f} {gb:+6.1f}% {gs:+8.1f}% {gr:+6.1f}% {r['k']:>2d}"
        if ds == "abilene":
            o = np.arange(N_FLOWS) != F86
            line += (f" {gain(mae(y[:, o], r['rule'][2][:, o]), mae(y[:, o], last[:, o])):+10.1f}%"
                     f" {gain(mae(y[:, F86], r['rule'][2][:, F86]), mae(y[:, F86], last[:, F86])):+8.1f}%")
        ea, eb = np.abs(r["rule"][2] - y).mean(1), np.abs(r["base"][2] - y).mean(1)
        _, _, p = dm_test(ea, eb, STEPS_PER_DAY[ds])
        g_, lo, hi = block_bootstrap_gain(ea, eb, STEPS_PER_DAY[ds])
        print(line + f"   {g_:+5.1f}% [{lo:+5.1f}, {hi:+5.1f}] p={p:.3f}")
        ROWS.append((ds, fname, gb, gs, gr))
        log_result(f"p12g_rule_mean3_{fname}", evaluate(y, r["rule"][2]), horizon=h, dataset=ds,
                   notes=f"Phase 12, rolling origin fold {fname}, test split, seasonal + rule (k={r['k']})")
        log_result(f"p12g_base_mean3_{fname}", evaluate(y, r["base"][2]), horizon=h, dataset=ds,
                   notes=f"Phase 12, rolling origin fold {fname}, test split, base")

# %% 4. ablation results
section("4. Ablations with 3 seeds (Abilene, test)")
print("MAE change vs persistence, mean of 3 seeds' forecasts; single seeds in brackets. 'full' is the GPU base model.")
print(f"  {'':16s} {'5 min':>24s} {'30 min':>24s} {'60 min':>24s}")
for key, lab in (("full", "full CNN-LSTM"),) + tuple((k, v[0]) for k, v in ABL.items()):
    cells = []
    for h in (1, 6, 12):
        y, last = ABL_RES[(h, "data")][0][2], ABL_RES[(h, "data")][1][2]
        pt = mae(y, last)
        ps = ABL_RES[(h, key)]
        mean = gain(mae(y, np.mean([p[2] for p in ps], 0)), pt)
        singles = ", ".join(f"{gain(mae(y, p[2]), pt):+.1f}" for p in ps)
        cells.append(f"{mean:+5.2f}% ({singles})")
    print(f"  {lab:16s} " + " ".join(f"{c:>24s}" for c in cells))

# %% 5. error analysis rerun
section("5. Error analysis rerun: where the seasonal + rule model gains")
print("As Phase 7: move = |true - last| / the flow's mean training traffic, binned by percentile of the pooled\n"
      "test moves; MAE change vs persistence per bin (positive = better). Damping = slope of the forecast change\n"
      "on the true change (1 = whole move forecast, 0 = persistence).")
BINS = (0, 50, 80, 95, 99, 100)
NAMES = ("smallest 50%", "50-80%", "80-95%", "95-99%", "top 1%")
for ds, h in (("abilene", 1), ("abilene", 6), ("abilene", 12), ("geant", 1), ("geant", 4)):
    d = context_dataset(ds, h, seasonal="seasonal_mean7")
    b = context_dataset(ds, h)
    ens = lambda pref, dd: {i: np.mean([forecasts(load_saved(f"{pref}_s{s}"), dd, ds)[i] for s in SEEDS], 0) for i in (1, 2)}
    rule, _ = with_rule(d, ens(f"p9g_{ds}_h{h}_seas", d), ds)
    base = ens(f"p9g_{ds}_h{h}_base", b)
    tf.keras.backend.clear_session()
    y, last = d["y_raw"][2], d["last_raw"][2]
    move = np.abs(y - last) / np.maximum(d["y_raw"][0].mean(0), 1e-9)
    edges = np.percentile(move, BINS)
    cells = {"base": [], "rule": []}
    for lo_, hi_ in zip(edges[:-1], edges[1:]):
        m = (move >= lo_) & (move <= hi_)
        for key, p in (("base", base[2]), ("rule", rule[2])):
            cells[key].append(gain(np.abs(p - y)[m].mean(), np.abs(last - y)[m].mean()))
    damp = lambda p: float(((p - last) * (y - last)).sum() / ((y - last) ** 2).sum())
    print(f"\n{case_label(ds, h)}: " + "  ".join(f"{n}" for n in NAMES))
    for key, lab in (("base", "base         "), ("rule", "seasonal+rule")):
        print(f"  {lab} " + "  ".join(f"{v:+7.1f}%" for v in cells[key]) + f"   damping {damp(base[2] if key == 'base' else rule[2]):.3f}")
    del d, b

# %% 6. figure
section("6. Figure")
fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.9), sharey=True)
for ax, ds in zip(axes, ("abilene", "geant")):
    rows = [r for r in ROWS if r[0] == ds]
    x = np.arange(len(rows))
    for n, (j, lab, col) in enumerate(((2, "base", COLORS["ink2"]), (3, "seasonal", COLORS["aqua"]),
                                       (4, "seasonal + rule", COLORS["violet"]))):
        ax.bar(x + (n - 1) * 0.26, [r[j] for r in rows], 0.25, color=col, label=lab)
    ax.axhline(0, color=COLORS["ink2"], linewidth=1)
    ax.set_xticks(x, [r[1] for r in rows])
    ax.set_title(f"{DS_NAME[ds]} 60 min: test MAE change vs persistence, by fold"); ax.grid(axis="x", visible=False)
axes[0].set_ylabel("% better than persistence"); axes[0].legend(frameon=False, fontsize=8)
fig.tight_layout()
show(fig, "10_rolling_origin.png", "Each fold has its own training period, scaler, rule threshold and 3 seeds; "
     "'original' is the split used in Phases 4-11 (test = last 20%).")

report.finish(open_browser="--no-browser" not in sys.argv)
