"""07 Better inputs: Phase 9.

Gives the CNN-LSTM information the Phase 4-6 models did not have, one change at a time:
 time       sin/cos of the time of day and day of week of the target
 seas       the seasonal value of every flow (mean of the same time on the last 7 days,
            the best reference in Phase 8) at the target time and at the last input time,
            minus the last value; each flow gets its own two weights for these (a learned
            per-flow version of Phase 8's blend), added to the network's correction
 time_seas  both
 long       a longer input window: 36 steps on Abilene (3 hours), 24 on GÉANT (6 hours)
 long_*     the long window plus the best extra inputs, trained only if BOTH the long
            window and an extra input beat the base on validation by more than 0.5%
            (the Phase 4 rule)
Everything else is the Phase 4-6 setup: residual CNN-LSTM, original-units MAE loss,
Phase 4 settings, 3 seeds. The base is the saved Phase 4-6 models (window 12, no extras).
Extra inputs join right before the final layer, so every model is still exactly
persistence before training (see common._output).

Same targets, splits and scaler as before. A longer window cannot be built for a few
targets just after a data gap, so ALL methods in a case are scored on the targets the
longer window keeps (about 99%); the base's numbers therefore differ slightly from Phase 8.
Selection is on validation only; test is reported for every variant (no test-based choice).

Run `python scripts/07_features.py --only abilene` (or geant, or one case such as abilene:6)
to train without writing the page (to train several in parallel); then run it without options.
On a GPU (WSL) the models are named p9g_* and the base is retrained on the GPU as well,
because GPU and CPU training give slightly different numbers.
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

ONLY = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None   # "abilene" or "abilene:6"
GPU = bool(tf.config.list_physical_devices("GPU"))
PREFIX = "p9g" if GPU else "p9"     # GPU and CPU training give slightly different numbers: never mix them
report = None if ONLY else Report("07_features", "Phase 9: Better inputs")
section = (lambda t: print(f"\n== {t} ==")) if report is None else report.section
section("Project status")
set_seeds()
status()
print(f"training on {'the GPU (WSL); the base is retrained on the GPU too, so every comparison is GPU vs GPU' if GPU else 'the CPU; the base is the saved Phase 4-6 models'}")

import matplotlib.pyplot as plt
use_plot_style()
RETRAIN = False
EPOCHS, PATIENCE, LR = 200, 15, 1e-3
SEEDS = (42, 43, 44)
CASES = [("abilene", 1), ("abilene", 6), ("abilene", 12), ("geant", 1), ("geant", 4)]
if ONLY:
    CASES = [c for c in CASES if f"{c[0]}:{c[1]}" == ONLY or c[0] == ONLY]
DS_NAME = {"abilene": "Abilene", "geant": "GÉANT"}
UNIT = {"abilene": MBIT, "geant": 1 / 1000}
STEP_MIN = {"abilene": 5, "geant": 15}
N_F = {"abilene": N_FLOWS, "geant": GEANT_FLOWS}
CAPS = {"abilene": None, "geant": GEANT_MAX_FLOW_KBPS}
SEASONAL = "seasonal_mean7"
LONG = {"abilene": 36, "geant": 24}
VARIANTS = {    # key: (label, long window?, seasonal, time)
    "base": ("window 12, no extras (Phases 4-6)", False, None, False),
    "time": ("+ time of day / week", False, None, True),
    "seas": ("+ seasonal values", False, SEASONAL, False),
    "time_seas": ("+ time + seasonal", False, SEASONAL, True),
    "long": ("longer window", True, None, False),
}
RULE = 0.5          # % on validation needed to count as an improvement (Phase 4 rule)
case_label = lambda ds, h: f"{DS_NAME[ds]} {h * STEP_MIN[ds]} min"
mae = lambda y, p: float(np.mean(np.abs(p - y)))
gain = lambda m, base: (1 - m / base) * 100


def label(ds, key):
    lab = VARIANTS[key][0] if key in VARIANTS else "longer window " + VARIANTS[key[5:]][0]
    return lab.replace("longer window", f"window {LONG[ds]}")


DATA = {}


def data_key(ds, h, key):
    long_, seas, tm = VARIANTS[key][1:] if key in VARIANTS else (True,) + VARIANTS[key[5:]][2:]
    return (ds, h, long_, seas, tm)


def data_for(ds, h, key):
    k = data_key(ds, h, key)
    if k not in DATA:
        long_, seas, tm = k[2:]
        DATA[k] = context_dataset(ds, h, window=LONG[ds] if long_ else WINDOW, seasonal=seas, use_time=tm)
    return DATA[k]


def drop_data(ds, h, key):
    """Free a variant's arrays (several hundred MB each); data_for rebuilds them if needed again."""
    DATA.pop(data_key(ds, h, key), None)
    import gc
    gc.collect()


def inputs(d, i):
    return model_inputs(d, i)


def run_variant(ds, h, key):
    """Train (or load) the 3 seeds; return 3-seed mean forecasts (raw) and single-seed val/test MAE."""
    d = data_for(ds, h, key)
    sc = d["scaler"]
    preds = {1: [], 2: []}
    if key == "base" and not GPU:
        names = saved_model_names(ds, h, "cnn_lstm", SEEDS)     # the Phase 4-6 CPU models
    else:
        names = [f"{PREFIX}_{ds}_h{h}_{key}_s{s}" for s in SEEDS]
    ref = float(data_for(ds, h, "base")["y_raw"][0].mean())          # the Phase 4-6 loss scale
    loss = make_loss("mae_raw", sc, ref)
    for seed, name in zip(SEEDS, names):
        if key == "base" and not GPU:
            model = tf.keras.models.load_model(MODELS / f"{name}.keras", compile=False)
        else:
            nc = 0 if d["C"][0] is None else d["C"][0].shape[1]
            build = lambda: build_cnn_lstm(window=d["X"][0].shape[1], n_flows=N_F[ds], residual=True,
                                           n_context=nc, seasonal=d["S"][0] is not None)
            print(f"  {case_label(ds, h)}, {label(ds, key)}, seed {seed}")
            model, _, _ = fit_or_load(build, name, inputs(d, 0), d["y"][0], inputs(d, 1), d["y"][1],
                                      retrain=RETRAIN, seed=seed, epochs=EPOCHS, patience=PATIENCE, lr=LR, loss=loss)
        for i in (1, 2):
            preds[i].append(sc.inverse_transform(predict(model, inputs(d, i), scaler=sc, max_raw=CAPS[ds])).astype(np.float64))
        tf.keras.backend.clear_session()
    out = {"val": np.mean(preds[1], 0), "test": np.mean(preds[2], 0), "kept": d["kept"],
           "seeds": {s: [mae(d["y_raw"][i], p) for p in preds[i]] for s, i in (("val", 1), ("test", 2))},
           "pers_test": mae(d["y_raw"][2], d["last_raw"][2])}       # persistence on this variant's own targets
    del d
    if key != "base":
        drop_data(ds, h, key)
    return out


# %% 1. train (or load) every variant
section("1. Training")
RES = {}
for ds, h in CASES:
    case = (ds, h)
    RES[case] = {key: run_variant(ds, h, key) for key in VARIANTS}
    # common targets: the ones the long window keeps (positions in the window-12 samples)
    keep = RES[case]["long"]["kept"]
    b = data_for(ds, h, "base")
    on_common = lambda key, s, i: RES[case][key][s] if VARIANTS.get(key, (0, True))[1] else RES[case][key][s][keep[i]]
    y_va, y_te = b["y_raw"][1][keep[1]], b["y_raw"][2][keep[2]]
    pv = mae(y_va, b["last_raw"][1][keep[1]])
    vg = {key: gain(mae(y_va, on_common(key, "val", 1)), pv) for key in VARIANTS}
    extras = max(("time", "seas", "time_seas"), key=lambda k: vg[k])
    print(f"{case_label(ds, h)} validation gain vs persistence: " + ", ".join(f"{k} {g:+.2f}%" for k, g in vg.items()))
    if vg["long"] - vg["base"] > RULE and vg[extras] - vg["base"] > RULE:
        combo = "long_" + extras
        print(f"  both the long window and '{extras}' help by more than {RULE}%: training {combo}")
        RES[case][combo] = run_variant(ds, h, combo)
    else:
        print(f"  no combination trained (long window {vg['long'] - vg['base']:+.2f}%, best extra '{extras}' "
              f"{vg[extras] - vg['base']:+.2f}% vs base; rule: both > {RULE}%)")
    drop_data(ds, h, "base")

if ONLY:
    print(f"\ntraining for {ONLY} done; run the script without --only to write the page")
    raise SystemExit

# %% 2. results
section("2. Results (common targets)")
print("MAE change vs persistence on the same targets, positive = better. Mean of 3 seeds' forecasts;\n"
      "'seeds' gives each single seed's test change. 'chosen' = best validation MAE.\n"
      "Phase 8's best baseline is shown on the same targets for reference.")
CHOSEN, TAB = {}, {}
for ds, h in CASES:
    case, u = (ds, h), UNIT[ds]
    keep = RES[case]["long"]["kept"]
    b = data_for(ds, h, "base")
    y = {1: b["y_raw"][1][keep[1]], 2: b["y_raw"][2][keep[2]]}
    pers = {1: b["last_raw"][1][keep[1]], 2: b["last_raw"][2][keep[2]]}
    p8 = np.load(PROCESSED / "phase8" / f"{ds}_h{h}.npz")
    best8 = str(p8["best"])
    F = {}
    for key, r in RES[case].items():
        al = VARIANTS.get(key, (0, True))[1]
        F[key] = {i: (r[s] if al else r[s][keep[i]]) for s, i in (("val", 1), ("test", 2))}
    F["p8"] = {i: p8[f"{best8}_{s}"].astype(np.float64)[keep[i]] for s, i in (("val", 1), ("test", 2))}
    pv, pt = mae(y[1], pers[1]), mae(y[2], pers[2])
    CHOSEN[case] = min((k for k in RES[case]), key=lambda k: mae(y[1], F[k][1]))
    TAB[case] = {"y": y, "pers": pers, "F": F, "pv": pv, "pt": pt, "best8": best8}
    print(f"\n{case_label(ds, h)}: {len(y[2])} test targets ({len(y[2]) / len(b['y_raw'][2]):.1%} of all); "
          f"persistence MAE val {pv * u:.3f}, test {pt * u:.3f} Mbit/s")
    print(f"  {'':40s} {'val':>8s} {'test':>8s}   seeds (test)")
    print(f"  {'Phase 8 best baseline: ' + best8:40s} {gain(mae(y[1], F['p8'][1]), pv):+7.2f}% {gain(mae(y[2], F['p8'][2]), pt):+7.2f}%")
    for key in RES[case]:
        mv, mt = mae(y[1], F[key][1]), mae(y[2], F[key][2])
        # single-seed changes, on each variant's own targets
        s_te = RES[case][key]["seeds"]["test"]
        own_pt = RES[case][key]["pers_test"]
        seeds = ", ".join(f"{gain(m, own_pt):+.1f}" for m in s_te)
        tag = "  chosen" if key == CHOSEN[case] else ""
        print(f"  {label(ds, key):40s} {gain(mv, pv):+7.2f}% {gain(mt, pt):+7.2f}%   {seeds}{tag}")
        for split, i in (("val", 1), ("test", 2)):
            log_result(f"{PREFIX}_{key}_mean3", evaluate(y[i], F[key][i]), horizon=h, dataset=ds,
                       notes=f"Phase 9, {split} split, all flows, targets the window-{LONG[ds]} data keeps")
    del b
    drop_data(ds, h, "base")

# %% 3. significance of the chosen variant
section("3. Is the chosen input set better? (test split)")
print("Same tests as Phase 8: Diebold-Mariano (Newey-West, one day of lags) and a 95% moving-block\n"
      "bootstrap interval (one-day blocks) of the MAE change. Positive = the first is better.")
SIG = {}
for ds, h in CASES:
    case = (ds, h)
    T = TAB[case]
    day = STEPS_PER_DAY[ds]
    ch = CHOSEN[case]
    err = lambda p: np.abs(p - T["y"][2]).mean(1)
    print(f"\n{case_label(ds, h)}: chosen '{label(ds, ch)}'")
    for other, name in (("base", "base CNN-LSTM"), ("pers", "persistence"), ("p8", "Phase 8 best baseline")):
        if other == ch:
            continue
        po = T["pers"][2] if other == "pers" else T["F"][other][2]
        _, _, p = dm_test(err(T["F"][ch][2]), err(po), day)
        g, lo, hi = block_bootstrap_gain(err(T["F"][ch][2]), err(po), day)
        SIG[(case, other)] = (g, lo, hi, p)
        verdict = "significant" if p < 0.05 and lo * hi > 0 else "not significant"
        print(f"  vs {name:24s} {g:+6.2f}%  [{lo:+5.2f}, {hi:+5.2f}]  DM p = {p:.3f}  {verdict}")

# %% 4. figures
section("4. Figures")
keys = list(VARIANTS)
HUE = {"base": COLORS["ink2"], "time": COLORS["yellow"], "seas": COLORS["aqua"], "time_seas": COLORS["blue"],
       "long": COLORS["orange"]}
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
for ax, i, title in zip(axes, (1, 2), ("validation", "test")):
    for j, key in enumerate(keys + ["combo"]):
        xs, gs = [], []
        for c_i, case in enumerate(CASES):
            T = TAB[case]
            k = key if key != "combo" else next((x for x in RES[case] if x.startswith("long_")), None)
            if k is None:
                continue
            base_m = T["pv"] if i == 1 else T["pt"]
            xs.append(c_i + (j - 2.5) * 0.14)
            gs.append(gain(mae(T["y"][i], T["F"][k][i]), base_m))
        if xs:
            ax.bar(xs, gs, 0.13, color=HUE.get(key, COLORS["violet"]),
                   label=VARIANTS[key][0] if key in VARIANTS else "longer window + best extra")
    ax.axhline(0, color=COLORS["ink2"], linewidth=1)
    ax.set_xticks(range(len(CASES)), [case_label(*c).replace(" ", "\n", 1) for c in CASES], fontsize=8.5)
    ax.set_title(f"{title}: MAE change vs persistence"); ax.grid(axis="x", visible=False)
axes[0].set_ylabel("% better than persistence")
axes[0].legend(frameon=False, fontsize=8, loc="upper left")
fig.tight_layout()
show(fig, "07_features.png", "MAE change against persistence for each input set (CNN-LSTM, mean of 3 seeds' "
     "forecasts), on the targets every variant can forecast. 'longer window' is 3 hours on Abilene and 6 hours on GÉANT.")

# %% write the results page and open it in the browser
report.finish(open_browser="--no-browser" not in sys.argv)
