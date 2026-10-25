"""04 Analysis: Phases 6 and 7.

Phase 6 (this version): from flow forecasts to link loads and congestion
warnings on Abilene. Forecast flows are mapped onto the 30 backbone links with
the routing matrix (link load = A @ flows), turned into utilisation, and
compared with a warning threshold. Two forecasters are compared: persistence
(the reactive baseline: "the link will look as it looks now") and the Phase 4/5
CNN-LSTM (mean of its 3 seeds). GÉANT and the peak error analysis come later.

Threshold: the plan said 80%, but no Abilene backbone link ever reaches 80%
(busiest: 75% in test, 70% in validation), so 80% gives zero events. The
primary threshold is 50%, fixed before looking at any warning results, with
40% and 60% shown as a sensitivity check. Only the test split is scored.
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
report = Report("04_analysis", "Phase 6: Link loads and congestion warnings")
report.section("Project status")
set_seeds()
status()

import matplotlib.pyplot as plt
use_plot_style()
HORIZONS = (1, 6, 12)
SEEDS = (42, 43, 44)
THRESHOLDS = (0.4, 0.5, 0.6)
PRIMARY = 0.5


def model_name(h, seed):
    if h == 1:
        return f"p4_cnn_lstm_f64_k3_u100-50_d20_lr0.001_s{seed}"
    return f"p5_h{h}_cnn_lstm_s{seed}"


# %% 1. links, and the flow-to-link mapping
report.section("1. Links and the routing matrix")
R = load_abilene_routing()
A = R["A"]
backbone = np.array(R["link_type"]) == 0
A_bb = A[backbone]
link_names = np.array(R["links"])[backbone]
cap_kbps = R["capacity_kbps"][backbone]
print(f"{backbone.sum()} backbone links ({(cap_kbps == 9.92e6).sum()} at 9.92 Gbit/s, "
      f"{(cap_kbps == 2.48e6).sum()} at 2.48 Gbit/s); every flow follows exactly one path")
flows_per_link = (A_bb > 0).sum(1)
print(f"flows per link: min {flows_per_link.min()}, median {int(np.median(flows_per_link))}, max {flows_per_link.max()}")


def utilisation(flows_raw):
    """Flows (n, 144) in raw units per 5 minutes -> backbone utilisation (n, 30), 0..1."""
    kbps = flows_raw.astype(np.float64) @ A_bb.T * ABILENE_UNIT_BYTES * 8 / 300 / 1000
    return kbps / cap_kbps


# %% 2. forecasts at each horizon, mapped onto links
report.section("2. Forecasts mapped onto links")
steps_all = np.load(PROCESSED / "target_steps.npy")
D = {}
for h in HORIZONS:
    d = PROCESSED if h == 1 else PROCESSED / f"h{h}"
    X_tr, y_tr, X_va, y_va, X_te, y_te = load_splits(d)
    scaler = FlowScaler.load(d / "scaler.npz")
    steps = np.load(d / "target_steps.npy")[len(y_tr) + len(y_va):]
    pred = np.mean([scaler.inverse_transform(predict(
        tf.keras.models.load_model(MODELS / f"{model_name(h, s)}.keras", compile=False), X_te, scaler=scaler))
        for s in SEEDS], axis=0)
    D[h] = {"steps": steps,
            "true": utilisation(scaler.inverse_transform(y_te)),
            "pers": utilisation(scaler.inverse_transform(persistence(X_te))),
            "model": utilisation(pred)}
    mae_pp = lambda k: np.abs(D[h][k] - D[h]["true"]).mean() * 100
    print(f"horizon {h:2d}: link utilisation MAE  persistence {mae_pp('pers'):.3f} pp   "
          f"CNN-LSTM {mae_pp('model'):.3f} pp   ({(1 - mae_pp('model') / mae_pp('pers')) * 100:+.1f}%)")
print("(pp = percentage points of link capacity, over all 30 backbone links)")

# %% 3. congestion warnings
report.section("3. Congestion warnings")


prf, onsets = warn_prf, warn_onsets


warn = {}   # warn[h][thr][name] = (precision, recall, f1, tp, fp, fn, onset_total, onset_warned)
for h in HORIZONS:
    warn[h] = {}
    for thr in THRESHOLDS:
        t_ev = D[h]["true"] >= thr
        on = onsets(t_ev, D[h]["steps"])
        warn[h][thr] = {}
        for name in ("pers", "model"):
            p_ev = D[h][name] >= thr
            warn[h][thr][name] = (*prf(t_ev, p_ev), int(on.sum()), int((on & p_ev).sum()))

for thr in THRESHOLDS:
    print(f"\nThreshold {thr:.0%}{'  (primary)' if thr == PRIMARY else ''}. Test split, warning = forecast utilisation >= threshold")
    print(f"  {'':8s} {'link-steps':>10s} {'onsets':>6s} | {'':12s} {'precision':>9s} {'recall':>7s} {'F1':>6s} {'onsets warned':>14s}")
    for h in HORIZONS:
        for name, label in (("pers", "persistence"), ("model", "CNN-LSTM")):
            p, r, f, tp, fp, fn, n_on, n_w = warn[h][thr][name]
            lead = f"{n_w}/{n_on}"
            first = f"h={h:2d} ({h * 5:2d} min)" if name == "pers" else ""
            ev = int((D[h]["true"] >= thr).sum()) if name == "pers" else ""
            on_s = n_on if name == "pers" else ""
            print(f"  {first:13s} {ev!s:>5s} {on_s!s:>6s} | {label:12s} {p:9.2f} {r:7.2f} {f:6.2f} {lead:>14s}")

# %% 4. what the events are
report.section("4. What the congestion events are")
h = 6
t_ev = D[h]["true"] >= PRIMARY
links_hit = np.where(t_ev.any(0))[0]
print(f"At {PRIMARY:.0%}, {t_ev.sum()} link-steps on {len(links_hit)} of 30 links, in {t_ev.any(1).sum()} of "
      f"{len(t_ev)} test steps ({t_ev.any(1).mean() * 100:.1f}%).")
print("Links hit, by number of congested steps:")
for j in links_hit[np.argsort(t_ev.sum(0)[links_hit])[::-1]]:
    print(f"  {link_names[j]:16s} {t_ev[:, j].sum():5d} steps")
# how many of the events are explained by the one bursty flow?
LOSA_CHIN = 7 * 12 + 2
y_te6 = FlowScaler.load(PROCESSED / "h6" / "scaler.npz")
_, _, _, y_va6, _, y_te6_s = load_splits(PROCESSED / "h6")
flows6 = y_te6.inverse_transform(y_te6_s).astype(np.float64)
without = flows6.copy(); without[:, LOSA_CHIN] = 0
ev_without = utilisation(without) >= PRIMARY
print(f"\nWith the {ABILENE_NODES[7]}->{ABILENE_NODES[2]} flow set to zero, congested link-steps drop from "
      f"{t_ev.sum()} to {ev_without.sum()}: nearly all events come from its bursts.")

# %% 5. figures
report.section("5. Figures")
h = 6
t_ev = D[h]["true"] >= PRIMARY
j = int(np.argmax(t_ev.sum(0)))
steps = D[h]["steps"]
i0 = int(np.argmax(onsets(t_ev, steps)[:, j]))
sl = slice(max(i0 - 36, 0), i0 + 60)
fig, ax = plt.subplots(figsize=(10.5, 3.9))
x = (np.arange(len(steps))[sl] - i0) * 5
ax.plot(x, D[h]["true"][sl, j] * 100, color=COLORS["ink2"], linewidth=2.4, label="actual")
ax.plot(x, D[h]["pers"][sl, j] * 100, color=COLORS["orange"], linewidth=1.4, label="persistence (reactive)")
ax.plot(x, D[h]["model"][sl, j] * 100, color=COLORS["violet"], linewidth=1.6, label="CNN-LSTM, 30 min ahead")
ax.axhline(PRIMARY * 100, color="red", linewidth=1, linestyle=":", label=f"{PRIMARY:.0%} threshold")
ax.axvline(0, color=COLORS["ink2"], linewidth=0.6, linestyle="--")
ax.set_xlabel("minutes from the congestion onset"); ax.set_ylabel("link utilisation, %")
ax.set_title(f"Link {link_names[j]}, test: one congestion episode, forecasts made 30 minutes earlier")
ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize=9)
fig.tight_layout()
show(fig, "04_episode.png", "A congestion onset on the busiest link. Each forecast line is the forecast for that "
     "moment, made 30 minutes before it. Persistence can only repeat the load it saw 30 minutes earlier.")

fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6))
w = 0.36
for ax, idx, title in ((axes[0], 2, "F1 of per-step warnings"), (axes[1], None, "Onsets warned in advance")):
    for k, (name, label, col) in enumerate((("pers", "persistence", COLORS["orange"]),
                                            ("model", "CNN-LSTM", COLORS["violet"]))):
        vals = [warn[hh][PRIMARY][name][2] if idx == 2 else
                warn[hh][PRIMARY][name][7] / max(warn[hh][PRIMARY][name][6], 1) for hh in HORIZONS]
        ax.bar(np.arange(3) + (k - 0.5) * w, vals, w, color=col, label=label)
    ax.set_xticks(range(3), [f"{hh * 5} min" for hh in HORIZONS]); ax.set_ylim(0, 1.05)
    ax.set_title(title + f" ({PRIMARY:.0%} threshold)"); ax.grid(axis="x", visible=False)
axes[0].legend(frameon=False, loc="lower left")
fig.tight_layout()
show(fig, "04_warnings.png", "Warning quality by horizon on the test split. Right: share of congestion onsets "
     "for which the forecast made that far ahead was already above the threshold.")

# %% write the results page and open it in the browser
report.finish()
