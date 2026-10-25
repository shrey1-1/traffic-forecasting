"""01 Data: Phase 1 (run once).

Exit condition: load_splits() returns the contracted shapes and the daily cycle is visible.

Reads the 24 Abilene weeks, confirms the realOD column, checks where the weeks
join, applies the missing-data policy, then scales, windows, splits and saves.
Figures go to results/figures/. Everything printed and plotted is also
collected into results/reports/01_data.html, which opens in the browser at the end.
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
report = Report("01_data", "Phase 1: Data layer")
report.section("Project status")
set_seeds()
status()

import matplotlib.pyplot as plt

# %% 1. load all 24 weeks of realOD (first run parses the files, later runs use the cache)
report.section("1. Load the 24 weeks")
weeks = load_abilene_raw()
print("weeks:", weeks.shape, "(week, step, flow), raw units of 100 bytes / 5 min")
print("NaN:", int(np.isnan(weeks).sum()), " negative:", int((weeks < 0).sum()))

# %% 2. missing-data policy (rules R1-R4 in common.abilene_clean)
report.section("2. Missing-data policy")
segments, info = abilene_clean(weeks)
print("removed steps:", info["removed"], "| duplicate first days in weeks", info["duplicate_weeks"])
for r, direction, a, b in info["router_outages"]:
    print(f"  router outage: all traffic {direction} {r}, {step_label(a)} .. {step_label(b - 1)}")
for i in info["impossible_steps"]:
    print(f"  impossible value (above link capacity) at {step_label(i)}")
print(f"kept {info['keep'].mean() * 100:.1f}% of steps in {len(segments)} segments:")
for start, seg in segments:
    print(f"  from {step_label(start)}: {len(seg)} steps ({len(seg) / 288:.1f} days)")

# removed steps become NaN, so figures and statistics below only use kept data
clean = weeks.reshape(-1, N_FLOWS).copy()
clean[~info["keep"]] = np.nan
clean = clean.reshape(weeks.shape)

use_plot_style()
BLUE, ORANGE, AQUA = COLORS["blue"], COLORS["orange"], COLORS["aqua"]
INK, INK2, GRID = COLORS["ink"], COLORS["ink2"], COLORS["grid"]
MB = MBIT                                  # raw units -> Mbit/s averaged over the 5 minutes

# %% 3. which column? realOD is the measurement; the other four are model estimates
# Evidence: the README lists realOD first in each group of 5; the other four are
# built from each router's totals (which all 5 columns share exactly), so they
# follow the real flow but drift from it (see the lower panel of the figure).
report.section("3. Which column is the real traffic")
cols = np.loadtxt(RAW / "X06.gz").reshape(STEPS_PER_WEEK, N_FLOWS, 5)
names = ["realOD", "simpleGravity", "simpleTomogravity", "generalGravity", "generalTomogravity"]
for c in range(5):
    print(f"{names[c]:19s} zero entries {np.mean(cols[:, :, c] == 0) * 100:5.2f}%")
print("realOD has more exact zeros: small flows really record nothing in some 5-minute slots,\n"
      "while the estimates are calculated from router totals and spread traffic more evenly.")

busy = int(np.argmax(np.nanmean(clean, axis=(0, 1))))
s, d = divmod(busy, 12)
print(f"busiest flow: #{busy} {ABILENE_NODES[s]} -> {ABILENE_NODES[d]}")

flow = 98   # NYCMng -> CHINng: a busy flow that crosses the backbone
fs, fd = divmod(flow, 12)
days = np.arange(STEPS_PER_WEEK) / 288
fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
ax[0].plot(days, weeks[5, :, busy] * MB, color=BLUE)
ax[0].set_title(f"Busiest flow, {ABILENE_NODES[s]} → {ABILENE_NODES[d]} (realOD, week X06)")
ax[0].set_ylabel("Mbit/s")
ax[1].plot(days, cols[:, flow, 0] * MB, color=BLUE, label="realOD (measured)")
ax[1].plot(days, cols[:, flow, 4] * MB, color=ORANGE, label="generalTomogravity (estimate)")
ax[1].set_title(f"{ABILENE_NODES[fs]} → {ABILENE_NODES[fd]}: measured vs model estimate")
ax[1].set_ylabel("Mbit/s"); ax[1].set_xlabel("day of week X06 (2004-05-01 onward)")
ax[1].legend(frameon=False, loc="upper left")
ax[1].set_xticks(range(8))
fig.tight_layout()
show(fig, "01_column_check.png", "Top: busiest flow over one week. Bottom: measured realOD against a model estimate.")

# %% 4. the daily cycle (exit condition): mean network total by time of day
report.section("4. Daily cycle (exit condition)")
tot = clean.sum(2) * MB / 1e3                                  # Gbit/s, (24, 2016), NaN where removed
profile = np.nanmean(tot.reshape(24, 7, 288), axis=(0, 1))
hours = np.arange(288) / 12
fig, ax = plt.subplots(figsize=(10, 3.5))
ax.plot(hours, profile, color=BLUE)
ax.set_title("Daily cycle: whole-network traffic by time of day, mean over all kept days")
ax.set_xlabel("hour of day (file clock; each week file assumed to start at 00:00)")
ax.set_ylabel("Gbit/s"); ax.set_xticks(range(0, 25, 3))
fig.tight_layout()
show(fig, "01_daily_cycle.png", "Whole-network traffic by time of day, averaged over every kept day.")
hhmm = lambda i: f"{i * 5 // 60:02d}:{i * 5 % 60:02d}"
print(f"peak/trough ratio: {profile.max() / profile.min():.2f} "
      f"(peak at {hhmm(profile.argmax())}, lowest at {hhmm(profile.argmin())})")

# %% 5. timeline figure: what was kept, what was removed and why
report.section("5. What was removed")
t = np.arange(tot.size) / 288
fig, ax = plt.subplots(figsize=(11, 4.2))
ax.plot(t, tot.ravel(), color=BLUE, linewidth=0.6, label="network total")
# long removals (days) as shaded bands; short ones (minutes) as marked lines, or they vanish
VIOLET, RED = COLORS["violet"], COLORS["red"]
shade = {"duplicate": (INK2, "duplicated day (R1)"), "flow": (ORANGE, "flow outage (R2)")}
for key, (colour, label) in shade.items():
    for i, (a, b) in enumerate(common._true_runs(info["reason"] == key)):
        ax.axvspan(a / 288, b / 288, color=colour, alpha=0.25, linewidth=0,
                   label=label if i == 0 else None)
marks = {"router": (VIOLET, "router outage (R3)"), "impossible": (RED, "impossible value (R4)")}
for key, (colour, label) in marks.items():
    for i, (a, b) in enumerate(common._true_runs(info["reason"] == key)):
        x = (a + b) / 2 / 288
        ax.axvline(x, color=colour, linewidth=1.5, zorder=3)
        ax.plot([x], [1.0], marker="v", markersize=9, color=colour, clip_on=False, zorder=4,
                transform=ax.get_xaxis_transform(), linestyle="none",
                label=label if i == 0 else None)
for w in range(1, 24):
    ax.axvline(w * 7, color=GRID, linewidth=0.8, zorder=0)
ax.set_xticks([w * 7 + 3.5 for w in range(24)])
ax.set_xticklabels([f"{w + 1:02d}" for w in range(24)], fontsize=8)
ax.grid(axis="x", visible=False)
ax.set_xlabel("week file (X01 .. X24); weeks are stored back to back here, real dates have gaps")
ax.set_ylabel("Gbit/s")
ax.set_title("Abilene, 24 weeks: steps removed by the missing-data policy", pad=16)
ax.legend(frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.22), fontsize=9)
ax.set_xlim(0, 24 * 7)
fig.tight_layout()
show(fig, "01_missing_data.png", "All 24 weeks. Removed by the missing-data policy: shaded bands (days long) "
     "and marked lines (short outages and the one impossible value).")

# %% 6. window, split chronologically, fit scaler on train only, save
report.section("6. Windows, split, scaling, save")
summary = prepare_abilene(WINDOW, HORIZON)
print(f"samples {summary['n_samples']}: train {summary['n_train']}, "
      f"val {summary['n_val']}, test {summary['n_test']}")
print("train ends at", step_label(summary["train_end_step"]),
      "| val ends at", step_label(summary["val_end_step"]))

# %% 7. exit check: contracted shapes, dtype, range, and scaling round-trip
report.section("7. Exit check")
X_train, y_train, X_val, y_val, X_test, y_test = load_splits()
for name, a in zip(("X_train", "y_train", "X_val", "y_val", "X_test", "y_test"),
                   (X_train, y_train, X_val, y_val, X_test, y_test)):
    print(f"{name:8s} {str(a.shape):18s} {a.dtype}  min {a.min():.3f}  max {a.max():.3f}")
    assert a.dtype == np.float32 and a.shape[-1] == N_FLOWS
    assert a.ndim == (3 if name[0] == "X" else 2) and (a.ndim == 2 or a.shape[1] == WINDOW)
assert X_train.min() >= 0 and X_train.max() <= 1 + 1e-6     # train is exactly [0, 1]

scaler = FlowScaler.load()
steps = np.load(PROCESSED / "target_steps.npy")
raw_targets = weeks.reshape(-1, N_FLOWS)[steps[-len(y_test):]]
assert np.allclose(scaler.inverse_transform(y_test), raw_targets, rtol=1e-4, atol=1)
print("inverse_transform(y_test) matches the raw data: OK")
print("Phase 1 exit condition met: contracted shapes, daily cycle in results/figures/01_daily_cycle.png")

# %% write the report page and open it in the browser
report.finish()
