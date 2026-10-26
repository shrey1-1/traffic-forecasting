"""Shared code for the CNN-LSTM network traffic forecasting project.

This is the only place these functions live. Scripts import it with
`from common import *` after locating it on disk (see the header cell in scripts/).

Data contract (every script depends on this):
    X: (samples, WINDOW, N_FLOWS) float32
    y: (samples, N_FLOWS)         float32
    scaling: log1p, then per-flow min-max to [0, 1], fitted on TRAIN only
    split:   chronological 60/20/20, never random
"""

import csv
import os
import re
import random
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT = Path(os.environ.get("TRAFFIC_PROJECT", Path(__file__).resolve().parent)).resolve()
RAW = PROJECT / "raw"
PROCESSED = PROJECT / "processed"
MODELS = PROJECT / "models"
RESULTS = PROJECT / "results"
FIGURES = RESULTS / "figures"
REPORTS = RESULTS / "reports"
METRICS_CSV = RESULTS / "metrics.csv"


def ensure_dirs():
    """Create the project folders if they are missing."""
    for d in (RAW, PROCESSED, MODELS, RESULTS, FIGURES, REPORTS):
        d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_FLOWS = 144               # Abilene: 12 nodes x 12 nodes OD pairs
WINDOW = 12                 # input length: 12 x 5 min = 1 hour
HORIZON = 1                 # steps ahead to predict (later 6 and 12)
STEPS_PER_WEEK = 2016       # 7 days x 288 five-minute intervals
COLS_PER_ROW = 720          # 144 flows x 5 alternative estimates
SPLIT_FRACTIONS = (0.6, 0.2, 0.2)
SEED = 42

# Abilene facts, from raw/readme.txt and raw/topo-2003-04-10.txt.
# Flow k is ABILENE_NODES[k // 12] -> ABILENE_NODES[k % 12] (same order as raw/demands).
ABILENE_NODES = ["ATLA-M5", "ATLAng", "CHINng", "DNVRng", "HSTNng", "IPLSng",
                 "KSCYng", "LOSAng", "NYCMng", "SNVAng", "STTLng", "WASHng"]
ABILENE_COLUMN = 0          # realOD; the other 4 of each 5 are gravity/tomogravity estimates
ABILENE_UNIT_BYTES = 100    # raw values are in units of 100 bytes per 5 minutes
ABILENE_WEEK_STARTS = [     # first day of X01 .. X24; the weeks are NOT all back to back
    "2004-03-01", "2004-03-08", "2004-04-02", "2004-04-09", "2004-04-22", "2004-05-01",
    "2004-05-08", "2004-05-15", "2004-05-22", "2004-05-29", "2004-06-05", "2004-06-12",
    "2004-06-19", "2004-06-26", "2004-07-03", "2004-07-10", "2004-07-17", "2004-07-24",
    "2004-07-31", "2004-08-07", "2004-08-13", "2004-08-21", "2004-08-28", "2004-09-04"]

SPLIT_NAMES = ("X_train", "y_train", "X_val", "y_val", "X_test", "y_test")
METRIC_COLUMNS = ["timestamp", "name", "dataset", "horizon",
                  "mae", "rmse", "mape", "mape_coverage", "notes"]


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seeds(seed=SEED):
    """Seed Python, NumPy and (if already imported) TensorFlow.

    TensorFlow is only seeded when it has been imported, so scripts that do
    not need it (01, 02) do not pay its import time. Scripts that train
    models must `import tensorflow` before calling set_seeds().
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if "tensorflow" in sys.modules:
        import tensorflow as tf
        tf.keras.utils.set_random_seed(seed)
        try:
            tf.config.experimental.enable_op_determinism()   # same result on every run
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Scaling
# ---------------------------------------------------------------------------
class FlowScaler:
    """log1p, then per-flow min-max to [0, 1].

    Works on any array whose last axis is the flow axis, e.g. a series
    (T, F), windows (samples, WINDOW, F) or targets (samples, F).
    Fit on the TRAINING split only. Flows that are constant in training get
    a range of 1 so they do not divide by zero.
    """

    def __init__(self):
        self.min_ = None
        self.max_ = None

    @staticmethod
    def _check(a):
        a = np.asarray(a, dtype=np.float64)
        if np.isnan(a).any():
            raise ValueError("NaN in input: apply the missing-data policy before scaling")
        if (a < 0).any():
            raise ValueError("negative traffic values: log1p scaling expects >= 0")
        return a

    def fit(self, a):
        logged = np.log1p(self._check(a)).reshape(-1, np.shape(a)[-1])
        self.min_ = logged.min(axis=0)
        self.max_ = logged.max(axis=0)
        return self

    @property
    def range_(self):
        r = self.max_ - self.min_
        return np.where(r > 0, r, 1.0)

    def _require_fit(self):
        if self.min_ is None:
            raise RuntimeError("FlowScaler is not fitted")

    def transform(self, a):
        self._require_fit()
        return ((np.log1p(self._check(a)) - self.min_) / self.range_).astype(np.float32)

    def fit_transform(self, a):
        return self.fit(a).transform(a)

    def inverse_transform(self, z):
        self._require_fit()
        z = np.asarray(z, dtype=np.float64)
        return np.expm1(z * self.range_ + self.min_).astype(np.float32)

    def save(self, path=None):
        self._require_fit()
        path = Path(path) if path else PROCESSED / "scaler.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, min_=self.min_, max_=self.max_)
        return path

    @classmethod
    def load(cls, path=None):
        path = Path(path) if path else PROCESSED / "scaler.npz"
        with np.load(path) as d:
            s = cls()
            s.min_, s.max_ = d["min_"], d["max_"]
        return s


# ---------------------------------------------------------------------------
# Windowing and splitting
# ---------------------------------------------------------------------------
def make_windows(series, window=WINDOW, horizon=HORIZON):
    """Turn a (T, F) series into supervised samples.

    X[i] = series[i : i + window]
    y[i] = series[i + window + horizon - 1]   (the value `horizon` steps
                                               after the last input step)
    Returns X (samples, window, F) and y (samples, F), both float32.
    """
    series = np.asarray(series, dtype=np.float32)
    if series.ndim != 2:
        raise ValueError(f"series must be 2-D (T, F), got shape {series.shape}")
    n = len(series) - window - horizon + 1
    if n <= 0:
        raise ValueError("series too short for this window and horizon")
    idx = np.arange(window)[None, :] + np.arange(n)[:, None]
    X = series[idx]
    y = series[window + horizon - 1: window + horizon - 1 + n]
    return X, y


def chronological_split(X, y, fractions=SPLIT_FRACTIONS):
    """Split samples in time order (never shuffled) into train/val/test."""
    if abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError("split fractions must sum to 1")
    n = len(X)
    n_train = int(n * fractions[0])
    n_val = int(n * fractions[1])
    a, b = n_train, n_train + n_val
    return X[:a], y[:a], X[a:b], y[a:b], X[b:], y[b:]


def save_splits(X_train, y_train, X_val, y_val, X_test, y_test, directory=None):
    """Save the six split arrays as .npy files in processed/."""
    directory = Path(directory) if directory else PROCESSED
    directory.mkdir(parents=True, exist_ok=True)
    arrays = (X_train, y_train, X_val, y_val, X_test, y_test)
    for name, arr in zip(SPLIT_NAMES, arrays):
        np.save(directory / f"{name}.npy", np.asarray(arr, dtype=np.float32))
    return directory


def load_splits(directory=None):
    """Return X_train, y_train, X_val, y_val, X_test, y_test from processed/."""
    directory = Path(directory) if directory else PROCESSED
    missing = [n for n in SPLIT_NAMES if not (directory / f"{n}.npy").exists()]
    if missing:
        raise FileNotFoundError(f"missing {missing} in {directory}; run scripts/01_data.py first")
    return tuple(np.load(directory / f"{n}.npy") for n in SPLIT_NAMES)


# ---------------------------------------------------------------------------
# Evaluation and logging
# ---------------------------------------------------------------------------
def evaluate(y_true, y_pred, mape_floor=None):
    """Metrics in RAW units (invert scaling before calling this).

    MAPE is only computed where y_true >= mape_floor, because many flows are
    near zero and would make MAPE explode. mape_coverage is the fraction of
    entries included, so always report it next to MAPE.
    Default floor: 10% of the mean of y_true.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch {y_true.shape} vs {y_pred.shape}")
    if np.isnan(y_true).any() or np.isnan(y_pred).any():
        raise ValueError("NaN in y_true or y_pred")
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    floor = 0.1 * float(np.mean(y_true)) if mape_floor is None else float(mape_floor)
    mask = y_true >= floor
    if mask.any():
        mape = float(np.mean(np.abs(err[mask]) / y_true[mask]) * 100)
    else:
        mape = float("nan")
    return {"mae": mae, "rmse": rmse, "mape": mape,
            "mape_coverage": float(mask.mean())}


def evaluate_scaled(y_true_scaled, y_pred_scaled, scaler, mape_floor=None):
    """evaluate() for arrays in scaled units: inverts both with `scaler` first."""
    return evaluate(scaler.inverse_transform(y_true_scaled),
                    scaler.inverse_transform(y_pred_scaled), mape_floor)


def persistence(X):
    """Persistence baseline: predict that the next value equals the last input value."""
    return X[:, -1, :]


def log_result(name, metrics, horizon=HORIZON, dataset="abilene", notes="", path=None):
    """Append one run to results/metrics.csv."""
    path = Path(path) if path else METRICS_CSV
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    row = {"timestamp": datetime.now().isoformat(timespec="seconds"),
           "name": name, "dataset": dataset, "horizon": horizon,
           "mae": metrics["mae"], "rmse": metrics["rmse"],
           "mape": metrics["mape"], "mape_coverage": metrics["mape_coverage"],
           "notes": notes}
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=METRIC_COLUMNS)
        if new:
            w.writeheader()
        w.writerow(row)
    return row


def show_results(path=None):
    """Return metrics.csv as a DataFrame (empty if nothing logged yet)."""
    import pandas as pd
    path = Path(path) if path else METRICS_CSV
    if not path.exists():
        return pd.DataFrame(columns=METRIC_COLUMNS)
    return pd.read_csv(path)


def status():
    """Print where the project is and which pipeline outputs exist."""
    print(f"project    : {PROJECT}")
    print(f"python     : {sys.version.split()[0]}   numpy {np.__version__}")
    raw_files = sorted(RAW.glob("X*.gz")) if RAW.exists() else []
    print(f"raw        : {len(raw_files)}/24 Abilene week files")
    have_splits = all((PROCESSED / f"{n}.npy").exists() for n in SPLIT_NAMES)
    print(f"processed  : splits {'yes' if have_splits else 'no'}, "
          f"scaler {'yes' if (PROCESSED / 'scaler.npz').exists() else 'no'}")
    n_models = len([p for p in MODELS.glob("*") if not p.name.startswith(".")]) if MODELS.exists() else 0
    print(f"models     : {n_models} file(s)")
    n_runs = 0
    if METRICS_CSV.exists():
        with open(METRICS_CSV, encoding="utf-8") as f:
            n_runs = max(sum(1 for _ in f) - 1, 0)
    print(f"metrics    : {n_runs} logged run(s)")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_abilene_week(week, column=ABILENE_COLUMN):
    """Read raw/Xnn.gz and return one column of it as (2016, 144) float64, raw units."""
    a = np.loadtxt(RAW / f"X{week:02d}.gz")
    if a.shape != (STEPS_PER_WEEK, COLS_PER_ROW):
        raise ValueError(f"X{week:02d}.gz has shape {a.shape}, expected "
                         f"({STEPS_PER_WEEK}, {COLS_PER_ROW})")
    return a.reshape(STEPS_PER_WEEK, N_FLOWS, 5)[:, :, column]


def load_abilene_raw(column=ABILENE_COLUMN, use_cache=True):
    """All 24 weeks as (24, 2016, 144) float64, in raw units (100 bytes / 5 min).

    Weeks are kept separate because they are not all back to back (see
    abilene_segments). Parsing takes a while, so the result is cached in
    processed/abilene_col{column}.npy; delete it to force a re-parse.
    """
    cache = PROCESSED / f"abilene_col{column}.npy"
    if use_cache and cache.exists():
        return np.load(cache)
    missing = [w for w in range(1, 25) if not (RAW / f"X{w:02d}.gz").exists()]
    if missing:
        raise FileNotFoundError(f"missing raw/X??.gz for weeks {missing}")
    weeks = np.stack([load_abilene_week(w, column) for w in range(1, 25)])
    np.save(cache, weeks)
    return weeks


def abilene_segments():
    """Group week numbers (1..24) into runs of back-to-back weeks.

    Two weeks are back to back when the second starts exactly 7 days after
    the first (dates from raw/readme.txt). Windows must never span the gap
    between runs, because the time steps on either side are not adjacent.
    """
    starts = [datetime.fromisoformat(d) for d in ABILENE_WEEK_STARTS]
    runs = [[1]]
    for w in range(2, 25):
        if (starts[w - 1] - starts[w - 2]).days == 7:
            runs[-1].append(w)
        else:
            runs.append([w])
    return runs


def make_windows_segments(segments, window=WINDOW, horizon=HORIZON):
    """make_windows on each (T_i, F) segment separately, concatenated in order.

    No window crosses from one segment into the next. Segments too short for
    a single window are skipped.
    """
    Xs, ys = [], []
    for seg in segments:
        if len(seg) >= window + horizon:
            X, y = make_windows(seg, window, horizon)
            Xs.append(X)
            ys.append(y)
    return np.concatenate(Xs), np.concatenate(ys)


# Missing-data policy thresholds (see docs/PROJECT_LOG.md, Phase 1)
OUTAGE_FLOW_STEPS = 288      # R2: an active flow at zero for >= 1 day is an outage
OUTAGE_ROUTER_STEPS = 12     # R3: all traffic in/out of a router at zero for >= 1 hour
ACTIVE_FLOW_MAX_ZERO = 0.5   # R2 only applies to flows that are zero < 50% of the time
MAX_LINK_KBPS = 9_920_000    # R4: fastest Abilene link (raw/topo-2003-04-10.txt)
MAX_FLOW_RAW = MAX_LINK_KBPS * 1e3 / 8 * 300 / ABILENE_UNIT_BYTES   # same, in raw units per step


def _true_runs(b):
    """(start, end) of each run of True in a 1-D boolean array, end exclusive."""
    b = np.concatenate([[False], np.asarray(b, bool), [False]])
    d = np.diff(b.astype(np.int8))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def abilene_clean(weeks):
    """Apply the missing-data policy to (24, 2016, 144) raw weeks.

    R1 duplicate: a week whose first day is an exact copy of the previous
       week's last day; the copy is removed.
    R2 flow outage: a flow that is normally active (zero < 50% of the time)
       reads zero for >= 1 day; those time steps are removed for all flows.
    R3 router outage: all traffic into, or out of, one router is zero for
       >= 1 hour; those time steps are removed.
    R4 impossible value: a flow above the fastest link's capacity for the
       5 minutes (9.92 Gbit/s); that time step is removed.
    All other zeros are kept: small flows really do record zero under
    1-in-100 packet sampling. Large but possible spikes are kept too.

    The kept steps are then cut into segments wherever a step was removed or
    two weeks are not back to back (README dates not 7 days apart, or R1 fired).

    Returns (segments, info). segments is a list of (start, array) where start
    is the index of the segment's first step in the 24 x 2016 timeline and
    array is (T_i, 144). info holds the keep mask, the reason per removed step
    and counts, for reporting.
    """
    n_w, T, F = weeks.shape
    full = weeks.reshape(-1, F)
    reason = np.full(len(full), "", dtype="<U10")

    dup_weeks = []
    for w in range(1, n_w):
        if np.array_equal(weeks[w, :288], weeks[w - 1, -288:]):
            reason[w * T: w * T + 288] = "duplicate"
            dup_weeks.append(w + 1)

    zero = full == 0
    active = zero.mean(axis=0) < ACTIVE_FLOW_MAX_ZERO
    flow_outages = []
    for k in np.where(active)[0]:
        for a, b in _true_runs(zero[:, k]):
            if b - a >= OUTAGE_FLOW_STEPS:
                reason[a:b] = np.where(reason[a:b] == "", "flow", reason[a:b])
                flow_outages.append((int(k), int(a), int(b)))

    tm = full.reshape(-1, 12, 12)
    router_outages = []
    for r in range(12):
        for direction, tot in (("into", tm[:, :, r].sum(1)), ("out of", tm[:, r, :].sum(1))):
            for a, b in _true_runs(tot == 0):
                if b - a >= OUTAGE_ROUTER_STEPS:
                    reason[a:b] = np.where(reason[a:b] == "", "router", reason[a:b])
                    router_outages.append((ABILENE_NODES[r], direction, int(a), int(b)))

    impossible = np.where((full > MAX_FLOW_RAW).any(axis=1))[0]
    reason[impossible] = np.where(reason[impossible] == "", "impossible", reason[impossible])

    keep = reason == ""
    starts = [datetime.fromisoformat(d) for d in ABILENE_WEEK_STARTS]
    new_segment = np.zeros(len(full), bool)
    for w in range(1, n_w):
        joined = (starts[w] - starts[w - 1]).days == 7 and (w + 1) not in dup_weeks
        if not joined:
            new_segment[w * T] = True

    segments = []
    for a, b in _true_runs(keep):
        cuts = [a] + [int(i) for i in np.where(new_segment[a + 1:b])[0] + a + 1] + [b]
        for s, e in zip(cuts[:-1], cuts[1:]):
            segments.append((s, full[s:e]))

    info = {"keep": keep, "reason": reason, "duplicate_weeks": dup_weeks,
            "flow_outages": flow_outages, "router_outages": router_outages,
            "impossible_steps": [int(i) for i in impossible],
            "removed": {r: int((reason == r).sum())
                        for r in ("duplicate", "flow", "router", "impossible")}}
    return segments, info


def prepare_abilene(window=WINDOW, horizon=HORIZON, directory=None):
    """Phase 1 pipeline: raw -> clean -> windows -> chronological split -> scale -> save.

    The scaler is fitted on the training inputs only. Saves the six splits,
    scaler.npz and target_steps.npy (timeline index of every sample's target,
    in sample order) to `directory` (default processed/). Returns a summary dict.
    """
    segments, info = abilene_clean(load_abilene_raw())
    return _prepare_segments(segments, info, window, horizon, directory)


def _prepare_segments(segments, info, window, horizon, directory):
    """Windows inside each (start, array) segment -> split -> scale -> save (see prepare_abilene)."""
    directory = Path(directory) if directory else PROCESSED

    Xs, ys, idx = [], [], []
    for start, seg in segments:
        if len(seg) >= window + horizon:
            X, y = make_windows(seg, window, horizon)
            Xs.append(X)
            ys.append(y)
            idx.append(start + window + horizon - 1 + np.arange(len(X)))
    X, y, idx = np.concatenate(Xs), np.concatenate(ys), np.concatenate(idx)

    X_tr, y_tr, X_va, y_va, X_te, y_te = chronological_split(X, y)
    scaler = FlowScaler().fit(X_tr)
    save_splits(scaler.transform(X_tr), scaler.transform(y_tr),
                scaler.transform(X_va), scaler.transform(y_va),
                scaler.transform(X_te), scaler.transform(y_te), directory=directory)
    scaler.save(directory / "scaler.npz")
    np.save(directory / "target_steps.npy", idx)

    n_tr, n_va = len(X_tr), len(X_va)
    return {"segments": [(s, len(seg)) for s, seg in segments], "info": info,
            "n_samples": len(X), "n_train": n_tr, "n_val": n_va, "n_test": len(X_te),
            "train_end_step": int(idx[n_tr - 1]), "val_end_step": int(idx[n_tr + n_va - 1]),
            "target_steps": idx}


# ---------------------------------------------------------------------------
# GÉANT (Phase 6): 23 nodes, 15-minute matrices, 1 Jan to 29 Apr 2005
# ---------------------------------------------------------------------------
GEANT_DIR = PROJECT / "raw_geant"
GEANT_NODES = 23
GEANT_FLOWS = GEANT_NODES ** 2          # 529 origin-destination flows
GEANT_STEP_S = 900                      # 15 minutes
GEANT_START = (2005, 1, 1)


def load_geant_raw(cache=True):
    """All GÉANT traffic matrices as one (T, 529) float32 array in kbit/s; NaN = no entry.

    Reads raw_geant/totem/traffic-matrices/IntraTM-*.xml (TOTEM dataset, anonymised node ids
    1-23; flow = source index * 23 + destination index, ids sorted ascending). The row of a
    matrix is its time slot counted in 15-minute steps from 2005-01-01 00:00; slots with no
    file, and pairs a file does not list, stay NaN. Cached in processed/geant_raw.npy.
    """
    import datetime as dt
    import re
    path = PROCESSED / "geant_raw.npy"
    if cache and path.exists():
        return np.load(path)
    files = sorted((GEANT_DIR / "totem" / "traffic-matrices").glob("IntraTM-*.xml"))
    if not files:
        raise FileNotFoundError(f"no GÉANT matrices in {GEANT_DIR}/totem/traffic-matrices")
    t0 = dt.datetime(*GEANT_START)
    src_re = re.compile(r'<src id="(\d+)">(.*?)</src>', re.S)
    dst_re = re.compile(r'<dst id="(\d+)">([^<]*)</dst>')
    rows = {}
    for f in files:
        s = f.read_text()
        ts = dt.datetime.fromisoformat(re.search(r"<date>([^<]+)</date>", s).group(1))
        v = np.full((GEANT_NODES, GEANT_NODES), np.nan)
        for a, b in src_re.findall(s):
            for c, x in dst_re.findall(b):
                v[int(a) - 1, int(c) - 1] = float(x)
        rows[int((ts - t0).total_seconds() // GEANT_STEP_S)] = v.ravel()
    arr = np.full((max(rows) + 1, GEANT_FLOWS), np.nan, np.float32)
    for k, v in rows.items():
        arr[k] = v
    if cache:
        PROCESSED.mkdir(parents=True, exist_ok=True)
        np.save(path, arr)
    return arr


# GÉANT cleaning thresholds (see docs/PROJECT_LOG.md, section 17)
GEANT_MAX_FLOW_KBPS = 1e7        # G3: a flow above the 10 Gbit/s core link speed is impossible
GEANT_MAX_ABSENT = 0.5           # G2: more than half the usual entries missing = collector failure
GEANT_MIN_TOTAL = 0.2            # G4: network total below 20% of the median = outage


def geant_clean(raw):
    """GÉANT missing-data policy. Returns (segments, info) like abilene_clean.

    Steps are dropped, in this order, when: G1 there is no file; G2 more than
    GEANT_MAX_ABSENT of the flows that ever appear are absent; G3 any flow is
    above GEANT_MAX_FLOW_KBPS; G4 the network total is below GEANT_MIN_TOTAL of
    the median total. Pairs a kept file does not list are set to 0: they are
    mostly small flows and no listed value is ever exactly 0, so absence means
    traffic below the measurement floor. Segments are runs of consecutive kept steps.
    """
    raw = np.asarray(raw, dtype=np.float64)
    reason = np.full(len(raw), "", dtype=object)
    empty = np.isnan(raw).all(1)
    reason[empty] = "no_file"
    live = ~np.isnan(raw[~empty]).all(0)
    absent = np.isnan(raw[:, live]).mean(1)
    reason[(reason == "") & (absent > GEANT_MAX_ABSENT)] = "incomplete"
    x = np.nan_to_num(raw, nan=0.0)
    reason[(reason == "") & (x.max(1) > GEANT_MAX_FLOW_KBPS)] = "impossible"
    total = x.sum(1)
    median = np.median(total[reason == ""])
    reason[(reason == "") & (total < GEANT_MIN_TOTAL * median)] = "outage"

    keep = reason == ""
    segments, start = [], None
    for t in range(len(raw) + 1):
        if t < len(raw) and keep[t]:
            start = t if start is None else start
        elif start is not None:
            segments.append((start, x[start:t].astype(np.float32)))
            start = None
    removed = {r: int((reason == r).sum()) for r in ("no_file", "incomplete", "impossible", "outage")}
    return segments, {"reason": reason, "kept": int(keep.sum()), "total": len(raw), "removed": removed}


def prepare_geant(window=WINDOW, horizon=HORIZON, directory=None):
    """Phase 6 pipeline for GÉANT: raw -> clean -> windows -> split -> scale -> save.

    Same as prepare_abilene; the default directory is processed/geant/h<horizon>.
    """
    directory = Path(directory) if directory else PROCESSED / "geant" / f"h{horizon}"
    segments, info = geant_clean(load_geant_raw())
    return _prepare_segments(segments, info, window, horizon, directory)


def load_geant_routing():
    """Approximate GÉANT routing: minimum-hop paths with equal-cost splitting.

    The anonymised TOTEM topology has no IGP weights and no capacities, so the real
    routing is unknown. Every flow is routed over its shortest paths by hop count and
    split equally among equal-length paths. Returns a dict: A (n_links, 529) with
    link_load = A @ flows, links (list of "src_dst" ids), pairs (src, dst) indices.
    """
    t = (GEANT_DIR / "totem" / "topology-anonymised.xml").read_text()
    t = t[t.index("<topology>"):t.index("</topology>")]
    edges = [(int(a), int(b)) for a, b in
             re.findall(r'<link id="[^"]*">\s*<from [^>]*node="(\d+)"/>\s*<to [^>]*node="(\d+)"/>', t)]
    n = GEANT_NODES
    adj = np.zeros((n, n), bool)
    for a, b in edges:
        adj[a - 1, b - 1] = True
    INF = 10 ** 6
    dist = np.where(adj, 1, INF).astype(np.int64)
    np.fill_diagonal(dist, 0)
    for k in range(n):                                   # Floyd-Warshall, hop counts
        dist = np.minimum(dist, dist[:, [k]] + dist[[k], :])
    paths = np.zeros((n, n))                             # number of shortest paths s -> t
    for s_ in range(n):
        order = np.argsort(dist[s_])
        paths[s_, s_] = 1
        for v in order:
            if v == s_ or dist[s_, v] >= INF:
                continue
            paths[s_, v] = sum(paths[s_, u] for u in range(n) if adj[u, v] and dist[s_, u] + 1 == dist[s_, v])
    links = sorted(edges)
    A = np.zeros((len(links), GEANT_FLOWS))
    for s_ in range(n):
        for t_ in range(n):
            if s_ == t_ or dist[s_, t_] >= INF:
                continue
            for li, (a, b) in enumerate(links):
                u, v = a - 1, b - 1
                if dist[s_, u] + 1 + dist[v, t_] == dist[s_, t_]:
                    A[li, s_ * n + t_] = paths[s_, u] * paths[v, t_] / paths[s_, t_]
    return {"A": A, "links": [f"{a}_{b}" for a, b in links], "n_edges": len(edges),
            "hops": dist}


def warn_prf(true_ev, pred_ev):
    """Precision, recall, F1 and counts (tp, fp, fn) of boolean event arrays of equal shape."""
    tp = int((true_ev & pred_ev).sum()); fp = int((~true_ev & pred_ev).sum()); fn = int((true_ev & ~pred_ev).sum())
    p = tp / (tp + fp) if tp + fp else float("nan")
    r = tp / (tp + fn) if tp + fn else float("nan")
    f = 2 * p * r / (p + r) if tp else (0.0 if tp + fp + fn else float("nan"))
    return p, r, f, tp, fp, fn


def warn_onsets(true_ev, steps):
    """Boolean (rows, links): congested now and not on the previous step. `steps` is the timeline
    index of every row; a row that does not directly follow the one above (a data gap) is never
    an onset, because we cannot see what came before it."""
    consecutive = np.r_[False, np.diff(steps) == 1]
    prev = np.vstack([np.zeros((1, true_ev.shape[1]), bool), true_ev[:-1]])
    return true_ev & ~prev & consecutive[:, None]


def step_label(step):
    """'X07 day 3 14:35' for a step index in the 24 x 2016 timeline.

    The README gives only dates, so each week file is assumed to start at 00:00.
    """
    w, s = divmod(int(step), STEPS_PER_WEEK)
    d, m = divmod(s, 288)
    return f"X{w + 1:02d} day {d + 1} {m * 5 // 60:02d}:{m * 5 % 60:02d}"


def load_abilene_routing():
    """Routing matrix and link facts from raw/A, raw/links and raw/topo-*.txt.

    Returns a dict:
      A          (54, 144) 0/1 matrix, link_loads = A @ traffic_matrix
      links      54 names like "ATLAng,HSTNng" (rows of A)
      link_type  54 ints: 0 internal backbone link, 1 inbound, 2 outbound
      capacity_kbps  54 floats, capacity of internal links (NaN for the others)
    """
    A = np.zeros((54, N_FLOWS))
    with open(RAW / "A") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            _, _, li, di, frac = line.split()
            A[int(li) - 1, int(di) - 1] = float(frac)

    links, link_type = [], []
    with open(RAW / "links") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            name, _, t = line.split()
            links.append(name)
            link_type.append(int(t))

    cap = {}
    topo = next(RAW.glob("topo-*.txt"))
    in_links = False
    for line in topo.read_text().splitlines():
        if line.strip() == "link":
            in_links = True
        elif in_links and line.strip() and not line.startswith("#"):
            x, y, c, _ = line.split()
            cap[f"{x},{y}"] = float(c)
    capacity = np.array([cap.get(n, np.nan) for n in links])
    return {"A": A, "links": links, "link_type": np.array(link_type),
            "capacity_kbps": capacity}


# ---------------------------------------------------------------------------
# Phase 8: real-time lookup, extra baselines, significance tests
# ---------------------------------------------------------------------------
STEPS_PER_DAY = {"abilene": 288, "geant": 96}


def clean_timeline(dataset):
    """The cleaned traffic on a real-time axis, for baselines that look back further than the window.

    Abilene week files are not all back to back, so the timeline index used in
    target_steps.npy (week * 2016 + step) is not real time. Here row r of `values`
    is real step r: days since the first week start * 288 + step within the day
    (GÉANT's slots are already real time). Steps removed by the cleaning policy and
    steps with no data are known=False.

    Returns (values (N, F) float64 raw units, known (N,) bool, to_time) where
    to_time[timeline_index] is the row of `values` for that timeline index.
    """
    if dataset == "abilene":
        weeks = load_abilene_raw()
        _, info = abilene_clean(weeks)
        full, keep = weeks.reshape(-1, N_FLOWS), info["keep"]
        d0 = datetime.fromisoformat(ABILENE_WEEK_STARTS[0])
        offs = np.array([(datetime.fromisoformat(d) - d0).days * 288 for d in ABILENE_WEEK_STARTS])
        i = np.arange(len(full))
        to_time = offs[i // STEPS_PER_WEEK] + i % STEPS_PER_WEEK
    elif dataset == "geant":
        raw = load_geant_raw()
        _, info = geant_clean(raw)
        full, keep = np.nan_to_num(raw.astype(np.float64)), info["reason"] == ""
        to_time = np.arange(len(full))
    else:
        raise ValueError(f"unknown dataset {dataset!r}")
    rows = to_time[keep]
    if len(np.unique(rows)) != len(rows):
        raise ValueError("two kept steps fall on the same real time")
    values = np.zeros((to_time.max() + 1, full.shape[1]))
    known = np.zeros(to_time.max() + 1, bool)
    values[rows], known[rows] = full[keep], True
    return values, known, to_time


def lookback(values, known, times, lag):
    """values[t - lag] for every real time t in `times`, NaN where that step is unknown."""
    src = np.asarray(times) - lag
    ok = (src >= 0) & known[np.clip(src, 0, None)]
    out = np.full((len(src), values.shape[1]), np.nan)
    out[ok] = values[src[ok]]
    return out


def seasonal_mean(values, known, times, period, n=7):
    """Mean of the values 1..n periods earlier (e.g. the same time on each of the last 7 days),
    over the ones that are known; NaN where none is known."""
    stack = np.stack([lookback(values, known, times, k * period) for k in range(1, n + 1)])
    with np.errstate(invalid="ignore"):
        cnt = (~np.isnan(stack)).sum(0)
        return np.where(cnt > 0, np.nansum(stack, 0) / np.maximum(cnt, 1), np.nan)


def ridge_path(X, Y, alphas):
    """Ridge regression of Y (n, k) on X (n, p) for several penalties at once.

    Columns of X are standardised (with the training mean and std) and Y is centred,
    so the penalty is the same for every input and there is an unpenalised intercept.
    The penalty is alpha * n. Returns a list of (W, b), one per alpha, such that
    the forecast for new inputs Xn is Xn @ W + b.
    """
    X = np.asarray(X, np.float64)
    Y = np.asarray(Y, np.float64)
    mu, sd = X.mean(0), X.std(0)
    sd = np.where(sd > 0, sd, 1.0)
    Xs = (X - mu) / sd
    ym = Y.mean(0)
    G = Xs.T @ Xs
    XtY = Xs.T @ (Y - ym)
    lam, V = np.linalg.eigh(G)
    VtXtY = V.T @ XtY
    out = []
    for a in alphas:
        Ws = V @ (VtXtY / (lam + a * len(X))[:, None])
        W = Ws / sd[:, None]
        out.append((W, ym - mu @ W))
    return out


def dm_test(loss_a, loss_b, lag):
    """Diebold-Mariano test of equal expected loss for two forecasts of the same targets.

    loss_a, loss_b: (n,) loss per time step (e.g. absolute error averaged over flows).
    d = loss_a - loss_b; its variance uses a Newey-West (Bartlett) estimator with `lag`
    autocovariances, because forecast errors on consecutive steps are correlated.
    Returns (mean d, DM statistic, two-sided p-value from the normal distribution).
    Negative mean d: A has the lower loss.
    """
    from math import erfc, sqrt
    d = np.asarray(loss_a, np.float64) - np.asarray(loss_b, np.float64)
    n = len(d)
    e = d - d.mean()
    var = e @ e / n
    for k in range(1, min(lag, n - 1) + 1):
        var += 2 * (1 - k / (lag + 1)) * (e[k:] @ e[:-k]) / n
    stat = d.mean() / sqrt(max(var, 1e-300) / n)
    return float(d.mean()), float(stat), float(erfc(abs(stat) / sqrt(2)))


def block_bootstrap_gain(err_a, err_b, block, n_boot=2000, seed=SEED):
    """95% interval for the MAE change of A against B, in %, positive = A better.

    err_a, err_b: (n,) absolute error per time step (summed or averaged over flows).
    Moving-block bootstrap over time: blocks of `block` consecutive steps are drawn
    with replacement, so the correlation between nearby steps is kept.
    Returns (point estimate, low, high).
    """
    a, b = np.asarray(err_a, np.float64), np.asarray(err_b, np.float64)
    n = len(a)
    rng = np.random.default_rng(seed)
    k = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(n_boot, k))
    idx = (starts[:, :, None] + np.arange(block)).reshape(n_boot, -1)[:, :n]
    g = (1 - a[idx].sum(1) / b[idx].sum(1)) * 100
    lo, hi = np.percentile(g, [2.5, 97.5])
    return float((1 - a.sum() / b.sum()) * 100), float(lo), float(hi)


def saved_model_names(dataset, horizon, key, seeds=(42, 43, 44)):
    """File names (without .keras) of the Phase 4-6 models used in the report tables.

    dataset "abilene" or "geant"; key "cnn_lstm", "lstm" or "cnn". Abilene horizon 1
    CNN-LSTM and LSTM-only come from Phase 4 (same settings as Phase 5).
    """
    if dataset == "geant":
        return [f"g_h{horizon}_{key}_s{s}" for s in seeds]
    if horizon == 1 and key == "cnn_lstm":
        return [f"p4_cnn_lstm_f64_k3_u100-50_d20_lr0.001_s{s}" for s in seeds]
    if horizon == 1 and key == "lstm":
        return [f"p4_lstm_u100-50_d20_lr0.001_s{s}" for s in seeds]
    return [f"p5_h{horizon}_{key}_s{s}" for s in seeds]


def data_dir(dataset, horizon):
    """processed/ folder holding the splits for this dataset and horizon (window 12)."""
    if dataset == "geant":
        return PROCESSED / "geant" / f"h{horizon}"
    return PROCESSED if horizon == 1 else PROCESSED / f"h{horizon}"


# ---------------------------------------------------------------------------
# Phase 9: extra inputs (time of day, seasonal context, longer windows)
# ---------------------------------------------------------------------------
FIRST_WEEKDAY = {"abilene": datetime.fromisoformat(ABILENE_WEEK_STARTS[0]).weekday(),   # Monday = 0
                 "geant": datetime(*GEANT_START).weekday()}


def time_features(dataset, times):
    """(n, 4) float32: sin and cos of the time of day and of the day of the week, for real times
    (rows of clean_timeline). Abilene week files are assumed to start at 00:00 (as step_label);
    GÉANT slots count from 1 Jan 2005 00:00 in the files' clock (time zone not documented)."""
    day = STEPS_PER_DAY[dataset]
    t = np.asarray(times)
    tod = 2 * np.pi * (t % day) / day
    dow = 2 * np.pi * ((t // day + FIRST_WEEKDAY[dataset]) % 7 + (t % day) / day) / 7
    return np.stack([np.sin(tod), np.cos(tod), np.sin(dow), np.cos(dow)], 1).astype(np.float32)


def segment_ids(dataset, to_time, n_rows):
    """Segment number of every real-time row (-1 = not in a kept segment). Windows must stay
    inside one segment, exactly as in the original data preparation."""
    segs = (abilene_clean(load_abilene_raw())[0] if dataset == "abilene"
            else geant_clean(load_geant_raw())[0])
    sid = np.full(n_rows, -1)
    for i, (start, seg) in enumerate(segs):
        sid[to_time[start:start + len(seg)]] = i
    return sid


def context_dataset(dataset, horizon, window=WINDOW, seasonal=None, use_time=False):
    """The Phase 4-6 samples (same targets, same split, same scaler), with optional extras.

    window   input length; the targets are those of the window-12 data, and a target is
             kept only if the longer window also fits inside its segment. `kept` lists
             which window-12 samples remain, per split, so every method can be scored
             on the same targets.
    seasonal None, "seasonal_day" or "seasonal_mean7": adds the seasonal value of every flow
             at the target time and at the last input time, each minus the last input
             value (scaled units; 0 where the seasonal value is missing).
    use_time adds the time-of-day / day-of-week features of the target time.

    Returns a dict, each entry a (train, val, test) list: X (n, window, F); C time features
    (n, 4) or None; S seasonal (n, 2, F) or None; y (scaled); last_raw / y_raw (float64
    original units); kept (index arrays); times; plus the scaler. Model input: model_inputs().
    """
    d = data_dir(dataset, horizon)
    sc = FlowScaler.load(d / "scaler.npz")
    splits = load_splits(d)
    steps = np.load(d / "target_steps.npy")
    n_tr, n_va = len(splits[1]), len(splits[3])
    bounds = [(0, n_tr), (n_tr, n_tr + n_va), (n_tr + n_va, len(steps))]
    values, known, to_time = clean_timeline(dataset)
    day = STEPS_PER_DAY[dataset]
    sid = segment_ids(dataset, to_time, len(values)) if window != WINDOW else None
    out = {"X": [], "C": [], "S": [], "y": [], "y_raw": [], "last_raw": [], "kept": [], "times": [],
           "scaler": sc}
    for i, (a, b) in enumerate(bounds):
        X12, y = splits[2 * i], splits[2 * i + 1]
        t = to_time[steps[a:b]]
        if window == WINDOW:
            keep = np.arange(b - a)
            X = X12
        else:
            first = t - horizon - window + 1
            ok = (first >= 0) & (sid[np.clip(first, 0, None)] == sid[t])
            keep = np.where(ok)[0]
            rows = first[keep][:, None] + np.arange(window)
            X = np.empty((len(rows), window, values.shape[1]), np.float32)
            for c in range(0, len(rows), 2048):          # in chunks: the float64 copy is large
                X[c:c + 2048] = sc.transform(values[rows[c:c + 2048]])
        t, y = t[keep], y[keep]
        last_raw = sc.inverse_transform(X[:, -1]).astype(np.float64)
        S = None
        if seasonal:
            parts = []
            for when in (t, t - horizon):
                s = (lookback(values, known, when, day) if seasonal == "seasonal_day"
                     else seasonal_mean(values, known, when, day, 7))
                # as a difference from the last input value (scaled), so it is centred near
                # zero; 0 where missing
                parts.append(sc.transform(np.where(np.isnan(s), last_raw, s)) - X[:, -1])
            S = np.stack(parts, 1).astype(np.float32)
        out["X"].append(X)
        out["C"].append(time_features(dataset, t) if use_time else None)
        out["S"].append(S)
        out["y"].append(y)
        out["y_raw"].append(sc.inverse_transform(y).astype(np.float64))
        out["last_raw"].append(last_raw)
        out["kept"].append(keep)
        out["times"].append(t)
    return out


def fold_dataset(dataset, horizon, split, seasonal=None):
    """Like context_dataset (window 12), but with a different chronological split (Phase 12).

    split = (a, b, c) fractions of the time-ordered samples: train [0, a), val [a, b), test [b, c).
    The samples are exactly the Phase 1 / Phase 6 ones (same targets, windows never cross a gap);
    the inputs are rebuilt from the cleaned timeline in original units, and a NEW scaler is fitted on
    this fold's training inputs only. Seasonal inputs, if asked for, are built as in context_dataset.
    """
    steps = np.load(data_dir(dataset, horizon) / "target_steps.npy")
    values, known, to_time = clean_timeline(dataset)
    day = STEPS_PER_DAY[dataset]
    t_all = to_time[steps]
    n = len(t_all)
    a, b, c = (int(f * n) for f in split)
    bounds = [(0, a), (a, b), (b, c)]
    first = t_all - horizon - WINDOW + 1
    if not known[(first[:, None] + np.arange(WINDOW)).ravel()].all():
        raise ValueError("a window touches an unknown step")
    tr_rows = first[:a][:, None] + np.arange(WINDOW)
    sc = FlowScaler().fit(values[np.unique(tr_rows)])         # every step that appears in a training input
    out = {"X": [], "C": [], "S": [], "y": [], "y_raw": [], "last_raw": [], "kept": [], "times": [],
           "scaler": sc, "split": split}
    for lo, hi in bounds:
        t = t_all[lo:hi]
        rows = first[lo:hi][:, None] + np.arange(WINDOW)
        X = np.empty((len(rows), WINDOW, values.shape[1]), np.float32)
        for k in range(0, len(rows), 2048):
            X[k:k + 2048] = sc.transform(values[rows[k:k + 2048]])
        last_raw = values[t - horizon].astype(np.float64)
        S = None
        if seasonal:
            parts = []
            for when in (t, t - horizon):
                s = (lookback(values, known, when, day) if seasonal == "seasonal_day"
                     else seasonal_mean(values, known, when, day, 7))
                parts.append(sc.transform(np.where(np.isnan(s), last_raw, s)) - X[:, -1])
            S = np.stack(parts, 1).astype(np.float32)
        out["X"].append(X)
        out["C"].append(None)
        out["S"].append(S)
        out["y"].append(sc.transform(values[t]))
        out["y_raw"].append(values[t].astype(np.float64))
        out["last_raw"].append(last_raw)
        out["kept"].append(np.arange(lo, hi))
        out["times"].append(t)
    return out


def model_inputs(d, i, rows=slice(None)):
    """What a Phase 9 model takes for split i (0 train, 1 val, 2 test) of a context_dataset:
    X alone, or [X, time features?, seasonal?] in the order _output expects."""
    parts = [d["X"][i][rows]] + [d[k][i][rows] for k in ("C", "S") if d[k][i] is not None]
    return parts[0] if len(parts) == 1 else parts


# ---------------------------------------------------------------------------
# Models (TensorFlow is imported inside, so scripts without models stay fast)
# ---------------------------------------------------------------------------
def _seasonal_gate_class():
    """Keras layer for Phase 10 (built lazily so TensorFlow is only imported when needed).

    gate = sigmoid(a - b * |seasonal(t) - last| / sd_f), one value per flow and sample, where
    sd_f is that flow's spread of (seasonal(t) - last) in training (fixed, not trained) and a, b
    are two learned numbers shared by all flows (b kept >= 0). The seasonal blend is multiplied
    by the gate: near 1 when a flow is close to its usual level, falling towards 0 when it is far
    from it (a burst), where pulling it back to the seasonal level is wrong. Starts almost open
    (a = 4, b ~ 0.13), so the model begins as the Phase 9 seasonal model and learns the gate.
    """
    global _SeasonalGate
    if "_SeasonalGate" in globals():
        return _SeasonalGate
    import tensorflow as tf

    @tf.keras.utils.register_keras_serializable(package="traffic")
    class SeasonalGate(tf.keras.layers.Layer):
        def __init__(self, sd=None, **kw):
            super().__init__(**kw)
            self.sd_init = None if sd is None else [float(v) for v in np.asarray(sd).ravel()]

        def build(self, shapes):
            n = shapes[0][-1]
            sd = self.sd_init if self.sd_init is not None else [1.0] * n
            self.sd = self.add_weight(name="sd", shape=(n,), trainable=False,
                                      initializer=tf.keras.initializers.Constant(sd))
            self.a = self.add_weight(name="a", shape=(), initializer=tf.keras.initializers.Constant(4.0))
            self.b_raw = self.add_weight(name="b_raw", shape=(), initializer=tf.keras.initializers.Constant(-2.0))

        def call(self, inputs):
            seas, blend = inputs                         # seas (n, 2, F), blend (n, F)
            z = tf.abs(seas[:, 0, :]) / tf.maximum(self.sd, 1e-6)
            gate = tf.sigmoid(self.a - tf.nn.softplus(self.b_raw) * z)
            return blend * gate

        def get_config(self):
            return {**super().get_config(), "sd": self.sd_init}

    _SeasonalGate = SeasonalGate
    return SeasonalGate


def _output(inp, x, n_flows, residual, name, n_context=0, seasonal=False, gate_sd=None):
    """Final Dense(n_flows) layer, optionally residual.

    residual=False: the Dense layer predicts the next value directly (the plan).
    residual=True:  prediction = last input step + Dense correction. The Dense
    layer starts at zero, so before training the model is exactly persistence,
    and training only learns how the next step differs from the current one.

    Phase 9 extras (residual only); the model then takes [X, context?, seasonal?]:
    n_context > 0  a vector of n_context numbers (time of day / week) joins the network's
                   summary right before the correction layer.
    seasonal=True  a (2, n_flows) input: every flow's seasonal value at the target time and
                   at the last input time, minus the last value. Each flow gets its own two
                   weights and bias (a per-flow blend, as the Phase 8 baseline), added to the
                   correction. Starts at zero, so the model is still persistence before training.
    gate_sd        (Phase 10, with seasonal) per-flow training spread of seasonal(t) - last; the
                   seasonal term is scaled down when a flow is far from its usual level (see
                   _seasonal_gate_class).
    """
    from tensorflow.keras import layers, models
    inputs = [inp]
    if n_context:
        ctx = layers.Input(shape=(n_context,), name="context")
        x = layers.Concatenate(name="with_context")([x, ctx])
        inputs.append(ctx)
    if seasonal:
        seas = layers.Input(shape=(2, n_flows), name="seasonal")
        inputs.append(seas)
    if not residual:
        if n_context or seasonal:
            raise ValueError("extra inputs need residual=True")
        return models.Model(inp, layers.Dense(n_flows)(x), name=name)
    correction = layers.Dense(n_flows, kernel_initializer="zeros", name="correction")(x)
    if seasonal:
        blend = layers.Flatten()(layers.DepthwiseConv1D(2, depthwise_initializer="zeros", name="seasonal_blend")(seas))
        if gate_sd is not None:                          # Phase 10: switch the seasonal term off in bursts
            blend = _seasonal_gate_class()(sd=gate_sd, name="seasonal_gate")([seas, blend])
        correction = layers.Add(name="correction_plus_seasonal")([correction, blend])
    window = inp.shape[1]
    last = layers.Flatten(name="last_step")(layers.Cropping1D((window - 1, 0))(inp))
    return models.Model(inputs if len(inputs) > 1 else inp,
                        layers.Add(name="last_plus_correction")([last, correction]), name=name + "_res")


def build_lstm(window=WINDOW, n_flows=N_FLOWS, units=(100, 50), dropout=0.2, residual=False, n_context=0,
               seasonal=False):
    """LSTM-only: the planned CNN-LSTM with the CNN layers removed."""
    from tensorflow.keras import layers
    inp = layers.Input(shape=(window, n_flows))
    x = layers.LSTM(units[0], return_sequences=True)(inp)
    x = layers.LSTM(units[1])(x)
    x = layers.Dropout(dropout)(x)
    return _output(inp, x, n_flows, residual, "lstm", n_context, seasonal)


def build_cnn(window=WINDOW, n_flows=N_FLOWS, filters=64, kernel=3, dropout=0.2, residual=False, n_context=0,
              seasonal=False):
    """CNN-only: the planned CNN-LSTM with the LSTM layers replaced by Flatten.

    padding="causal" so no output step can see a later input step.
    """
    from tensorflow.keras import layers
    inp = layers.Input(shape=(window, n_flows))
    x = layers.Conv1D(filters, kernel, padding="causal", activation="relu")(inp)
    x = layers.Conv1D(filters, kernel, padding="causal", activation="relu")(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(dropout)(x)
    return _output(inp, x, n_flows, residual, "cnn", n_context, seasonal)


def build_cnn_lstm(window=WINDOW, n_flows=N_FLOWS, filters=64, kernel=3,
                   units=(100, 50), dropout=0.2, residual=False, n_conv=2, pool=True, n_context=0,
                   seasonal=False, gate_sd=None):
    """The planned hybrid: causal Conv1D x2 -> MaxPooling -> LSTM x2 -> Dropout -> Dense.

    n_conv, pool and len(units) exist for the Phase 5 ablations (remove one part).
    n_context, seasonal: extra inputs (Phase 9); gate_sd: seasonal gate (Phase 10); see _output.
    """
    from tensorflow.keras import layers
    x = inp = layers.Input(shape=(window, n_flows))
    for _ in range(n_conv):
        x = layers.Conv1D(filters, kernel, padding="causal", activation="relu")(x)
    if pool:
        x = layers.MaxPooling1D(2)(x)
    for i, u in enumerate(units):
        x = layers.LSTM(u, return_sequences=i < len(units) - 1)(x)
    x = layers.Dropout(dropout)(x)
    return _output(inp, x, n_flows, residual, "cnn_lstm", n_context, seasonal, gate_sd)


def make_loss(kind, scaler=None, ref=None):
    """Training loss by name, all on scaled predictions/targets.

    "mse"      mean squared error in scaled units (the plan; Phase 3).
    "mae"      mean absolute error in scaled units.
    "mae_raw"  mean absolute error in ORIGINAL units (undoes min-max and log1p
               inside the graph), divided by `ref` (e.g. the mean training
               traffic) so the numbers stay near 1. This is the metric the
               report scores, so the model is trained on what it is judged on.
    """
    if kind in ("mse", "mae"):
        return kind
    if kind != "mae_raw":
        raise ValueError(f"unknown loss {kind!r}")
    import tensorflow as tf
    lo = tf.constant(scaler.min_, tf.float32)
    rng = tf.constant(scaler.range_, tf.float32)
    ref = float(ref)

    def mae_raw(y_true, y_pred):
        back = lambda z: tf.math.expm1(z * rng + lo)
        return tf.reduce_mean(tf.abs(back(y_true) - back(y_pred))) / ref
    mae_raw.__name__ = "mae_raw"
    return mae_raw


def train_model(model, name, X_train, y_train, X_val, y_val, epochs=100, batch_size=128,
                lr=1e-3, patience=10, verbose_every=5, loss="mse"):
    """Train with Adam (loss: see make_loss), keep the best epoch by validation loss.

    Callbacks: ModelCheckpoint (best model to models/<name>.keras), EarlyStopping
    (stop after `patience` epochs without improvement, restore the best weights),
    ReduceLROnPlateau (halve the learning rate after patience // 2 such epochs).
    Training examples are shuffled each epoch; that never mixes splits.
    Returns the training history as a dict of lists.
    """
    import time
    import tensorflow as tf

    class Progress(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if (epoch + 1) % verbose_every == 0 or epoch == 0:
                print(f"  epoch {epoch + 1:3d}  loss {logs['loss']:.5f}  val_loss {logs['val_loss']:.5f}"
                      f"  lr {float(self.model.optimizer.learning_rate.numpy()):.1e}")

    path = MODELS / f"{name}.keras"
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss=loss)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(path, monitor="val_loss", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience,
                                         restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                             patience=max(patience // 2, 1), min_lr=1e-5),
        Progress(),
    ]
    t = time.time()
    hist = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs,
                     batch_size=batch_size, shuffle=True, callbacks=callbacks, verbose=0)
    h = {k: [float(v) for v in vals] for k, vals in hist.history.items()}
    best = int(np.argmin(h["val_loss"]))
    print(f"  stopped after {len(h['loss'])} epochs ({time.time() - t:.0f} s); best epoch {best + 1}, "
          f"val_loss {h['val_loss'][best]:.5f}; saved {path.name}")
    return h


def fit_or_load(build_fn, name, X_train, y_train, X_val, y_val, retrain=False, seed=SEED,
                **train_args):
    """Train `build_fn()` and save it, or load the saved model if it already exists.

    The training history is kept next to the model (models/<name>_history.json),
    so a script can be re-run for its figures without retraining. Pass
    retrain=True (or delete the files) after changing the model or the data.
    Returns (model, history, trained) where trained says whether it trained now.
    """
    import json
    import tensorflow as tf
    path, hpath = MODELS / f"{name}.keras", MODELS / f"{name}_history.json"
    if not retrain and path.exists() and hpath.exists():
        model = load_saved(name)                                 # only used to predict
        history = json.loads(hpath.read_text())
        print(f"  loaded saved model {path.name} ({len(history['loss'])} epochs, "
              f"best val_loss {min(history['val_loss']):.5f}); set RETRAIN = True to train again")
        return model, history, False
    set_seeds(seed)                              # every model starts from the same seed
    model = build_fn()
    history = train_model(model, name, X_train, y_train, X_val, y_val, **train_args)
    hpath.write_text(json.dumps(history))
    return model, history, True


def load_saved(name):
    """Load models/<name>.keras for prediction (registers the project's custom layers first)."""
    import tensorflow as tf
    _seasonal_gate_class()
    return tf.keras.models.load_model(MODELS / f"{name}.keras", compile=False)


def predict(model, X, batch_size=1024, scaler=None, max_raw=None):
    """Model predictions in scaled units.

    With `scaler`, forecasts are limited to the physically possible range:
    not below zero traffic and not above `max_raw` (default: the fastest Abilene
    link, as cleaning rule R4; GÉANT passes GEANT_MAX_FLOW_KBPS).
    Without it an input outside the training range can make the network
    extrapolate, and expm1 then turns a small scaled error into a huge one.
    """
    p = model.predict(X, batch_size=batch_size, verbose=0)
    if scaler is not None:
        lo = (np.log1p(0.0) - scaler.min_) / scaler.range_
        hi = (np.log1p(MAX_FLOW_RAW if max_raw is None else max_raw) - scaler.min_) / scaler.range_
        p = np.clip(p, lo, hi)
    return p


# ---------------------------------------------------------------------------
# Output report: everything a script prints and plots, on one HTML page
# ---------------------------------------------------------------------------
class _Tee:
    """Write to the real stdout and also record the text for the report."""

    def __init__(self, report, stream):
        self.report, self.stream = report, stream

    def write(self, text):
        self.stream.write(text)
        self.report._add_text(text)
        return len(text)

    def flush(self):
        self.stream.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


class Report:
    """Collects a script's printed output and figures, in order, for the results page.

    report = Report("01_data", "Phase 1: Data layer")       # start capturing
    report.section("2. Missing-data policy")                # a heading
    print(...)                                              # recorded
    show(fig, "01_daily_cycle.png", "caption")              # saved + recorded
    report.finish()                                         # update the page, open it

    Each script's part is saved as results/reports/<name>.json, and the single
    page results/report.html is rebuilt from all parts (see build_results_page).
    If the script stops with an error, its part is still saved, with the error.
    """

    def __init__(self, name, title):
        global _REPORT
        if isinstance(sys.stdout, _Tee):            # re-run in the same session
            sys.stdout = sys.stdout.stream
        if _REPORT is not None:                     # the old report must not write on exit
            _REPORT.done = True
        self.name, self.title = name, title
        self.blocks = []                            # ("text"|"heading"|"image"|"error", ...)
        self.started = datetime.now()
        self.path = REPORTS / f"{name}.json"
        self.done = False
        sys.stdout = _Tee(self, sys.stdout)
        _REPORT = self
        import atexit
        atexit.register(self._at_exit)
        self._old_hook = sys.excepthook
        sys.excepthook = self._on_error

    def _add_text(self, text):
        if self.blocks and self.blocks[-1][0] == "text":
            self.blocks[-1] = ("text", self.blocks[-1][1] + text)
        else:
            self.blocks.append(("text", text))

    def section(self, title):
        self.blocks.append(("heading", title))
        sys.stdout.stream.write(f"\n== {title} ==\n")

    def image(self, png_path, caption=""):
        import base64
        data = base64.b64encode(Path(png_path).read_bytes()).decode()
        self.blocks.append(("image", data, caption, Path(png_path).name))

    def _on_error(self, kind, value, tb):
        import traceback
        self.blocks.append(("error", "".join(traceback.format_exception(kind, value, tb))))
        self.finish(open_browser=True)
        self._old_hook(kind, value, tb)

    def _at_exit(self):
        if not self.done:
            self.finish()

    def finish(self, open_browser=True):
        """Save this script's part, rebuild results/report.html, and open it."""
        import html
        import json
        import webbrowser
        if isinstance(sys.stdout, _Tee):
            sys.stdout = sys.stdout.stream
        self.done = True
        seconds = (datetime.now() - self.started).total_seconds()
        failed = any(b[0] == "error" for b in self.blocks)
        parts = []
        for b in self.blocks:
            if b[0] == "heading":
                parts.append(f"<h3>{html.escape(b[1])}</h3>")
            elif b[0] == "text" and b[1].strip():
                parts.append(f"<pre>{html.escape(b[1].strip(chr(10)))}</pre>")
            elif b[0] == "image":
                cap = f"<figcaption>{html.escape(b[2])} <span>{html.escape(b[3])}</span></figcaption>"
                parts.append(f'<figure><div class="plate"><img src="data:image/png;base64,{b[1]}" '
                             f'alt="{html.escape(b[2] or b[3])}"></div>{cap}</figure>')
            elif b[0] == "error":
                parts.append(f'<h3>Error</h3><pre class="err">{html.escape(b[1])}</pre>')
        record = {"name": self.name, "title": self.title, "failed": failed,
                  "when": self.started.isoformat(timespec="seconds"),
                  "seconds": round(seconds), "body": "\n".join(parts)}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(record), encoding="utf-8")
        page = build_results_page()
        print(f"\nresults page updated: {page}")
        if open_browser:
            webbrowser.open(page.as_uri())
        return page


RESULTS_PAGE = RESULTS / "report.html"


def build_results_page():
    """Rebuild results/report.html: a summary of metrics.csv, then every script's part.

    Called by Report.finish(); can also be called on its own. The summary shows
    the latest row for each distinct run (same name, dataset, horizon and notes);
    metrics.csv itself keeps every run.
    """
    import html
    import json
    records = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(REPORTS.glob("*.json"))]
    latest = max((r["when"] for r in records), default="")

    df = show_results()
    if len(df):
        df = df.drop_duplicates(subset=["name", "dataset", "horizon", "notes"], keep="last")
        rows = []
        for _, r in df.iterrows():
            scale, unit = (MBIT, "Mbit/s") if r["dataset"] == "abilene" else (1.0, "raw")
            rows.append(
                f"<tr><td>{html.escape(str(r['name']))}</td><td>{html.escape(str(r['dataset']))}</td>"
                f"<td class=num>{r['horizon']}</td><td class=num>{r['mae'] * scale:,.3f}</td>"
                f"<td class=num>{r['rmse'] * scale:,.3f}</td><td>{unit}</td>"
                f"<td class=num>{r['mape']:.1f}%</td><td class=num>{r['mape_coverage'] * 100:.0f}%</td>"
                f"<td>{html.escape(str(r['notes']))}</td><td class=when>{html.escape(str(r['timestamp']))}</td></tr>")
        summary = ("<div class=table><table><thead><tr><th>Model</th><th>Dataset</th><th class=num>Horizon</th>"
                   "<th class=num>MAE</th><th class=num>RMSE</th><th>Unit</th><th class=num>MAPE</th>"
                   "<th class=num>Coverage</th><th>Notes</th><th>Logged</th></tr></thead><tbody>"
                   + "".join(rows) + "</tbody></table></div>"
                   "<p class=note>Latest result for each distinct run; <code>results/metrics.csv</code> keeps every run. "
                   "MAPE only counts values at least 10% of the average, so read it with its coverage.</p>")
    else:
        summary = "<p class=note>Nothing logged yet.</p>"

    nav, sections = [], []
    for r in records:
        when = datetime.fromisoformat(r["when"]).strftime("%d %b %Y, %H:%M")
        badge = ('<span class="bad">Stopped with an error</span>' if r["failed"]
                 else '<span class="ok">Finished</span>')
        new = ' <span class="new">latest run</span>' if r["when"] == latest else ""
        nav.append(f'<li><a href="#{r["name"]}">{html.escape(r["title"])}</a>{new}</li>')
        sections.append(
            f'<section id="{r["name"]}"><div class="phase-head"><div class="eyebrow">scripts/{r["name"]}.py</div>'
            f'<h2>{html.escape(r["title"])}</h2><div class="meta">{badge}<span>Run {when}, took {r["seconds"]} s</span>'
            f'<a href="#top">back to top</a></div></div>{r["body"]}</section>')

    page = _RESULTS_HTML.format(
        updated=datetime.now().strftime("%d %b %Y, %H:%M"), summary=summary,
        nav="".join(nav) or "<li>No scripts run yet.</li>", sections="\n".join(sections))
    RESULTS_PAGE.write_text(page, encoding="utf-8")
    return RESULTS_PAGE


_REPORT = None

# Figure colours: reference palette categorical slots, in fixed order, plus ink tones
COLORS = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100",
          "violet": "#4a3aa7", "red": "#e34948",
          "ink": "#0b0b0b", "ink2": "#52514e", "grid": "#e4e3df"}
MBIT = ABILENE_UNIT_BYTES * 8 / 300 / 1e6    # raw units -> Mbit/s averaged over the 5 minutes


def use_plot_style():
    """Shared matplotlib style: white background, recessive grid, text in ink tones."""
    import matplotlib.pyplot as plt
    c = COLORS
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white", "axes.edgecolor": c["grid"],
        "axes.labelcolor": c["ink2"], "xtick.color": c["ink2"], "ytick.color": c["ink2"],
        "text.color": c["ink"], "axes.grid": True, "grid.color": c["grid"], "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.5,
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.titlelocation": "left",
    })


def show(fig, filename, caption=""):
    """Save a figure to results/figures/, add it to the running Report, then
    display it (in VS Code's Interactive Window) or close it (in a terminal run)."""
    import matplotlib.pyplot as plt
    path = FIGURES / filename
    fig.savefig(path, dpi=150)
    if _REPORT is not None and not _REPORT.done:
        _REPORT.image(path, caption)
    if "ipykernel" in sys.modules:
        plt.show()
    else:
        plt.close(fig)
    return path


_RESULTS_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Traffic Forecasting Results</title>
<style>
:root {{ --bg:#f6f7f8; --surface:#fff; --fg:#15191e; --muted:#57606b; --line:#dde1e6;
  --code:#eef1f4; --ok:#1d7a4a; --ok-bg:#e3f3ea; --bad:#b42318; --bad-bg:#fde8e6; --link:#1f63b8;
  color-scheme:light; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#121519; --surface:#1a1e23; --fg:#e8ebee;
  --muted:#a1aab4; --line:#2c3239; --code:#0d1013; --ok:#6fd39e; --ok-bg:#173226;
  --bad:#ff8a80; --bad-bg:#3a1714; --link:#7fb2f5; color-scheme:dark; }} }}
body {{ margin:0; background:var(--bg); color:var(--fg);
  font:15px/1.6 "Segoe UI", system-ui, -apple-system, sans-serif; }}
main {{ max-width:1000px; margin:0 auto; padding-inline:20px; padding-block:32px 64px;
  display:flex; flex-direction:column; gap:14px; }}
header {{ display:flex; flex-direction:column; gap:6px; margin-bottom:8px; }}
.eyebrow {{ font:500 12px/1 Consolas, ui-monospace, monospace; letter-spacing:.06em; color:var(--muted); }}
h1 {{ margin:0; font:700 30px/1.2 Georgia, "Times New Roman", serif; text-wrap:balance; }}
h2 {{ margin:0; font:700 24px/1.25 Georgia, "Times New Roman", serif; }}
h3 {{ margin:22px 0 0; font:600 18px/1.3 Georgia, "Times New Roman", serif; }}
section {{ display:flex; flex-direction:column; gap:14px; padding-top:28px; margin-top:20px;
  border-top:2px solid var(--line); min-width:0; }}
.phase-head {{ display:flex; flex-direction:column; gap:6px; }}
.meta {{ display:flex; flex-wrap:wrap; gap:10px; align-items:center; color:var(--muted); font-size:14px; }}
.meta a, nav a {{ color:var(--link); }}
nav ol {{ margin:0; padding-left:22px; display:flex; flex-direction:column; gap:4px; }}
.new {{ margin-left:8px; padding:1px 8px; border-radius:999px; font-size:12px; font-weight:600;
  background:var(--code); color:var(--muted); }}
.table {{ overflow-x:auto; border:1px solid var(--line); border-radius:8px; background:var(--surface); }}
table {{ border-collapse:collapse; width:100%; font-size:13.5px; }}
th, td {{ text-align:left; padding:8px 12px; border-bottom:1px solid var(--line); vertical-align:top; }}
tr:last-child td {{ border-bottom:0; }}
th {{ background:var(--code); color:var(--muted); font-weight:600; white-space:nowrap; }}
td.num, th.num {{ text-align:right; font-variant-numeric:tabular-nums; white-space:nowrap; }}
td.when {{ color:var(--muted); white-space:nowrap; font-size:12.5px; }}
.note {{ margin:0; color:var(--muted); font-size:14px; }}
code {{ font:13px Consolas, ui-monospace, monospace; }}
.ok, .bad {{ padding:3px 10px; border-radius:999px; font-weight:600; font-size:13px; }}
.ok {{ background:var(--ok-bg); color:var(--ok); }}
.bad {{ background:var(--bad-bg); color:var(--bad); }}
pre {{ margin:0; padding:14px 16px; background:var(--code); border:1px solid var(--line);
  border-radius:8px; font:13px/1.55 Consolas, ui-monospace, monospace; overflow-x:auto; }}
pre.err {{ color:var(--bad); }}
figure {{ margin:0; display:flex; flex-direction:column; gap:8px; }}
.plate {{ background:#fff; border:1px solid var(--line); border-radius:8px; padding:8px; }}
.plate img {{ display:block; width:100%; height:auto; }}
figcaption {{ color:var(--muted); font-size:14px; }}
figcaption span {{ font-family:Consolas, ui-monospace, monospace; font-size:12px; opacity:.8; }}
</style></head>
<body><main id="top">
<header>
  <div class="eyebrow">CNN-LSTM network traffic forecasting · Abilene</div>
  <h1>Traffic Forecasting Results</h1>
  <div class="meta"><span>Page rebuilt {updated} from every script's latest run</span></div>
</header>
<h3>All logged results</h3>
{summary}
<h3>Phases on this page</h3>
<nav><ol>{nav}</ol></nav>
{sections}
</main></body></html>
"""


# ---------------------------------------------------------------------------
# Self test
# ---------------------------------------------------------------------------
def self_test(verbose=True):
    """Run the whole pipeline on synthetic data. Touches no project files."""
    rng = np.random.default_rng(SEED)
    T = STEPS_PER_WEEK * 2
    t = np.arange(T)
    daily = 1 + 0.8 * np.sin(2 * np.pi * t / 288)                 # 288 steps = 1 day
    base = rng.lognormal(mean=10, sigma=2, size=N_FLOWS)          # flow sizes vary widely
    series = (daily[:, None] * base[None, :]
              * rng.lognormal(0, 0.1, size=(T, N_FLOWS))).astype(np.float32)
    series[:, 0] = 0.0                                            # an always-zero flow
    series[500:900, 1] = 0.0                                      # a zero stretch

    X, y = make_windows(series, WINDOW, HORIZON)
    assert X.shape == (T - WINDOW, WINDOW, N_FLOWS) and y.shape == (T - WINDOW, N_FLOWS)
    assert np.array_equal(X[1, -1], series[WINDOW]) and np.array_equal(y[0], series[WINDOW])

    X_tr, y_tr, X_va, y_va, X_te, y_te = chronological_split(X, y)
    assert len(X_tr) + len(X_va) + len(X_te) == len(X)
    assert np.array_equal(X_va[0], X[len(X_tr)])                  # order preserved

    scaler = FlowScaler().fit(X_tr)                               # train only
    Z_tr, z_tr = scaler.transform(X_tr), scaler.transform(y_tr)
    Z_va, z_va = scaler.transform(X_va), scaler.transform(y_va)
    Z_te, z_te = scaler.transform(X_te), scaler.transform(y_te)
    assert Z_tr.dtype == np.float32
    assert Z_tr.min() >= -1e-6 and Z_tr.max() <= 1 + 1e-6
    assert np.allclose(scaler.inverse_transform(z_te), y_te, rtol=1e-4, atol=1e-2)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        save_splits(Z_tr, z_tr, Z_va, z_va, Z_te, z_te, directory=tmp)
        loaded = load_splits(directory=tmp)
        assert all(np.array_equal(a, b) for a, b in
                   zip(loaded, (Z_tr, z_tr, Z_va, z_va, Z_te, z_te)))
        scaler.save(tmp / "scaler.npz")
        s2 = FlowScaler.load(tmp / "scaler.npz")
        assert np.allclose(s2.inverse_transform(z_te), scaler.inverse_transform(z_te))

        # persistence baseline: next = current, evaluated in raw units
        pred = scaler.inverse_transform(Z_te[:, -1, :])
        m = evaluate(y_te, pred)
        assert m["mae"] > 0 and 0 < m["mape_coverage"] <= 1
        assert evaluate(y_te, y_te)["mae"] == 0

        log_result("self_test_persistence", m, path=tmp / "metrics.csv")
        assert len(show_results(tmp / "metrics.csv")) == 1

    # Phase 8 helpers
    vals = np.arange(20, dtype=float)[:, None] * np.ones((1, 3))
    known = np.ones(20, bool); known[5] = False
    lb = lookback(vals, known, np.array([7, 8, 2]), 3)
    assert np.isnan(lb[1]).all() and lb[0, 0] == 4 and np.isnan(lb[2]).all()   # step 5 unknown, step -1 outside
    sm = seasonal_mean(vals, known, np.array([12]), 3, n=3)                        # mean of steps 9, 6, 3
    assert np.allclose(sm, 6.0)
    Xr = rng.normal(size=(500, 6)) * [1, 10, 100, 1, 1, 1] + 5
    Wt = rng.normal(size=(6, 2))
    Yr = Xr @ Wt + 3.0
    (W0, b0), (W1, _) = ridge_path(Xr, Yr, [1e-12, 10.0])
    assert np.allclose(W0, Wt, atol=1e-6) and np.allclose(b0, 3.0, atol=1e-6)    # no penalty = least squares
    assert np.abs(W1).sum() < np.abs(W0).sum()                                    # penalty shrinks
    la = np.abs(rng.normal(size=3000))
    assert dm_test(la, la, 10)[2] > 0.99                                          # identical: no difference
    assert dm_test(la * 0.8, la, 10)[2] < 1e-6                                     # 20% lower loss: clear
    g, lo, hi = block_bootstrap_gain(la * 0.9, la, 50)
    assert abs(g - 10) < 1e-9 and lo <= g <= hi

    if verbose:
        print(f"self_test passed: X {X.shape}, train/val/test "
              f"{len(X_tr)}/{len(X_va)}/{len(X_te)}, persistence MAE {m['mae']:.1f}, "
              f"MAPE {m['mape']:.1f}% (coverage {m['mape_coverage']:.2f})")
    return True


ensure_dirs()

__all__ = [
    "PROJECT", "RAW", "PROCESSED", "MODELS", "RESULTS", "FIGURES", "METRICS_CSV",
    "ensure_dirs", "N_FLOWS", "WINDOW", "HORIZON", "STEPS_PER_WEEK", "COLS_PER_ROW",
    "SPLIT_FRACTIONS", "SEED", "ABILENE_NODES", "ABILENE_COLUMN", "ABILENE_UNIT_BYTES",
    "ABILENE_WEEK_STARTS", "set_seeds", "FlowScaler", "make_windows", "make_windows_segments",
    "chronological_split", "save_splits", "load_splits", "evaluate", "evaluate_scaled",
    "persistence", "log_result",
    "show_results", "status", "load_abilene_week", "load_abilene_raw", "abilene_segments",
    "abilene_clean", "prepare_abilene", "step_label", "GEANT_DIR", "GEANT_NODES", "GEANT_FLOWS", "GEANT_STEP_S", "load_geant_raw", "geant_clean", "prepare_geant", "load_geant_routing", "warn_prf", "warn_onsets", "GEANT_MAX_FLOW_KBPS", "GEANT_MAX_ABSENT", "GEANT_MIN_TOTAL", "load_abilene_routing", "self_test",
    "REPORTS", "Report", "show", "COLORS", "MBIT", "use_plot_style",
    "RESULTS_PAGE", "build_results_page", "build_lstm", "build_cnn", "build_cnn_lstm",
    "make_loss", "train_model", "fit_or_load", "predict",
    "STEPS_PER_DAY", "clean_timeline", "lookback", "seasonal_mean", "ridge_path", "dm_test",
    "block_bootstrap_gain", "saved_model_names", "data_dir", "MAX_FLOW_RAW",
    "time_features", "segment_ids", "context_dataset", "model_inputs", "load_saved", "fold_dataset",
]

if __name__ == "__main__":
    status()
    self_test()
