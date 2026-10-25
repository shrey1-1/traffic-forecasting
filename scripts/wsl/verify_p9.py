"""Phase 9 checks (run on the GPU environment): no look-ahead in the seasonal inputs, and the
Abilene result without the bursty LOSA->CHIN flow (86), as in Phase 8."""
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))      # the project folder
import numpy as np
import tensorflow as tf
from common import *

mae = lambda y, p: float(np.mean(np.abs(p - y)))
g = lambda m, b: (1 - m / b) * 100

# 1. look-ahead: rebuild one test sample's seasonal input by hand from the raw timeline
values, known, to_time = clean_timeline("abilene")
for h in (1, 12):
    d = context_dataset("abilene", h, seasonal="seasonal_mean7")
    i = 1234
    t = d["times"][2][i]
    last = d["X"][2][i, -1]
    latest = 0
    for j, when in enumerate((t, t - h)):
        src = [when - k * 288 for k in range(1, 8) if known[when - k * 288]]
        assert max(src) <= t - h, "seasonal input uses a step after the last input"
        latest = max(latest, max(src))
        manual = d["scaler"].transform(np.mean(values[src], 0)) - last
        assert np.allclose(manual, d["S"][2][i, j], atol=1e-5)
    print(f"h{h}: seasonal input rebuilt by hand and matches; the latest step it uses is {t - latest} steps "
          f"before the target (the last input is {h} before), so no look-ahead")
    del d

# 2. without flow 86, and per-flow wins
keep86 = np.arange(N_FLOWS) != 86
for h, key in ((1, "seas"), (6, "seas"), (6, "time_seas"), (12, "seas"), (12, "long_time_seas")):
    win = 36 if key.startswith("long") else 12
    seas = "seasonal_mean7" if "seas" in key else None
    d = context_dataset("abilene", h, window=win, seasonal=seas, use_time="time" in key)
    base = context_dataset("abilene", h)
    k2 = d["kept"][2]
    y, last = d["y_raw"][2], d["last_raw"][2]
    def ens(dd, prefix, rows=slice(None)):
        out = []
        for s in (42, 43, 44):
            m = tf.keras.models.load_model(MODELS / f"{prefix}_s{s}.keras", compile=False)
            out.append(dd["scaler"].inverse_transform(predict(m, model_inputs(dd, 2), scaler=dd["scaler"])).astype(float)[rows])
            tf.keras.backend.clear_session()
        return np.mean(out, 0)
    p = ens(d, f"p9g_abilene_h{h}_{key}")
    pb = ens(base, f"p9g_abilene_h{h}_base", k2)
    e = lambda q, f=slice(None): mae(y[:, f], q[:, f])
    flows_better = int((np.abs(p - y).mean(0) < np.abs(pb - y).mean(0)).sum())
    print(f"h{h:2d} {key:15s} test vs persistence: all {g(e(p), e(last)):+5.1f}%  without 86 {g(e(p, keep86), e(last, keep86)):+5.1f}%"
          f"  | base: all {g(e(pb), e(last)):+5.1f}%  without 86 {g(e(pb, keep86), e(last, keep86)):+5.1f}%"
          f"  | flow 86 only: {g(e(p, [86]), e(last, [86])):+6.1f}% vs base {g(e(pb, [86]), e(last, [86])):+6.1f}%"
          f"  | beats base on {flows_better}/144 flows")
    del d, base
