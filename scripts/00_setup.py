"""00 Setup: Phase 0 check.

Run all cells (or `python scripts/00_setup.py`). Phase 0 is done when this
prints "Phase 0 complete": common.py is found, libraries import, self_test()
passes, and TensorFlow builds the planned CNN-LSTM (143,608 parameters).
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
import tensorflow as tf  # imported before set_seeds() so TF is seeded too
set_seeds()
status()

# %% library versions
import matplotlib, pandas, statsmodels
for mod in (np, pandas, matplotlib, statsmodels, tf):
    print(f"{mod.__name__:12s} {mod.__version__}")
print("GPUs:", tf.config.list_physical_devices("GPU") or "none (CPU is fine for this model)")

# %% pipeline self test on synthetic data
self_test()

# %% smoke test: build the planned model (not trained here; training is in 03_models)
from tensorflow.keras import layers, models

model = models.Sequential([
    layers.Input(shape=(WINDOW, N_FLOWS)),
    layers.Conv1D(64, 3, padding="causal", activation="relu"),
    layers.Conv1D(64, 3, padding="causal", activation="relu"),
    layers.MaxPooling1D(2),
    layers.LSTM(100, return_sequences=True),
    layers.LSTM(50),
    layers.Dropout(0.2),
    layers.Dense(N_FLOWS),
])
model.summary()
assert model.count_params() == 143_608, model.count_params()

# %% one forward pass on random input in [0, 1]
out = model(np.random.rand(4, WINDOW, N_FLOWS).astype("float32"))
assert out.shape == (4, N_FLOWS)
print("forward pass OK:", out.shape)
print("Phase 0 complete")
