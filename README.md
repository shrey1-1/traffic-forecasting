# CNN-LSTM Based Network Traffic Forecasting

Short-term forecasting of origin-destination traffic on research backbone networks (Abilene, then GÉANT),
converted to link loads through the routing matrix to flag congestion before it happens.

This is about **computer network traffic** (bytes between routers), not road traffic.

## Main findings so far

- A residual CNN-LSTM trained on MAE in original units beats persistence ("the next value equals the
  last one") by a small but statistically significant margin: about 1% at 5-30 minutes on Abilene and
  0.6-3% on GÉANT (Diebold-Mariano and block-bootstrap tests, `docs/PROJECT_LOG.md` section 19).
- **Seasonal inputs** (each flow's mean at the same time on the last 7 days) are what the model was
  missing: on Abilene they raise the test gain over persistence at 5 minutes from +1.1% to **+7.1%**,
  and to **+8-12%** at 5-60 minutes on 143 of 144 flows (section 20).
- One bursty flow (LOSA to CHIN), far outside its training range in the test period, decides most
  rankings; robust handling of such out-of-range bursts is the open problem.
- Link-level early warning is not yet better than reacting to the current load (sections 16-17).

The full, plain-language record of every phase, decision and result is in
[`docs/PROJECT_LOG.md`](docs/PROJECT_LOG.md).

## Data

The data is not in this repository; download it into these folders:

| Dataset | Source | Put in |
|---|---|---|
| Abilene (24 weeks, 5-minute traffic matrices, 12 routers) | https://www.cs.utexas.edu/~yzhang/research/AbileneTM/ | `raw/` (`X01.gz` .. `X24.gz`, `A`, `links`, `topo-*.txt`, `readme.txt`) |
| GÉANT (TOTEM, 15-minute matrices, 23 routers, Jan-Apr 2005) | https://totem.info.ucl.ac.be/dataset.html (`traffic-matrices-anonymized-v2.tar.bz2`) | `raw_geant/totem/` |

`processed/`, `models/` and the results page are rebuilt by the scripts.

## Setup (Windows)

```
py -3.13 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python common.py        # prints status and runs self_test()
```

TensorFlow does not support Python 3.14 yet, so use Python 3.13.

Open the folder in VS Code. `.vscode/settings.json` selects `.venv` as the interpreter.
The scripts in `scripts/` are split into cells by `# %%` markers: click **Run Cell** above a
block and its output (including plots) appears in the Interactive Window. You can also run a
whole script in a terminal, e.g. `.venv\Scripts\python scripts\00_setup.py`.
Start with `scripts/00_setup.py`; it prints "Phase 0 complete" when the environment is ready.
Then run the scripts in number order; trained models are saved in `models/` and reused on later runs.

**All results are on one page: `results/report.html`.** Every script saves what it prints and
plots, then rebuilds this page. The page starts with a table of every logged result, then has one
section per phase.

### GPU training (optional, WSL2)

TensorFlow only uses an NVIDIA GPU on Linux, so GPU training runs in WSL2:

```
wsl --install -d Ubuntu-24.04
wsl -d Ubuntu-24.04 -u root -- bash -c "apt-get update && apt-get install -y python3-venv && python3 -m venv /root/tfgpu && /root/tfgpu/bin/pip install -r requirements-gpu.txt"
wsl -d Ubuntu-24.04 -u root -- bash scripts/wsl/run.sh scripts/00_setup.py
```

(run from the project folder). `scripts/wsl/env.sh` points TensorFlow at the CUDA libraries that pip
installs. GPU and CPU training give slightly different numbers, so models from the two are never mixed
in one comparison (Phase 9 models are named `p9g_*`). Each Abilene training job needs about 3.5 GB of
RAM; with WSL's default memory limit, run at most two at once.

## Layout

| Path | Contents |
|---|---|
| `docs/PROJECT_LOG.md` | Plain-language record of everything done, decisions and results |
| `docs/presentation/` | Review slides and viva Q&A |
| `common.py` | All shared code (paths, constants, scaler, windowing, split, metrics, models, report page) |
| `scripts/00_setup.py` | Phase 0: environment check |
| `scripts/01_data.py` | Phase 1: parse Abilene, missing-data policy, scale, window, split, save |
| `scripts/02_baselines.py` | Phase 2: persistence and ARIMA |
| `scripts/03_models.py` | Phase 3: LSTM-only and CNN-only |
| `scripts/03b_hybrid.py`, `03c_tuning.py` | Phase 4: CNN-LSTM, loss choice, tuning |
| `scripts/03d_horizons.py` | Phase 5: 30 and 60 minutes ahead, ablations |
| `scripts/04_analysis.py` | Phase 6: link loads and congestion warnings (Abilene) |
| `scripts/03e_geant.py`, `03f_geant_tuning.py` | Phase 6: GÉANT |
| `scripts/05_errors.py` | Phase 7: error analysis |
| `scripts/06_baselines2.py` | Phase 8: seasonal, blend, ridge and multi-step ARIMA baselines; significance tests |
| `scripts/07_features.py` | Phase 9: time of day, seasonal inputs, longer windows |
| `scripts/wsl/` | GPU helpers for WSL2 and the Phase 9 checks |
| `results/metrics.csv` | Every logged run |
| `results/figures/` | All figures |
| `raw/`, `raw_geant/`, `processed/`, `models/` | Data and checkpoints (not in git) |

## Data contract

- `X`: (samples, 12, 144) float32, `y`: (samples, 144) float32 (GÉANT: 529 flows)
- Scaling: log1p, then per-flow min-max to [0, 1], fitted on the training split only
- Split: chronological 60/20/20, never random; windows never cross a data gap
- WINDOW = 12 steps, HORIZON = 1, 6 or 12 (Abilene, 5-minute steps) and 1 or 4 (GÉANT, 15-minute steps)

## Rules

- Never split randomly; never fit the scaler on the full dataset.
- Invert scaling before computing metrics.
- Choose settings on validation only; report persistence alongside every result.
- Log every run to `results/metrics.csv` with `log_result()`.
