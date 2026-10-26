# Project Log: What Has Been Done

**Project:** CNN-LSTM Based Network Traffic Forecasting (with proactive congestion avoidance)
**Last updated:** 25 October 2026
**Current stage:** All improvement phases (8-12) done, sections 19-23; overall findings in section 24.

This document records, in simple terms, what has been built so far, why, and what is still left.
It is updated at the end of each phase.

---

## 1. What the project is about (in one paragraph)

A backbone network (like Abilene, the US research network) has routers that send data to each
other. For every pair of routers (an "origin-destination flow") we can measure how many bytes
were sent every 5 minutes. If we can **predict** how much traffic each flow will carry in the
next few minutes, we can work out how full each network link will be **before** it happens,
and warn about congestion early instead of reacting after links are already overloaded.
The prediction model is a **CNN-LSTM**: a CNN layer picks up short patterns across the last hour
of traffic, and an LSTM layer learns how those patterns change over time.

This is about **computer network traffic** (bytes between routers), not road traffic.

---

## 2. Summary of progress

| Phase | What it is | Status |
|---|---|---|
| 0 Scaffold | Folders, shared code, scripts, Python environment, self test | **Done** (section 8) |
| 1 Data layer | Read the Abilene files, clean, scale, cut into samples | **Done** (section 10) |
| 2 Baselines | Simple "predict the same as now" method and ARIMA | **Done** (section 11) |
| 3 Single models | LSTM-only and CNN-only | **Done** (section 13) |
| 4 Hybrid | CNN-LSTM, tuned | **Done** (section 14) |
| 5 Depth | Predicting further ahead, removing parts to test them | **Done** (section 15) |
| 6 Application | Link loads, congestion warnings, GÉANT dataset | **Done** (sections 16 and 17) |
| 7 Consolidation | Error analysis, report, presentation | Error analysis **done** (section 18); report and presentation not started |
| 8 Stronger baselines | Seasonal, blend, ridge, multi-step ARIMA; significance tests | **Done** (section 19) |
| 9 Better inputs | Time of day, seasonal values, longer window (trained on the GPU) | **Done** (section 20) |
| 10 Peaks and bursts | Keep the seasonal gain without the loss in bursts (fallback rule, learned gate) | **Done** (section 21) |
| 11 Early warning redone | Better forecasts and calibrated 90th-percentile margins for warnings | **Done** (section 22) |
| 12 Robustness | Rolling-origin evaluation, more seeds, error analysis rerun | **Done** (section 23) |

**Results so far:** the CNN-LSTM beats persistence by a small amount that is statistically significant at
5-30 minutes on Abilene and at both horizons on GÉANT (section 19); see section 18.5 for the main findings.

---

## 3. The project folder

Every folder in the project has one job:

```
traffic_project/
├── common.py          all shared code (the only place it lives)
├── requirements.txt   list of Python libraries needed
├── README.md          setup instructions
├── .gitignore         tells git which files not to track
├── .vscode/           VS Code settings (which Python to use)
├── docs/              this document
├── scripts/           one script per stage: 00_setup ... 04_analysis
├── raw/               the original Abilene data files (not in git)
├── processed/         cleaned, ready-to-use data (not in git, can be regenerated)
├── models/            saved trained models (not in git)
└── results/           report.html (all results on one page, section 12), metrics.csv
                       (a log of every experiment), figures/, reports/
```

**Why this layout:** raw data, processed data and trained models are large and can always be
recreated, so they are kept out of version control (git). Results and code are small and
important, so they are tracked. Empty folders contain a tiny `.gitkeep` file only so that git
keeps the folder.

---

## 4. The shared code: `common.py`

Everything that more than one script needs is written **once**, in `common.py`. Each script
loads it at the top. This avoids having slightly different copies of the same function in
different places, which is a common source of mistakes.

### 4.1 Paths and folders
`common.py` works out where the project is from its own location, so the code works no matter
which folder you run it from. (It can be overridden with the environment variable
`TRAFFIC_PROJECT`.) It also creates any missing folders automatically.

### 4.2 Fixed settings (constants)

| Name | Value | Meaning |
|---|---|---|
| `N_FLOWS` | 144 | Abilene has 12 routers, so 12 × 12 = 144 origin-destination flows |
| `WINDOW` | 12 | The model looks at the last 12 readings = 1 hour (12 × 5 minutes) |
| `HORIZON` | 1 | It predicts 1 step (5 minutes) ahead. Later: 6 and 12 steps |
| `STEPS_PER_WEEK` | 2016 | 7 days × 288 five-minute readings per day |
| `COLS_PER_ROW` | 720 | Each Abilene row has 144 flows × 5 different estimates of each |
| `SPLIT_FRACTIONS` | 60 / 20 / 20 | Training / validation / test share of the data |
| `SEED` | 42 | Fixed random seed so results can be repeated exactly |

### 4.3 Repeatable results: `set_seeds()`
Neural networks start from random numbers. Fixing the "seed" means the same random numbers are
used every run, so if a result changes, it is because the code or settings changed, not luck.
`set_seeds()` fixes the seed for Python, NumPy and (when it is loaded) TensorFlow.

### 4.4 Scaling the data: `FlowScaler`
Traffic values range from almost zero to very large numbers, and some flows are thousands of
times bigger than others. Neural networks learn badly from such uneven numbers, so we rescale
them in two steps:

1. **log1p** (take the logarithm of value + 1). This shrinks huge values much more than small
   ones, so one very busy flow does not dominate everything. The "+1" makes zero safe.
2. **Min-max per flow.** Each flow is then stretched separately so that its smallest value
   becomes 0 and its largest becomes 1.

**Important rule:** the minimum and maximum are measured **only on the training data**. If we
measured them on all the data, information about the future (the test period) would leak into
training and make the results look better than they really are.

The scaler can also **undo** the scaling (`inverse_transform`), which is needed before
measuring errors, and can be saved to and loaded from `processed/scaler.npz`, so later stages
use exactly the same scaling. It refuses data with missing values (NaN) or negative numbers,
so problems are caught early instead of silently producing wrong results.

### 4.5 Turning a time series into examples: `make_windows()`
A model learns from examples of the form "here is the input, here is the right answer".
`make_windows()` slides a 1-hour window along the traffic data:

- **Input (X):** 12 consecutive readings of all 144 flows
- **Answer (y):** the reading of all 144 flows `HORIZON` steps after the window ends

Then the window moves forward one step and the next example is made. The result has the
agreed shapes: X is (examples, 12, 144) and y is (examples, 144).

### 4.6 Splitting the data: `chronological_split()`
The examples are split **in time order**: the first 60% for training, the next 20% for
validation (used to tune settings), the last 20% for testing. It is never shuffled randomly,
because in real use we always predict the future from the past. A random split would let the
model see data from "after" the test period during training.

### 4.7 Saving and loading: `save_splits()` and `load_splits()`
Preparing the data is done once (in `01_data`). The six resulting arrays (training, validation
and test, inputs and answers) are saved to `processed/` and every later script simply loads
them, so all experiments use exactly the same data.

### 4.8 Measuring error: `evaluate()`
Compares predictions with the true values, **in the original units** (after undoing the
scaling), and reports:

- **MAE** (mean absolute error): on average, how far off the prediction is.
- **RMSE** (root mean squared error): like MAE but punishes large mistakes more.
- **MAPE** (mean absolute percentage error): the error as a percentage.
- **MAPE coverage**: what fraction of values were included in MAPE.

**Why the MAPE coverage:** many flows are close to zero. A tiny absolute error on a near-zero
value becomes an enormous percentage and makes MAPE meaningless. So MAPE is only calculated
where the true value is at least 10% of the average, and the coverage figure says how much of
the data that was. The two must always be reported together.

### 4.9 Recording experiments: `log_result()` and `show_results()`
Every experiment adds one row to `results/metrics.csv` (time, model name, dataset, horizon,
the four metrics, notes). This creates a permanent record instead of relying on output that
disappears when the program is closed. `show_results()` displays the table.

### 4.10 Project check: `status()`
Prints where the project is, the Python and NumPy versions, how many of the 24 Abilene files
are present, whether processed data and the scaler exist, how many models are saved, and how
many experiments are logged.

### 4.11 Abilene data functions (added in Phase 1)
`load_abilene_raw()` reads the 24 weekly files, `abilene_clean()` applies the missing-data
policy, `prepare_abilene()` runs the whole Phase 1 pipeline, and `load_abilene_routing()` reads
the routing matrix. They are explained in section 10.

### 4.12 Automatic check: `self_test()`
Runs the whole pipeline on **made-up data** to prove the code works before real data is used.
It creates two weeks of fake traffic with a daily up-and-down pattern, including one flow that
is always zero and one with a long zero stretch (like the real data's gaps). It then checks:

- windows have the right shapes and the answer is the right time step
- the split keeps time order and loses no examples
- the scaler, fitted on training data only, puts training values between 0 and 1
- undoing the scaling gives back the original numbers
- saving and loading the data and scaler gives identical results
- the simplest forecast ("next value = current value") can be evaluated and logged

It works in a temporary folder, so it never touches the real project files.

### 4.13 Added in later phases
- **Baselines and evaluation (Phase 2):** `persistence()`, `evaluate_scaled()`; section 11.5.
- **The results page:** `Report`, `show()`, `build_results_page()`; section 12.
- **Models and training (Phase 3):** `build_lstm`, `build_cnn`, `build_cnn_lstm`, `train_model`,
  `fit_or_load`, `predict`; section 13.6.

---

## 5. The scripts (`scripts/`)

There is one script per stage. Each is a normal Python file divided into **cells** by
`# %%` lines. In VS Code, clicking **Run Cell** above a cell runs just that part and shows the
output and graphs in a side panel (the Interactive Window). A script can also be run whole from
the terminal, e.g. `python scripts\00_setup.py`.

Every script starts with the same **header cell**, which finds `common.py`, loads the latest
version of it, fixes the random seeds and prints the project status.

| Script | Purpose | Contents now |
|---|---|---|
| `00_setup.py` | Phase 0 check | Complete (see below) |
| `01_data.py` | Read and prepare Abilene data (run once) | Complete (section 10) |
| `02_baselines.py` | Simple comparison methods | Complete (section 11) |
| `03_models.py` | LSTM, CNN, CNN-LSTM experiments | Phase 3 done (section 13); Phases 4-5 to come |
| `04_analysis.py` | Link loads, congestion, GÉANT, peak errors | Header and plan |

### What `00_setup.py` checks
1. Prints the versions of all libraries, and whether a GPU exists (a GPU is not needed).
2. Runs `self_test()`.
3. Builds the planned CNN-LSTM model (without training it) and confirms its size.
4. Passes random data through it once to confirm the output has the right shape.
5. Prints **"Phase 0 complete"** if everything passed.

---

## 6. The planned model

| Layer | What it does | Parameters |
|---|---|---|
| Conv1D, 64 filters, size 3, causal | Finds short patterns across 3 consecutive readings | 27,712 |
| Conv1D, 64 filters, size 3, causal | Combines those into slightly longer patterns | 12,352 |
| MaxPooling1D(2) | Halves the time length, keeping the strongest signals | 0 |
| LSTM, 100 units | Learns how the patterns change over time | 66,000 |
| LSTM, 50 units | Summarises the sequence into one state | 30,200 |
| Dropout 0.2 | Randomly switches off 20% of units while training, to avoid memorising | 0 |
| Dense, 144 outputs | One prediction per flow | 7,344 |
| **Total** | | **143,608** |

**"Causal" padding** means each step can only see the past, never the future. Without it,
the CNN could quietly use information from later in the window and give unrealistically good
results. This parameter count is checked automatically in `00_setup.py`.

The model is small, so it can be trained on a normal laptop CPU.

---

## 7. Decisions made, and why

| Decision | Reason |
|---|---|
| Python **3.13** for the project environment, not 3.14 | TensorFlow does not support Python 3.14 yet. Python 3.13 was already installed. |
| A separate virtual environment (`.venv`) inside the project | Keeps this project's libraries apart from the rest of the computer; deleting `.venv` removes them cleanly. |
| **`.py` scripts with `# %%` cells instead of Jupyter notebooks** | Same cell-by-cell workflow in VS Code, but the files are plain text: easy to compare in git, runnable from the terminal. Results are saved to `metrics.csv` and `figures/` anyway, so notebooks' stored output is not needed. |
| TensorFlow only loaded by scripts that need it (`00`, `03`) | Phases 1 and 2 do not need TensorFlow, so they start faster and work even before it is installed. |
| MAPE only above a floor, always with coverage | Near-zero flows otherwise make MAPE meaningless. |
| VS Code settings point to `.venv` | So VS Code automatically uses the right Python. |
| Residual versions of the models, alongside the planned ones (Phase 3) | The planned models predict each flow's level from scratch and were about 3x worse than persistence. A residual model predicts "last value + correction", so it starts from persistence. Both versions are kept so the report can show the difference. |

### Problems found and fixed
- **Missing `np` in scripts.** `common.py` lists exactly what it shares, and NumPy (`np`) is not
  on that list, so the model test would have failed with "np is not defined". Each script now
  loads NumPy itself.
- **Wrong model count in `status()`.** The `.gitkeep` placeholder in `models/` was being counted
  as a saved model. Hidden files are now ignored.

---

## 8. What has been verified (Phase 0)

- `self_test()` passes, both with the computer's main Python 3.14 and inside `.venv`
  (Python 3.13). Its numbers come from made-up data and say nothing about real traffic.
- The libraries were installed in `.venv` by hand: TensorFlow 2.21.0, NumPy 2.5.3,
  pandas 3.0.6, matplotlib 3.11.2, statsmodels 0.15.0, ipykernel 7.4.0.
- `scripts/00_setup.py` prints **"Phase 0 complete"**: TensorFlow works and the planned model has
  exactly 143,608 parameters.
- Not done yet: library versions are not pinned in `requirements.txt`, and git is not set up.

---

## 9. Next steps

1. **Phase 4:** the CNN-LSTM, in both forms (as planned and residual), then tuning one setting at
   a time. First thing to test: a training loss closer to the final metric (see 13.5).
2. Pin library versions in `requirements.txt`; set up git and make the first commit.

---

## 10. Phase 1: the data layer

### 10.1 Getting the data
The 24 weekly files (`X01.gz` to `X24.gz`) and the small description files (`readme.txt`, `A`,
`links`, `demands`, `topo-2003-04-10.txt`) come from the dataset's original page:
https://www.cs.utexas.edu/~yzhang/research/AbileneTM/. They are in `raw/` (not in git). All 24
files were checked to be complete (each opens as a valid gzip file).

### 10.2 What the files contain
- Each file is one week: 2016 rows (one every 5 minutes) of 720 numbers.
- The 720 numbers are 144 flows × 5 versions. Flow number = source router × 12 + destination
  router, with routers in this order: ATLA-M5, ATLAng, CHINng, DNVRng, HSTNng, IPLSng, KSCYng,
  LOSAng, NYCMng, SNVAng, STTLng, WASHng.
- **Unit:** 100 bytes per 5 minutes (the 1-in-100 packet sampling is already scaled back).
  To get Mbit/s: value × 100 × 8 / 300 / 1,000,000.
- ATLA-M5 is a second Atlanta node. Papers that use "11 nodes / 121 flows" have merged or removed
  it. We keep all 12 nodes and 144 flows.

### 10.3 Choosing the column
Of the 5 versions of each flow, the README says the **first is `realOD`, the measured traffic**.
The other four are *estimates* made by gravity and tomogravity models, which guess the flows from
each router's totals. Checks on the data agree:
- all five versions have exactly the same per-router totals (as the estimation models require),
  so only one of them can be the measurement;
- in `01_column_check.png`, the estimate follows the measured flow in general but drifts away
  from it in places.

So the project uses `realOD` (`ABILENE_COLUMN = 0`).

**Daily cycle (exit condition):** `01_daily_cycle.png` shows whole-network traffic averaged by
time of day. It is lowest around 11:00 and highest around 18:00–20:00 (on the files' own clock,
assuming each file starts at 00:00, since the README gives only dates). The peak is 1.53× the
low point. The busiest single flow (WASHng → WASHng) shows the same day/night pattern.

### 10.4 Problems found in the data
1. **The weeks are not continuous.** The README's start dates have gaps (for example X02 starts
   8 March, X03 starts 2 April), so the data is not one unbroken 24-week series.
2. **Four duplicated days.** Weeks X10, X18, X21 and X22 each start with an *exact copy* of the
   previous week's last day (all 144 × 288 values identical). This happens even where the README
   says the weeks are 7 days apart, so the README dates are not fully reliable.
3. **A 22-day outage.** From the last day of X02 to the end of X05, every flow going *into*
   ATLA-M5 (except ATLA-M5 to itself) reads zero. Eleven flows switching off at the same moment
   is a measurement failure, not real traffic.
4. **A one-day outage.** On day 2 of X03, five busy flows into Denver (from WASH, NYCM, CHIN,
   IPLS and ATLA, which all reach Denver over the same link) read zero for exactly 24 hours.
5. **Two short router gaps.** All traffic out of KSCY is zero for 1.5 hours in X06, and all
   traffic out of LOSA for almost 2 hours in X11.
6. **One impossible value.** Flow ATLAng → ATLAng reads 5.33 × 10¹³ at X24 day 5 16:25. That is
   about 142 Tbit/s, roughly 14,000 times the fastest link (9.92 Gbit/s). Every other value in
   the dataset is below 71% of that link speed.
7. **Many ordinary zeros in small flows.** The smallest quarter of flows is zero about 19% of the
   time; the biggest quarter almost never. With only 1 packet in 100 sampled, a small flow often
   records nothing in 5 minutes. These zeros are *real measurements*.

### 10.5 The missing-data policy
Written as rules, so the code applies them automatically and gives the same result every time
(`abilene_clean()` in `common.py`):

| Rule | Detects | Action | Removed |
|---|---|---|---|
| R1 Duplicate | a week starting with an exact copy of the previous week's last day | remove the copy | 1,152 steps (4 days) |
| R2 Flow outage | a normally active flow (zero < 50% of the time) reading zero for ≥ 1 day | remove those time steps | 6,336 steps (22 days; problems 3 and 4) |
| R3 Router outage | all traffic into, or out of, one router at zero for ≥ 1 hour | remove those time steps | 40 steps (problem 5) |
| R4 Impossible value | a flow above the fastest link's capacity (9.92 Gbit/s) | remove that time step | 1 step (problem 6) |
| Everything else | ordinary zeros; large but possible spikes | **keep** | – |

**Why remove rather than fill in?** Filling gaps with guessed values would mean training on, and
later scoring against, numbers that were never measured. Removing them is simpler and honest, and
costs little: **84.4% of all time steps are kept** (40,855 of 48,384).

**Why keep large spikes?** They are physically possible bursts, and Phase 7 studies exactly how
the model handles peaks.

**Why R2 only for "normally active" flows?** One tiny flow (SNVAng → ATLA-M5) is zero 86% of the
time. Its long zero stretches are its normal state, so they must not trigger removal.

`01_missing_data.png` shows all 24 weeks with every removed period shaded by rule.

### 10.6 Segments: never crossing a gap
After removal, the kept data is cut into **segments**: stretches of truly consecutive 5-minute
steps. A new segment starts wherever steps were removed, and at every join between two weeks
that are not proven to be back to back (README dates not exactly 7 days apart, or a duplicate was
found there). This gives **9 segments**, from 2.3 to 46 days long.

The 1-hour windows are made **inside each segment only**, so no example ever mixes data from
both sides of a gap.

### 10.7 Final data
| Split | Examples | Covers (answer times) |
|---|---|---|
| Train | 24,448 | X01 day 1 to X16 day 4 |
| Validation | 8,149 | to X20 day 5 |
| Test | 8,150 | to X24 day 7 |

- Shapes: X is (examples, 12, 144) and y is (examples, 144), float32, as agreed.
- The scaler (log1p, then min-max per flow) is fitted on the **training inputs only**, so training
  values are exactly in [0, 1]. A very small share of validation/test values fall slightly
  outside (0.02% below, at most 0.26% above), because a few busy flows later go lower or higher
  than they ever did in training. These are **not clipped**: clipping would hide real behaviour,
  and errors are measured in original units anyway.
- `processed/target_steps.npy` records, for every example, which 5-minute step its answer is.
  Later phases use it to put results back on the timeline (time of day, week, peaks).
- Check: undoing the scaling on the test answers gives back exactly the original data.

### 10.8 Files added in Phase 1
- `common.py`: `load_abilene_week`, `load_abilene_raw` (with a cache file
  `processed/abilene_col0.npy`), `abilene_segments`, `abilene_clean`, `make_windows_segments`,
  `prepare_abilene`, `step_label`, `load_abilene_routing`.
- `scripts/01_data.py`: runs the checks, the policy, the figures and the save, top to bottom.
  When it finishes, its output goes onto the single results page (section 12).
- `results/figures/`: `01_column_check.png`, `01_daily_cycle.png`, `01_missing_data.png`.

### 10.9 Routing matrix (read now, used in Phase 6)
`load_abilene_routing()` reads `raw/A`: 54 links × 144 flows. Links 1–30 are backbone links
(28 at 9.92 Gbit/s; ATLAng–IPLSng in both directions at 2.48 Gbit/s). Links 31–54 are the entry
and exit points at each router. Every flow uses exactly one path (all fractions are 1), enters at
exactly one router and leaves at exactly one. Link loads will be `A @ traffic`.

---

## 11. Phase 2: baselines (the evaluation gate)

Every later model is judged against these numbers. All results are on the **test split**
(X20 day 5 to X24 day 7), measured in original units after undoing the scaling, and logged to
`results/metrics.csv`. Below they are shown in Mbit/s (average over the 5 minutes).

### 11.1 Persistence: "the next 5 minutes will look like the last 5 minutes"
The simplest possible forecast: the prediction for each flow is its most recent value. It needs
no training, and on 5-minute network traffic it is hard to beat, because traffic changes little
from one 5-minute step to the next.

| Split | Flows | MAE (Mbit/s) | RMSE (Mbit/s) | MAPE | MAPE coverage |
|---|---|---|---|---|---|
| Validation | all 144 | 2.742 | 19.285 | 17.0% | 73% |
| **Test** | **all 144** | **3.212** | **17.531** | **16.3%** | **74%** |

- MAPE is only calculated where the true value is at least 10% of the average, so it covers about
  three quarters of the values (see section 4.8).
- Relative to each flow's own size, the error is about 10% for the 10 busiest flows but about 55%
  for the 10 smallest. Small flows are mostly sampling noise, so they are much harder to predict.

### 11.2 ARIMA on the 5 busiest flows
ARIMA is the classic statistical forecasting model: it predicts the next value from a weighted
combination of recent values and recent errors. One model was fitted per flow, for the 5 busiest
flows (WASH→WASH, LOSA→CHIN, CHIN→LOSA, WASH→NYCM, NYCM→NYCM).

How it was done:
- **Fitting data:** the last 4 weeks of training data (X12 day 4 to X16 day 4), the longest
  unbroken stretch that ends where training ends. ARIMA needs an unbroken series.
- **Scale:** fitted on log1p of the traffic, like the neural networks will see it.
- **Choosing the model:** 7 standard ARIMA orders were tried per flow (e.g. ARIMA(1,0,0),
  ARIMA(2,1,1)). The one with the lowest error **on the validation split** was kept. The test
  split was not used for any choice.
- **Forecasting:** one step ahead, with the fitted parameters fixed, run separately inside each
  segment so it never crosses a gap. Each forecast uses only data from before that moment.
- **Fair comparison:** ARIMA is scored on exactly the same flows and test examples as persistence
  (`persistence_top5` in `metrics.csv`), and the code checks that both use identical true values.

| Flow | Chosen order | ARIMA test MAE | Persistence test MAE | ARIMA vs persistence |
|---|---|---|---|---|
| WASH→WASH | (2,0,1) | 31.05 | 33.20 | 6.5% better |
| LOSA→CHIN | (1,1,1) | 30.70 | 27.05 | 13.5% worse |
| CHIN→LOSA | (2,1,1) | 20.84 | 20.52 | 1.5% worse |
| WASH→NYCM | (2,1,1) | 14.69 | 14.21 | 3.4% worse |
| NYCM→NYCM | (1,0,0) | 12.09 | 11.43 | 5.8% worse |
| **All 5** | | **21.87** | **21.28** | **2.8% worse** |

### 11.3 What this means
- **ARIMA does not beat persistence on the test split.** On validation it was better on 4 of
  the 5 flows, but on test only on 1.
- Most of the loss comes from one flow, LOSA→CHIN, which is far burstier in the test period than in
  validation (persistence error 8.9 → 27.0 Mbit/s). ARIMA's forecasts are smoothed (see
  `02_forecast_day.png`: the blue line cuts through the spikes), so it reacts to a sudden burst
  later than persistence, which simply copies the last value.
- This is the **over-smoothing** problem in its simplest form, and it is what Phase 7 will study for
  the CNN-LSTM. It also confirms the plan's expectation that persistence is a strong baseline.
- **Checked and ruled out:** converting forecasts back from the log scale makes them slightly low
  on average (they are the median, not the mean). Correcting this to the mean was tried: it
  removes the bias but makes MAE worse, so the uncorrected forecast is the right one for MAE.
- **Short spikes.** The close-up in `02_forecast_day.png` shows that many spikes on the busiest
  flow last a single 5-minute step. Persistence pays for each one twice (it misses the spike, then
  repeats it one step late); ARIMA stays near the middle and misses the peak. In that window the
  spikes look evenly spaced, so this was checked over the longest unbroken stretch (46 days): the
  link between a value and the value 30 minutes earlier is weak (correlation at most 0.17), and the
  strongest links are always with the previous 1-3 steps. On WASH→WASH the biggest spikes fall at
  15 minutes past the hour about twice as often as chance (70 of 400 vs about 33), which may be a
  scheduled job, but there is no reliable schedule a model could simply learn.
- **Limitation:** ARIMA here has no daily seasonality. A seasonal ARIMA with a 288-step (1 day)
  season is very slow to fit; this can be noted as a limitation in the report.

### 11.4 The bar to beat
For one-step-ahead forecasts on the test split, **all 144 flows: MAE 3.212 Mbit/s** (persistence).
On the 5 busiest flows: persistence 21.28, ARIMA 21.87 Mbit/s.

### 11.5 Files added in Phase 2
- `common.py`: `persistence()`, `evaluate_scaled()` (undoes the scaling, then measures),
  `use_plot_style()`, `COLORS`, `MBIT` (shared figure style and unit conversion).
- `scripts/02_baselines.py`: persistence, ARIMA and figures; its output goes onto the results
  page (section 12).
- `results/figures/02_forecast_day.png` (one test day plus a 3-hour close-up), `02_mae_per_flow.png`.
- `results/metrics.csv`: the first 4 logged results.

---

## 12. The results page

**All results are on one page: `results/report.html`.** It opens in the browser automatically
whenever a script finishes.

- **Top:** a table of every logged result (model, dataset, horizon, MAE, RMSE, MAPE with its
  coverage, notes, time). Abilene numbers are shown in Mbit/s. If the same run was repeated, the
  table shows its latest result; `results/metrics.csv` still keeps every run.
- **Then one section per phase**, in order, with a contents list: everything the script printed and
  every figure, under the same headings as the script's cells. Each section shows when it was run,
  how long it took, and whether it finished or stopped with an error.
- **Running one script updates only its own section.** Each script saves its part in
  `results/reports/<script>.json`, and the page is rebuilt from all the parts.
- If a script stops with an error, its section still appears, marked in red, with the error.

How it works in the code: a script starts with `report = Report("02_baselines", "Phase 2: Baselines")`,
uses `report.section(...)` for headings and `show(fig, "name.png", "caption")` for figures, and ends
with `report.finish()`. `build_results_page()` in `common.py` rebuilds the page on its own.

The page and the parts are regenerated, so they are not stored in git.

---

## 13. Phase 3: single models (LSTM-only and CNN-only)

### 13.1 What was trained
Two model types, each the planned CNN-LSTM with one half removed (so Phase 5 can reuse them as
ablations):

| Model | Layers | Parameters |
|---|---|---|
| LSTM-only | LSTM(100) → LSTM(50) → Dropout(0.2) → Dense(144) | 135,544 |
| CNN-only | Conv1D(64, k=3, causal) ×2 → MaxPooling(2) → Flatten → Dropout(0.2) → Dense(144) | 95,504 |

Training, as planned: MSE loss on the scaled data, Adam (learning rate 0.001), up to 100 epochs,
batches of 128. Three automatic helpers:
- **ModelCheckpoint** saves the best model (lowest validation loss) to `models/`.
- **EarlyStopping** stops after 10 epochs without improvement and keeps the best epoch's weights.
- **ReduceLROnPlateau** halves the learning rate after 5 epochs without improvement.

Training examples are shuffled between epochs. This is allowed: each example is a separate 1-hour
window and all of them come from the training period, so nothing from validation or test leaks in.
The validation split decides when to stop; the test split is used only once, at the end. Training
is repeatable: the same seed gives the same result every run.

### 13.2 First result: the planned models lose clearly to persistence

| Test, all 144 flows | MAE (Mbit/s) | vs persistence |
|---|---|---|
| Persistence | 3.212 | – |
| LSTM-only | 9.076 | 2.8× worse |
| CNN-only | 10.821 | 3.4× worse |

What was checked before accepting this:
- **They did finish learning.** The learning rate had been lowered to its minimum and the
  validation loss was flat for the last epochs. More epochs would not help.
- **In the units they are trained on, they are close to persistence.** Validation loss (MSE on
  the scaled data): persistence 0.01426, LSTM-only 0.01223, CNN-only 0.01383.
- **One flow makes it much worse.** LOSA→CHIN normally carries about 100 Mbit/s but sometimes
  bursts to about 6,500 Mbit/s (6.5 Gbit/s) for several steps. Persistence follows a burst after its
  first step; the models keep predicting about 210 Mbit/s. On that flow the models' error is
  217 Mbit/s against persistence's 27.
- **But it is not only that flow.** Without LOSA→CHIN the LSTM is still 2.5× worse than
  persistence (7.6 vs 3.0 Mbit/s).

**Why:** the planned models predict each flow's *level* from scratch, and in practice they pull
their forecasts towards typical values. Persistence simply repeats the last value, which at
5-minute steps is very hard to beat, especially when traffic stays at a new level for a while.

### 13.3 The fix tested: residual models
A **residual** model predicts "the last value + a learned correction" instead of the value itself.
The correction layer starts at zero, so before training a residual model is exactly persistence;
training can only move it away from persistence if that lowers the loss. The rest of the model is
unchanged (same layers, same number of parameters). This was checked in code: an untrained residual
model gives exactly the persistence forecast.

Decision (by the project owner): **keep both versions.** The as-planned results stay in the record,
and the residual versions are added, so the comparison itself becomes a finding.

| Test, all 144 flows | MAE (Mbit/s) | vs persistence | Flows where it beats persistence |
|---|---|---|---|
| Persistence | 3.212 | – | – |
| LSTM-only (as planned) | 9.076 | 182.5% worse | 2 of 144 |
| CNN-only (as planned) | 10.821 | 236.8% worse | 2 of 144 |
| **LSTM-only, residual** | **3.226** | 0.4% worse | **48 of 144** |
| **CNN-only, residual** | **3.222** | 0.3% worse | 37 of 144 |

On the 5 busiest flows (test): persistence 21.281, ARIMA 21.871, **LSTM-only residual 21.247**
(0.2% better than persistence, the first model to beat both baselines there, narrowly),
CNN-only residual 21.312 Mbit/s.

### 13.4 What this means
- **The residual connection is essential.** It cuts the error by about two thirds and turns models
  that were 3× worse than persistence into models that match it.
- **Matching persistence is not yet beating it.** In `03_forecast_closeup.png` the residual models
  lie almost exactly on top of persistence: they learned only small corrections to the last value.
- The residual LSTM learned more than the residual CNN: its validation loss is 28% below
  persistence's (0.01032 vs 0.01426), the CNN's only 3.5% below (0.01376). The LSTM stopped by
  itself after 53 epochs; the CNN used all 100 with very slow progress.
- The planned models show a large gap between training and validation loss (`03_learning_curves.png`),
  meaning they memorise training patterns that do not carry over.

### 13.5 An open question for Phase 4
The residual LSTM is 28% better than persistence **in the loss it is trained on** (MSE on the log
scale), but only equal in MAE in Mbit/s. The training loss treats all 144 flows equally on a log
scale, while MAE in Mbit/s is dominated by the big flows. The working explanation is that its gains
are mostly on flows that matter little in Mbit/s. This is a hypothesis, not yet tested. Phase 4
will test it, for example by training with a loss closer to the final metric (MAE).

### 13.6 Files added in Phase 3
- `common.py`: `build_lstm`, `build_cnn`, `build_cnn_lstm` (each with `residual=True/False`),
  `train_model`, `fit_or_load` (trains a model, or loads it if already saved, with its training
  history), `predict`. `set_seeds()` now also makes TensorFlow repeatable.
- `scripts/03_models.py`: trains (or loads) the four models, compares them with the baselines, and
  adds its section to the results page. `RETRAIN = True` at the top forces retraining.
- `models/`: `lstm`, `cnn`, `lstm_res`, `cnn_res` (`.keras` files and `_history.json`).
- `results/figures/`: `03_learning_curves.png`, `03_test_mae.png`, `03_forecast_closeup.png`,
  `03_mae_per_flow.png`.

---

## 14. Phase 4: CNN-LSTM, loss and tuning

### 14.1 The loss was the missing piece
The open question of 13.5 is answered. Residual models trained on **MAE in original units**
(the traffic in Mbit/s, undone from the log scale inside the training graph: `make_loss("mae_raw")`)
are the first to beat persistence on all 144 flows. MAE on the scaled data does not help
(val 2.772 vs persistence 2.742), so the gain comes from matching the scale of the reported metric.

Step 1 (`03b_hybrid.py`, validation MAE in Mbit/s; persistence 2.742): CNN-LSTM as planned 5.833
(2x worse, same failure as the other planned models); residual with MSE 2.777 (see 14.2);
residual with scaled MAE 2.772; **residual with original-units MAE: CNN-LSTM 2.693, LSTM-only 2.646.**

### 14.2 A forecast blow-up, and its fix
The residual CNN-LSTM trained with MSE had a validation MAE of 23.99. One validation example
(flow 48) had an input at 1.34 in scaled units, above the training maximum of 1.0; the model
extrapolated to 1.99 and `expm1` turned that into about 25 million Mbit/s. Fix: `predict(..., scaler=)`
now limits forecasts to the physical range, zero up to the fastest link (the same limit as cleaning
rule R4). All Phase 4 tuning and seed numbers use it. Without the one bad example that model scored 2.777.

### 14.3 Tuning (`03c_tuning.py`, validation only)
Residual CNN-LSTM, original-units MAE, up to 200 epochs (patience 15). One setting at a time; a change
is kept only if validation MAE improves by more than 0.5%. Base: 2.684 Mbit/s.

| Setting | Tried | Result | Kept |
|---|---|---|---|
| filters | 32, 128 | 2.731, 2.731 | 64 |
| kernel | 2, 5 | 2.705, 2.731 | 3 |
| LSTM units | (64,32), (128,64) | 2.688, 2.725 | (100,50) |
| dropout | 0.0, 0.4 | 2.729, 2.705 | 0.2 |
| learning rate | 3e-4, 3e-3 | 2.691, 2.679 | 1e-3 |

**No change beat the base by more than 0.5%**, so the original settings stay. Differences between
trials are inside the noise (seed spread is 0.03). Window length was not tuned: it changes the data
and is left open (Phase 5 can test it with the horizon work).

### 14.4 Three seeds: CNN-LSTM against LSTM-only (same settings, same loss)

| MAE, Mbit/s | Validation | vs persistence | Test | vs persistence | Seeds beating persistence (test) |
|---|---|---|---|---|---|
| Persistence | 2.742 | – | 3.212 | – | – |
| CNN-LSTM, residual | 2.690 ± 0.033 | +1.9% | 3.200 ± 0.005 | +0.4% | 3 of 3 |
| LSTM-only, residual | 2.616 ± 0.009 | +4.6% | 3.190 ± 0.010 | +0.7% | 3 of 3 |

### 14.5 What this means
- All six models beat persistence on the test split, but only by 0.4-0.7%. At 5-minute,
  one-step-ahead forecasts persistence is very hard to beat; this is a small real gain, not a large one.
- **The CNN did not help here.** LSTM-only is better than CNN-LSTM on validation in every seed and by
  more than the seed spread (4.6% vs 1.9%). The "CNN picks up short patterns" idea is not supported
  at this horizon. Phase 5's ablations test whether it matters at 6 and 12 steps.
- The validation-to-test gap (4.6% vs 0.7%) is the same effect seen with ARIMA: the test period is
  burstier than validation.

### 14.6 Files added in Phase 4
`common.py`: `make_loss`, `loss=` and `seed=` in `train_model`/`fit_or_load`, `predict(scaler=)`.
`scripts/03b_hybrid.py`, `scripts/03c_tuning.py`; figures `03b_val_mae.png`, `03c_tuning.png`,
`03c_seeds.png`; about 30 models in `models/`.

---

## 15. Phase 5: longer horizons and ablations (`03d_horizons.py`)

Horizons 1, 6 and 12 steps (5, 30, 60 minutes). Each has its own data and scaler (`processed/h6`,
`processed/h12`) and its own persistence reference, so MAE is only comparable within a horizon.
All models residual, original-units MAE loss, Phase 4 settings. No selection step here; val and test both shown.

### 15.1 Main comparison (mean of 3 seeds; change vs persistence, positive = better)

| Horizon | Persistence test MAE | CNN-LSTM | LSTM-only | CNN-only |
|---|---|---|---|---|
| 1 (5 min) | 3.212 | +0.4% | +0.7% | +0.3% |
| 6 (30 min) | 5.300 | **+0.8%** | **-12.3%** (0 of 3 seeds beat persistence) | +0.9% |
| 12 (60 min) | 6.157 | **+1.1%** | **-12.4%** (0 of 3) | +0.9% |

On validation every model beats persistence at every horizon (+1.2% to +4.6%), LSTM-only most.

### 15.2 The LSTM-only failure on test is one flow
At horizon 6 the LSTM-only's extra test error over persistence comes from **LOSA→CHIN alone**
(+90 Mbit/s MAE on that flow; the next worst flow is +0.8). The same flow carries the 6.5 Gbit/s bursts
seen in 13.2; test inputs reach 1.28 in scaled units, above the training maximum of 1.0. The LSTM-only
extrapolates on those inputs; the CNN-LSTM on the same flow is only +2.9 Mbit/s worse than persistence.
So the CNN does earn its place, but as **robustness to out-of-range bursts**, not as a better average
forecaster. This is a finding for the Phase 7 error analysis, and also a warning that validation
rankings (where LSTM-only won) did not carry to the burstier test period.

### 15.3 Ablations (1 seed, test MAE change vs persistence)

| Variant | h=1 | h=6 | h=12 |
|---|---|---|---|
| Full CNN-LSTM (3-seed mean) | +0.4% | +0.8% | +1.1% |
| 1 conv layer | +0.3% | +0.3% | 0.0% |
| No pooling | +0.6% | +0.4% | -1.5% |
| 1 LSTM layer | +0.7% | +0.4% | +1.1% |

Single seeds, so differences of about 1% are noise (seed spread is 0.5-1%). The only clear signals:
dropping pooling or a conv layer does not help, and no ablation beats the full model clearly.

### 15.4 What this means
- Gains over persistence stay small (about 1%) even at 60 minutes; the improvement grows a little
  with horizon (0.4% -> 1.1%), as expected, but is not large. The persistence MAE itself grows 2x.
- CNN-only is as good as CNN-LSTM at 6 and 12 steps. The LSTM adds little on test; the CNN adds robustness.
- Not done: ARIMA at longer horizons; window length tuning; seasonal baselines ("same time yesterday").

Files: `scripts/03d_horizons.py`, figures `03d_horizons.png`, `03d_ablations.png`, `processed/h6`, `processed/h12`.

---

## 16. Phase 6 (Abilene part): link loads and congestion warnings (`04_analysis.py`)

### 16.1 Method
Forecast flows (raw units per 5 minutes) are mapped to the 30 backbone links with the routing matrix
(`A @ flows`), converted to Mbit/s and divided by link capacity to give utilisation. Compared:
**persistence** (the reactive baseline: the link will look as it looks now) and the **CNN-LSTM**
(mean of 3 seeds, Phase 4/5 models), at horizons 1, 6, 12. Test split only. A warning is a forecast
utilisation at or above the threshold; an **onset** is a link going from below to at/above it.

### 16.2 The 80% threshold in the plan cannot be used
No Abilene backbone link reaches 80% (busiest: 75% test, 70% validation), so 80% gives zero events.
**Primary threshold changed to 50%**, fixed before any warning result was seen; 40% and 60% are
shown as sensitivity. This is a deviation from the plan, and should be stated in the report.

### 16.3 Results (test, 50% threshold)

| Horizon | Link utilisation MAE, persistence / CNN-LSTM | F1 persistence / CNN-LSTM | Onsets warned in advance (both) |
|---|---|---|---|
| 5 min | 0.203 / 0.204 pp | 0.97 / 0.97 | 0 of 35 |
| 30 min | 0.381 / 0.386 pp | 0.93 / 0.94 | 5 of 35 |
| 60 min | 0.494 / 0.502 pp | 0.90 / 0.91 | 0 of 35 |

(pp = percentage points of capacity, over all 30 links. Same pattern at 40% and 60%.)

### 16.4 What this means
- **The CNN-LSTM gives no early-warning advantage over persistence on Abilene.** Link utilisation MAE is
  0.2-1.5% *worse* than persistence, F1 differs by at most 0.01, and the number of onsets warned in advance
  is the same or lower (60% threshold, 60 min: 9 vs 12 of 26). The high F1 comes from episodes being long:
  once a link is congested it stays so, which persistence already exploits.
- **The events are one flow.** At 50%, 1,091 congested link-steps occur on 10 of 30 links, but five links
  on one path (LOSA to CHIN: LOSA-SNVA-DNVR-KSCY-IPLS-CHIN) hold about 210 steps each. Setting the LOSA to CHIN
  flow to zero cuts congested link-steps from 1,091 to 43. So there are only a handful of independent
  episodes, and the statistics are weak. Validation had about 40 congested link-steps in total.
- The flow-level gain of about 1% (Phase 5) does not carry through to link level, because the link
  loads are dominated by the biggest flows and the one bursty flow, which the model forecasts no better
  than persistence.
- Honest framing for the report: the pipeline works end to end (flows to link loads to warnings), but on
  5-minute Abilene data a reactive forecast is already close to as good as the CNN-LSTM.

### 16.5 GÉANT
See section 17.

---

## 17. Phase 6 (GÉANT part): a second network (`03e_geant.py`)

### 17.1 The data
TOTEM GÉANT traffic matrices (https://totem.info.ucl.ac.be/dataset.html, archive
`traffic-matrices-anonymized-v2.tar.bz2`), in `raw_geant/` (not in git): 23 nodes = 529 flows, one matrix per
15 minutes, 1 Jan to 29 Apr 2005, in kbit/s, 10,772 files in a 11,396-slot timeline. Nodes are anonymised
(ids 1-23); flow = source index x 23 + destination index.

Problems found and the policy (`geant_clean` in `common.py`):

| Rule | Detects | Removed |
|---|---|---|
| G1 | no file for the slot (8 gaps; longest 456 slots = 4.75 days) | 624 |
| G2 | more than 50% of the usual flows absent (normal is 5-20%; 4 slots were 79-96%) | 4 |
| G3 | a flow above 10 Gbit/s, the core link speed (122 values in 9 slots, up to 9.2 Tbit/s) | 9 |
| G4 | network total below 20% of the median | 4 |

**Absent pairs are set to 0.** Every file lists only some pairs (5-20%), the absent ones are mostly small flows
(the flows absent most often have a median mean of 48 kbit/s) and no listed value is ever exactly 0 (the smallest
is 0.249 kbit/s). So absence is read as traffic below the measurement floor. This is an assumption, not a fact.
5 flows never appear and stay all-zero. 94.4% of slots are kept, in 13 segments (longest 36.7 days); windows never
cross a gap. Split 60/20/20: 6,384 train / 2,128 val / 2,129 test windows (about a quarter of Abilene's training
size); test = 6 Apr to 29 Apr. Test inputs again exceed the training range (max 1.56), so forecasts are limited
to 0-10 Gbit/s (`predict(..., max_raw=GEANT_MAX_FLOW_KBPS)`).

### 17.2 Generalisation test
Same architecture, residual form, original-units MAE loss and Phase 4 settings (window 12 = 3 hours), **retrained
on GÉANT, no retuning**, 3 seeds. (The Abilene models cannot be applied directly: the flow counts differ.)
Horizons 1 (15 min) and 4 (60 min, the same wall-clock time as Abilene's 12 steps).

| Test MAE, Mbit/s | Persistence | CNN-LSTM | LSTM-only | CNN-only |
|---|---|---|---|---|
| 15 min | 1.469 | 1.464 (+0.3%) | 1.464 (+0.3%) | 1.468 (+0.1%) |
| 60 min | 2.377 | **2.313 (+2.7%)** | 2.344 (+1.4%) | 2.342 (+1.4%) |

All CNN-LSTM and LSTM-only seeds beat persistence at both horizons (CNN-only: 2 of 3 at 15 min). Validation agrees.
At 60 minutes the CNN-LSTM beats persistence on 459 of 529 flows, and by 5.2% on the 10 busiest.

### 17.3 What this means
- **The Abilene result generalises in direction and size:** gains over persistence are about 0.3% at the shortest
  horizon and grow to a few percent at 60 minutes. On GÉANT the gain at 60 minutes (2.7%) is larger than on
  Abilene (1.1%) and the CNN-LSTM is the best of the three models, consistently over seeds; on Abilene validation
  LSTM-only ranked first. With a different network, data and 4x the flows, the ranking is therefore not stable;
  only "small gain that grows with horizon" is.
- **Caveats:** 3 seeds, one split, one network each; the GÉANT time zone of the timestamps is not documented (the
  daily cycle shows a low at 14:00 and a peak at 21:00, which looks shifted, but the models do not use time of day).

### 17.4 Link level on GÉANT (approximate, do not over-read)
The anonymised topology has no IGP weights and no capacities. Flows were routed over minimum-hop paths with equal
splitting (74 directed links, `load_geant_routing`), and "high load" means a link load in the top 10% of its own
training-period loads. This is **not congestion against capacity**.

| Test | Load MAE persistence / CNN-LSTM | F1 persistence / CNN-LSTM | Precision | Recall |
|---|---|---|---|---|
| 15 min | 9.13 / 9.33 Mbit/s | 0.84 / 0.83 | 0.84 / 0.88 | 0.84 / 0.78 |
| 60 min | 15.96 / 15.99 Mbit/s | 0.76 / 0.73 | 0.76 / 0.84 | 0.76 / 0.65 |

The CNN-LSTM is more cautious than persistence: fewer false alarms (higher precision) but more missed events
(lower recall), and no better F1. At 60 minutes it warns 491 of 2,129 onsets in advance against persistence's 832
(the relative threshold flickers, so onset counts are noisy). The flow-level gain does not carry to link level here
either, the same conclusion as Abilene (section 16.4), and consistent with a model that smooths towards typical values.

### 17.5 Files added
`common.py`: `load_geant_raw`, `geant_clean`, `prepare_geant`, `load_geant_routing`, `warn_prf`, `warn_onsets`
(shared with `04_analysis.py`), `prepare_abilene` refactored onto `_prepare_segments` (output checked identical),
`predict(max_raw=)`. `scripts/03e_geant.py`; figures `03e_data.png`, `03e_models.png`; `processed/geant/`; 18 models `g_h*`.
`raw_geant/` holds the downloaded data and `dl/geant.txt` (SNDlib network file, not used).

---

## 18. Phase 7: error analysis (`05_errors.py`)

Test split, CNN-LSTM forecast = mean of 3 seeds (LSTM-only too on Abilene), Abilene at 5/30/60 min and GÉANT at
15/60 min, pooled over flows in original units. Definitions are fixed in the script header, none tuned on test.
Note: the 3-seed mean forecast is slightly better than a single seed (Abilene 5 min: +0.9% vs +0.4% for a single
seed), so these numbers are a little more favourable than the per-seed tables in sections 14-17.

### 18.1 The earlier hypothesis was wrong in direction
Sections 11-13 suggested the models "over-smooth": gain on ordinary wobble, lose on big moves. The data say the
opposite. MAE change vs persistence by size of the true move (move = |true - last value| / the flow's mean):

| Move size (percentile) | Abilene 5 min | Abilene 30 min | Abilene 60 min | GÉANT 15 min | GÉANT 60 min |
|---|---|---|---|---|---|
| smallest 50% | -3.9% | -8.7% | -13.2% | -7.7% | -16.1% |
| 50-80% | +1.3% | +1.7% | +2.5% | -0.3% | +2.2% |
| 80-95% | +2.3% | +3.0% | +4.4% | +2.2% | +6.5% |
| 95-99% | +1.9% | +4.3% | +6.7% | +2.9% | +7.0% |
| top 1% | +1.2% | +3.8% | +5.6% | +2.9% | +7.0% |

The model **loses on the smallest moves** (it adds a correction where the value stays put) and **gains on large
moves**, and the gain grows with horizon. All of the overall gain comes from the larger moves, which is where
the forecastable structure is.

### 18.2 But it forecasts almost none of the move
Damping (slope of forecast change on true change; 1 = full move forecast, 0 = persistence) is only
0.01 (Abilene 5 min), 0.03 (30 min), 0.05 (60 min), and 0.02 / 0.05 on GÉANT, the same for big moves.
So the CNN-LSTM is persistence plus a very small correction, and the 1-3% gains are real but small because
5- and 15-minute moves are largely unpredictable from the past hour (consistent with section 11.3: the
correlation with earlier values is weak).

### 18.3 Peaks
For targets above the flow's own training 95th percentile, forecasts sit below the true value (bias, as % of the
true peak): Abilene persistence -4.8% / -11.3% / -14.9% against CNN-LSTM -5.8% / -15.4% / -21.3%; GÉANT -1.9% / -4.3%
against -3.8% / -8.5%. Peak MAE is slightly **worse** than persistence's everywhere (for example 24.5% vs 23.7% of
the mean true peak at 30 minutes). So at peaks the model pulls forecasts back towards typical levels: the
over-smoothing is a peak effect (mean reversion), not a general one, and it is consistent with the link-level
result (more cautious than persistence, fewer false alarms, more missed events; sections 16-17).

### 18.4 One flow decides the Abilene ranking
LOSA to CHIN (flow 86; mean 284 Mbit/s, bursts to 7,085) carries 8.2% of persistence's total test error at 30
minutes. Without it (143 flows), MAE change vs persistence:

| | 5 min | 30 min | 60 min |
|---|---|---|---|
| CNN-LSTM, all 144 | +0.9% | +0.9% | +1.3% |
| CNN-LSTM, without flow 86 | +1.0% | +1.2% | +1.9% |
| LSTM-only, all 144 | +1.3% | **-11.6%** | **-11.9%** |
| LSTM-only, without flow 86 | +1.4% | +1.3% | +2.6% |

So the LSTM-only's 12% failure in section 15 is entirely this one flow, and on the other 143 flows LSTM-only is
**as good or slightly better** than the CNN-LSTM. The CNN-LSTM's value on Abilene is robustness to out-of-range
bursts, not accuracy. On GÉANT the CNN-LSTM was the most accurate model (section 17), so this is not a
general statement about the CNN.

### 18.5 Summary of the findings to carry into the report
1. Persistence is a strong baseline at 5-15 minutes; every model gains only about 0.3-1% there.
2. Gains grow with horizon (to 1-3% at 60 minutes) and come from large moves, not from smoothing noise.
3. The loss function mattered most: training on MAE in original units, plus a residual output, were what made
   the models beat persistence at all (sections 13-14).
4. Models fail at the extremes: peaks are under-forecast more than by persistence, and out-of-range inputs need a
   physical cap on the forecast (section 14.2).
5. Link-level early warning is not better than reactive on either network (sections 16-17).
6. Rankings between architectures are not stable across networks and are decided by single flows.

Files: `scripts/05_errors.py`; figures `05_move_bins.png`, `05_damping.png`, `05_burst.png`.

---

## 19. Phase 8: stronger baselines and significance (`06_baselines2.py`)

Same data and splits as before. Every choice (blend weight, ridge penalty, ARIMA order) is made on validation.
Neural networks: the saved Phase 4-6 models, forecast = mean of 3 seeds (as section 18).

### 19.1 New baselines
- **Seasonal** (same time yesterday / last week / mean of the last 7 days): 35-340% *worse* than persistence
  everywhere. Traffic at the same time on another day is a poor forecast on its own.
- **Blend** (last value + w x (seasonal - last value)): the 7-day mean is the best reference on Abilene, with w
  growing with horizon (0.07 at 5 min, 0.30 at 60 min). Validation +1.7 / +5.3 / +7.3% on Abilene, but test
  **-2.3 / -2.8 / -1.7%**. On GÉANT a small, real gain: +0.3% (15 min), +1.2% (60 min) on test.
- **Ridge** (linear model on all 12 steps of every flow): on Abilene validation **+4.9 / +3.7 / +4.4%**, better than
  the CNN-LSTM (+2.7 / +1.3 / +2.2%), but on test **-4.5 / -12.9 / -13.3%**. On GÉANT it cannot be fitted usefully
  (6,348 inputs, 6,384 training samples); the chosen penalty makes it equal to persistence.
- **Why the linear baselines fail on Abilene test:** the LOSA to CHIN bursts again (sections 15.2 and 18.4). Test
  inputs reach 1.28 in scaled units (training maximum 1.0) and linear models extrapolate. Without flow 86 the
  blend is **+1.9 / +5.7 / +7.9%** on test, better than the CNN-LSTM without flow 86 (+1.0 / +1.2 / +1.9%);
  ridge is +0.8 / -4.1 / -4.8%. So the CNN-LSTM's advantage on Abilene test is robustness to one out-of-range
  flow, not better forecasting of ordinary flows; at 30-60 minutes a simple seasonal blend beats it on the other 143.

### 19.2 ARIMA at longer horizons (5 busiest Abilene flows)
Multi-step forecasts from the Kalman filter (checked to equal statsmodels' own forecasts). At 5 minutes the result
is exactly Phase 2's (21.87 vs 21.28 Mbit/s, -2.8%). At 30 and 60 minutes ARIMA is **+7.4% and +12.6%** better
than persistence on these flows, while the CNN-LSTM is only +0.3% on the same flows. Per flow (test MAE, Mbit/s,
persistence / ARIMA / CNN-LSTM), 60 minutes: WASH-WASH 43.5 / 38.2 / 43.0; CHIN-LOSA 60.0 / 41.3 / 56.3;
LOSA-CHIN 90.0 / 85.0 / 93.8. **On the busiest flows at 30-60 minutes a per-flow ARIMA clearly beats the CNN-LSTM.**
The CNN-LSTM, trained on all 144 flows at once, gives the big flows' own dynamics (e.g. mean reversion after a
burst) little weight. This points to per-flow components or an ARIMA + network combination as a next step.

### 19.3 Significance (test; Diebold-Mariano with one day of Newey-West lags, 95% block bootstrap with one-day blocks)

| Case | CNN-LSTM vs persistence | CNN-LSTM vs best baseline |
|---|---|---|
| Abilene 5 min | +0.9% [+0.6, +1.1], p < 0.001 | vs ridge +5.1% [+0.1, +13.2], p = 0.20 |
| Abilene 30 min | +0.9% [+0.4, +1.3], p < 0.001 | vs blend +3.6% [-1.3, +9.9], p = 0.26 |
| Abilene 60 min | +1.3% [-0.2, +2.4], p = 0.04 (borderline) | vs blend +2.9% [-2.6, +9.7], p = 0.40 |
| GÉANT 15 min | +0.6% [+0.4, +0.9], p < 0.001 | vs blend +0.3% [+0.1, +0.6], p = 0.006 |
| GÉANT 60 min | +3.2% [+2.5, +3.8], p < 0.001 | vs blend +1.9% [+1.1, +2.7], p < 0.001 |

- The small gains over persistence are **real** (not chance) everywhere except Abilene 60 min, which is borderline.
- On GÉANT the CNN-LSTM also significantly beats the best simple baseline. On Abilene it beats the baselines on
  average but the difference is not significant: a few burst days decide it.
- LSTM-only and CNN-only behave the same way (full table in the results page).

### 19.4 Files added
`common.py`: `clean_timeline`, `lookback`, `seasonal_mean`, `ridge_path`, `dm_test`, `block_bootstrap_gain`,
`saved_model_names`, `data_dir` (+ self tests). `scripts/06_baselines2.py`; figures `06_baselines.png`,
`06_significance.png`; baseline forecasts in `processed/phase8/`.

---

## 20. Phase 9: better inputs (`07_features.py`, trained on the GPU)

### 20.1 Setup
The CNN-LSTM (residual, original-units MAE loss, Phase 4 settings, 3 seeds) was given, one at a time:
- **time**: sin/cos of the time of day and day of week of the target;
- **seas**: every flow's seasonal value (mean of the same time on the last 7 days, the best reference in
  Phase 8) at the target time and at the last input time, minus the last value. Each flow gets its own two
  learned weights for these, added to the network's correction (a learned per-flow version of Phase 8's blend).
  A first version fed them into the dense layer with all flows mixed; it overfitted and trained unstably in a
  short test, so it was replaced before any results were taken;
- **time + seas**; **long**: a 3-hour window on Abilene (36 steps), 6 hours on GÉANT (24 steps);
- **long + best extra**: only if both beat the base by more than 0.5% on validation (happened only at Abilene 60 min).

**GPU.** Training moved to the laptop's RTX 4050 through WSL2 (Ubuntu 24.04, TensorFlow 2.21 with CUDA, the same
library versions as `.venv`; `scripts/wsl/run.sh` runs any script there). GPU and CPU training give slightly
different numbers, so **the base was retrained on the GPU** and every Phase 9 comparison is GPU vs GPU (`p9g_*`
models). The GPU base reproduces the CPU results (e.g. Abilene 5 min test +1.1% vs +0.9%; GÉANT 60 min +3.1% vs
+3.2%). WSL is limited to 8 GB of memory with 4 GB swap (`.wslconfig` in the Windows user folder); three training jobs at once
ran out of memory, two at a time works. Earlier CPU-trained Phase 9 models (`p9_*`) are not used.

A longer window cannot be built just after a data gap, so all methods are scored on the 99.1% of Abilene test
targets it keeps (100% on GÉANT). Selection on validation only.

### 20.2 Results (test MAE change vs persistence, 3-seed mean forecast)

| Input set | Abilene 5 min | Abilene 30 min | Abilene 60 min | GÉANT 15 min | GÉANT 60 min |
|---|---|---|---|---|---|
| base (window 12) | +1.1% | +0.8% | +1.2% | +0.7% | +3.1% |
| + time | +0.4% | +0.6% | +1.5% | +0.5% | +3.1% |
| **+ seasonal** | **+7.1%** | +0.1% | +2.9% | -0.4% | +4.1% |
| + time + seasonal | +7.1% | +2.8% | +2.9% | -0.4% | +4.6% |
| longer window | +0.5% | +0.9% | +1.8% | +0.7% | +2.9% |
| longer window + time + seasonal | – | – | +3.4% | – | – |
| *validation, best set* | *+9.7% (seas)* | *+11.0% (seas)* | *+13.6% (long+time+seas)* | *+1.0% (long)* | *+3.3% (long)* |

### 20.3 What this means
- **Seasonal values are the input that matters on Abilene.** On validation they raise the gain over persistence
  from about 1-2% to **10-14%**. On test the chosen set is **significantly better than the base at 5 min (+6.1%,
  95% interval [+4.8, +7.0], p < 0.001)**, and beats persistence by +7.1% and the best Phase 8 baseline by +11%.
- **At 30 and 60 min the test result is decided by LOSA to CHIN (flow 86) again.** Without it the seasonal models
  beat persistence by **+8.2 / +9.6 / +12.1%** at 5 / 30 / 60 min (base: +1.2 / +1.2 / +1.9%) and beat the base on
  113-130 of 144 flows. On flow 86 alone they are 75-106% worse than persistence (base: 3-5% worse): during its
  out-of-range bursts the seasonal term pulls the forecast back towards normal levels. With flow 86 included, the
  30 and 60 min gains (+0.1% to +3.4%) are not significant (intervals about [-18%, +12%]).
- **Checked: no look-ahead.** The seasonal input of a test sample was rebuilt by hand from the raw data; the newest
  value it uses is one day (288 steps) before the target (`scripts/wsl/verify_p9.py`).
- **Time of day and a longer window add almost nothing** on either network (within about ±0.6%): the 12-step window
  and the seasonal values already carry what they could add.
- **GÉANT:** nothing beats the base clearly on validation, so the rule picks the 6-hour window (by 0.1-0.2%), which is
  not better on test (15 min +0.1%, p = 0.07; 60 min -0.3%, p = 0.01). Time + seasonal is +4.6% vs +3.1% on GÉANT
  60 min test, but validation did not show it, so it cannot be claimed. GÉANT has about a quarter of Abilene's
  training data and 3.7x the flows, so per-flow seasonal weights are harder to learn.
- **For Phase 10:** the main remaining weakness is one bursty flow outside the training range, and the seasonal input
  makes it worse. Robust handling of out-of-range inputs (clipping, a peak-aware loss, or switching the seasonal term
  off when a flow is far outside its normal range) should keep the 8-12% gain on the other flows without the flow 86 loss.

### 20.4 Files added
`common.py`: `time_features`, `segment_ids`, `context_dataset`, `model_inputs`; the model builders take `n_context`
and `seasonal` (per-flow `DepthwiseConv1D` blend, initialised at zero so every model still starts as persistence).
`scripts/07_features.py`; `scripts/wsl/` (GPU helpers, `verify_p9.py`); figure `07_features.png`; 78 models `p9g_*`.

---

## 21. Phase 10: peaks and bursts (`08_peaks.py`, GPU)

### 21.1 Diagnosis: why the seasonal model loses on LOSA to CHIN
Phase 9's seasonal model is 75-106% worse than persistence on flow 86 at 30-60 min. Splitting its extra error:
- steps whose input is above the training maximum (0.5% of the flow's test steps): only **10-14%** of the extra error;
- steps where the flow is more than 4 training standard deviations away from its seasonal level (2.5% of its
  steps): **91-100%** of the extra error.
So the problem is not out-of-range inputs as such but **bursts**: when a flow is far from its usual level for that
time of day, the seasonal term pulls the forecast back to normal, while the burst persists.
**Note:** this diagnosis looked at test results (it compared simple fallback rules on test). To keep Phase 10
honest, nothing below was tuned on test: the rule's threshold is chosen on validation by a rule fixed in advance,
and the learned gate is trained on the training data only.

### 21.2 What was tried
- **Fallback rule** (no training): use persistence for a flow at a step when |seasonal - last| is more than k
  training standard deviations of that flow; k from 3, 4, 5, 6, 8 chosen on validation (largest gain; within 0.05%
  of the best the smaller, more cautious k wins). Chosen: k = 5 on Abilene (applied to 0.3% of forecasts), k = 3 on GÉANT.
- **Learned gate**: the seasonal term is multiplied by sigmoid(a - b z), z = |seasonal - last| / sd, with a and b
  learned from the training data (2 numbers shared by all flows), starting almost open. 15 new models (`p10g_*`).

### 21.3 Results (test MAE change vs persistence, 3-seed mean forecast)

| | Abilene 5 min | Abilene 30 min | Abilene 60 min | GÉANT 15 min | GÉANT 60 min |
|---|---|---|---|---|---|
| base (no seasonal) | +1.1% | +0.8% | +1.1% | +0.7% | +3.1% |
| seasonal (Phase 9) | +7.1% | +0.1% | +2.9% | -0.4% | +4.1% |
| **seasonal + fallback rule** | **+7.7%** | **+8.1%** | **+10.8%** | +0.6% | +4.6% |
| seasonal + learned gate | +6.9% | +1.2% | +2.4% | -0.4% | +4.3% |
| flow 86 only: seasonal / rule | -10% / -1% | -106% / -9% | -79% / 0% | | |
| peak bias: persistence / base / rule | -4.8 / -5.7 / -6.5% | -11.3 / -15.4 / -19.5% | -14.9 / -21.6 / -24.6% | | |

Significance (Diebold-Mariano, 95% block bootstrap): the **rule beats the base CNN-LSTM by +6.7 / +7.3 / +9.8% on
Abilene (all p < 0.001, intervals [+6.2, +7.3], [+5.9, +8.5], [+8.3, +11.1])** and by +1.5% on GÉANT 60 min (p < 0.001).
Against the Phase 9 seasonal model the rule's gain is not significant (p = 0.22-0.26): it comes from a few burst days
of one flow, which a day-block bootstrap cannot pin down.

### 21.4 What this means
- **With a fallback rule, the seasonal CNN-LSTM is 8-11% better than persistence on Abilene at 5-60 minutes,** about
  7-10 times the Phase 4-6 gain, and significantly better than the model without seasonal inputs. The rule touches
  only 0.3% of forecasts.
- **The learned gate did not work, and the reason is a finding in itself:** it learned to stay open (it would only
  half-close at 58-82 standard deviations). In the training period, pulling a far-off flow back to its seasonal level
  *helps*, because training bursts are short and revert. The long LOSA to CHIN bursts of the test period are a new
  kind of behaviour with no examples in training, so no model trained on that data can learn to handle them. A fixed
  rule encodes the prior "trust persistence in a burst" that the data cannot teach.
- **Caveats (important for the report):** (1) the rule's form came from a diagnosis that looked at test; (2) validation
  cannot tell k = 5 from k = 6 or 8 (all within 0.05%), but on test k = 6 would have failed at 30 min (+0.1%), so the
  result rests on a threshold validation could not really choose and on one flow's bursts in one test period.
  Phase 12's rolling-origin evaluation (other test periods) is the way to check whether it holds.
- **Peaks are still under-forecast** more than by persistence (e.g. -24.6% vs -14.9% bias at 60 min): the rule removes
  the worst burst errors but the model still pulls ordinary peaks towards typical levels.
- **GÉANT:** seasonal inputs still do not help clearly; the rule (k = 3) gives +4.6% vs +3.1% for the base at 60 min.
- **Not tried in Phase 10:** a peak-weighted loss and an ARIMA + network combination (section 19.2). Quantile
  (90th-percentile) forecasts move to Phase 11, where the early-warning work uses them.

### 21.5 Files added
`common.py`: `_seasonal_gate_class` (the gate layer, saved with the model), `build_cnn_lstm(gate_sd=)`, `load_saved`
(loads a model with the project's custom layers). `scripts/08_peaks.py`; figure `08_peaks.png`; 15 models `p10g_*`.

---

## 22. Phase 11: early warning, redone (`09_warnings.py`)

### 22.1 Method
Same events and scores as Phase 6 (Abilene: link utilisation at or above 50%, with 40% and 60% as sensitivity;
GÉANT: link load in the top 10% of that link's training loads, approximate routing). Three forecasters, all mean of
3 GPU seeds: persistence, the CNN-LSTM base, and Phase 10's seasonal model with the fallback rule (same k as Phase 10).
Each warns on its **point** forecast and on a **high estimate**: forecast + a per-link margin equal to the 90th
percentile of that link's validation errors (split-conformal calibration; level fixed in advance). This replaces the
planned quantile networks: it gives a calibrated 90th-percentile forecast for every method, persistence included,
without new training.

### 22.2 Results (test, F1 at the primary threshold; point -> q90 high)

| | persistence | CNN-LSTM base | seasonal + rule |
|---|---|---|---|
| Abilene 5 min | 0.97 -> 0.97 | 0.97 -> 0.97 | 0.97 -> 0.97 |
| Abilene 30 min | 0.93 -> 0.93 | 0.94 -> 0.94 | 0.91 -> 0.91 |
| Abilene 60 min | 0.90 -> 0.90 | 0.91 -> 0.91 | 0.89 -> 0.89 |
| GÉANT 15 min | 0.84 -> 0.78 | 0.83 -> 0.78 | 0.83 -> 0.79 |
| GÉANT 60 min | 0.76 -> 0.68 | 0.73 -> 0.67 | 0.74 -> 0.67 |

Link-load MAE, test (persistence / base / seasonal + rule): Abilene 0.203 / 0.203 / 0.195 pp at 5 min, 0.381 / 0.386 /
0.372 at 30 min, 0.494 / 0.503 / **0.453** at 60 min; GÉANT 9.13 / 9.33 / 9.32 and 15.96 / 16.12 / 15.91 Mbit/s.
Persistence's numbers equal Phase 6's exactly.

### 22.3 What this means
- **Better forecasts do not give better warnings.** The seasonal model forecasts link loads up to 8% more accurately
  (Abilene 60 min), but its warnings are no better than persistence anywhere; no Abilene F1 difference is clear in a
  day-block bootstrap (e.g. 30 min: 0.91 vs 0.93, interval [-0.26, +0.04]). At 30-60 min it misses more congestion
  (recall 0.85 vs 0.93): during LOSA to CHIN's episodes (which are nearly all the congestion, section 16.4) the
  seasonal term still pulls flow 86 down whenever it is within 5 sd of its usual level.
- **Warning on a high estimate does not change Abilene** (the margins are a fraction of a percentage point, far below
  what moves a link across 50%; even the 99% level changes F1 by at most 0.01). On GÉANT it trades precision for
  recall (recall 0.84 -> 0.95 at 15 min, onsets warned 0 -> about 1,450 of 2,130), but equally for persistence, so the
  model gains nothing over it.
- **Calibration check:** the true load is below the high estimate on 90-91% of GÉANT test link-steps (calibrated) but
  only 82-87% on Abilene (the test period is burstier than validation, as everywhere in this project).
- **Conclusion, confirming Phase 6 with stronger evidence:** on both networks, congestion here is a property of a few
  long-lasting bursts whose starts are not predictable from the past hour or the past week. Once an episode has started,
  reacting to the current load is already almost as good as any forecast. Forecasting improves the average error
  (Phases 9-10) but not the decisions that matter for congestion.

Files: `scripts/09_warnings.py`; figure `09_warnings.png`.

---

## 23. Phase 12: robustness (`10_robustness.py`, GPU)

### 23.1 Rolling origin: does it hold in other test periods?
The whole comparison was repeated on three earlier periods, each with its own chronological split (train =
everything before validation), its own scaler, its own rule threshold (chosen on that fold's validation as in
Phase 10) and 3 seeds per model (36 new models, `p12g_*_f*`). At 60 minutes, where the effects were largest.
Test MAE change vs persistence:

| Fold | Test period | base | seasonal | seasonal + rule (k) | rule vs base, 95% interval |
|---|---|---|---|---|---|
| Abilene original | 11 Aug - 10 Sep | +1.1% | +2.9% | **+10.8%** (5) | +9.8% [+8.3, +11.0], p < 0.001 |
| Abilene A | 14 Jun - 6 Jul | +0.4% | +9.9% | **+8.7%** (8) | +8.3% [+7.4, +9.1], p < 0.001 |
| Abilene B | 6 Jul - 28 Jul | +1.5% | +12.0% | **+12.0%** (8) | +10.7% [+9.0, +12.1], p < 0.001 |
| Abilene C | 28 Jul - 11 Aug | +2.5% | +13.1% | **+13.1%** (6) | +10.9% [+9.4, +12.6], p < 0.001 |
| GÉANT original | 7 Apr - 29 Apr | +3.1% | +4.1% | +4.6% (3) | +1.5% [+1.0, +2.0], p < 0.001 |
| GÉANT A | 15 Feb - 9 Mar | -1.0% | -3.2% | -1.9% (5) | -0.8% [-4.4, +1.8], p = 0.59 |
| GÉANT B | 9 Mar - 27 Mar | -6.1% | -3.9% | -2.9% (3) | +3.0% [+1.6, +4.4], p < 0.001 |
| GÉANT C | 27 Mar - 7 Apr | +3.2% | +4.6% | +4.4% (3) | +1.2% [+0.8, +2.2], p < 0.001 |

- **The Abilene result is robust.** Seasonal inputs give a 9-13% gain over persistence in every period, and the
  seasonal model with the rule beats the CNN-LSTM without seasonal inputs by 8-11% in all four, each p < 0.001.
- **The flow 86 problem was specific to the original test period.** In the earlier periods the seasonal model works
  without the rule (+9.9 / +12.0 / +13.1%); validation picks a lax threshold (k = 6-8) and the rule changes little,
  costing 1.2% in fold A. So the rule behaves as insurance: a small cost in calm periods, a large saving (2.9% -> 10.8%)
  in the period with long bursts. This answers the caveat of section 21.4: the rule's benefit does not rest on one
  period, although its threshold is still not well determined by validation.
- **The CNN-LSTM without seasonal inputs gains only 0.4-2.5% on Abilene in any period**, confirming Phases 4-6.
- **GÉANT is not robust.** With less training data (folds A and B train on 30-45% of an already short series, about
  5-7 weeks), every neural model is worse than persistence in two of four periods (base -6.1% in fold B). The rule
  helps relative to the base in three of four periods, but the overall GÉANT gains of Phases 6-11 do not carry to
  every period. GÉANT needs more training data than it has for these models to be reliable.

### 23.2 Ablations with 3 seeds (Abilene, test, change vs persistence; GPU)

| Model | 5 min | 30 min | 60 min |
|---|---|---|---|
| full CNN-LSTM | +1.08% | +0.82% | +1.12% |
| 1 conv layer | +0.33% | +0.77% | +1.20% |
| no pooling | +0.60% | +0.75% | +1.18% |
| 1 LSTM layer | +0.82% | +0.86% | +1.31% |

Seed spread is about 0.3-0.5%. Only one difference stands out: one conv layer is worse at 5 minutes (+0.33% vs +1.08%,
every seed lower). Otherwise removing a part changes nothing beyond noise. This confirms section 15.3 with 3 seeds:
the architecture details matter far less than the inputs (Phase 9) and the loss (Phase 4).

### 23.3 Error analysis rerun (as section 18)
MAE change vs persistence by size of the true move (Abilene 60 min): base -15.1 / +2.6 / +4.5 / +7.3 / +5.8% from the
smallest 50% of moves to the top 1%; **seasonal + rule -29.3 / +16.9 / +19.9 / +19.8 / +21.0%**. Damping (share of the
move that is forecast): **0.006 -> 0.047 at 5 min, 0.029 -> 0.149 at 30 min, 0.051 -> 0.168 at 60 min**.
- The seasonal model forecasts 15-17% of the move at 30-60 minutes instead of 3-5%: it now predicts a real part of where
  traffic is going (towards its usual level for that time), not just "slightly different from now".
- The price is larger on the smallest moves (it predicts a change where the value stays put), but the gains on the
  50% of larger moves are 3-4 times bigger than before.
- On GÉANT the picture barely changes (damping 0.049 -> 0.061 at 60 min), as expected from Phases 9-10.

Files: `common.py` `fold_dataset`; `scripts/10_robustness.py`; figure `10_rolling_origin.png`; 63 models `p12g_*`.

---

## 24. Overall findings after Phases 8-12 (for the report)

1. **Persistence is a strong baseline**, and architecture alone barely beats it: the CNN-LSTM, LSTM-only and CNN-only
   gain 0.3-2.5% on Abilene in every test period, significant but small (sections 14-15, 19, 23).
2. **What the model sees matters far more than its architecture.** Each flow's seasonal level (mean of the same time
   on the last 7 days) raises the Abilene gain to **9-13% over persistence at 60 minutes in all four test periods**, and
   the model then forecasts 15-17% of each move instead of 3-5% (sections 20, 23). Ablations change results by under 1%.
3. **Seasonal knowledge is wrong during long bursts**, which pull the forecast back to normal. A fixed rule (use
   persistence when a flow is more than about 5 standard deviations from its seasonal level) removes this loss where it
   matters and costs little elsewhere. A learned gate cannot learn it, because training periods contain no long bursts
   (sections 21, 23).
4. **The loss function made the models work at all**: residual output plus MAE in original units (sections 13-14).
5. **Better forecasts do not give better congestion warnings.** Congestion is a few long bursts whose onsets are not
   predictable; once started, reacting to the current load is as good as forecasting, with or without calibrated
   safety margins (sections 16-17, 22).
6. **GÉANT results are fragile**: small gains in some periods, losses in others, because its series is short for these
   models (sections 17, 23).
7. **Method notes for the viva:** every choice was made on validation; results are 3-seed means with Diebold-Mariano and
   day-block bootstrap tests; Phase 10's rule was motivated by a diagnosis that looked at test, which Phase 12's
   rolling-origin evaluation then checked on three other periods.
