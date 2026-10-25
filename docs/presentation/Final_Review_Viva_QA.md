# Final Review – Likely Viva Questions and Short Answers

CNN-LSTM Based Network Traffic Forecasting. Every number comes from `results/metrics.csv`,
`docs/PROJECT_LOG.md` or `Project_Explanation_Guide.pdf` (chapter 13 has 38 longer answers).

## Architecture and pipeline

**1. Describe the system in one sentence.**
It forecasts the traffic between every pair of backbone routers 5–60 minutes ahead with a residual
CNN-LSTM, maps those forecasts onto links with the routing matrix, and checks whether that warns of
congestion earlier than watching the current load.

**2. Walk through the pipeline.**
Read 24 weekly Abilene files → keep the measured realOD column → clean with rules R1–R4 (84.4% kept)
→ 9 gap-free segments → 12-step windows → chronological 60/20/20 split → log1p + per-flow min–max fitted
on train → train residual CNN-LSTM on MAE in Mbit/s → predict, clip, unscale → score against
persistence → link utilisation = A·flows / capacity → warnings → precision/recall/F1/onsets.

**3. Why forecast flows and not link loads directly?**
Flows are what the dataset measures, and flow forecasts map onto any routing (load = A × flows). The cost
is that link errors combine many flow errors.

**4. Why remove bad data instead of filling it?**
Filling would mean training and scoring on values that were never measured. Removal costs 15.6% of steps.

## Model and algorithms

**5. Layers and parameter count?**
Conv1D(64, k=3, causal) ×2 → MaxPooling(2) → LSTM(100) → LSTM(50) → Dropout(0.2) → Dense(144) correction,
added to the last input step. 143,608 parameters on Abilene, 237,163 on GÉANT.

**6. Why combine a CNN and an LSTM?**
Convolutions pick up short patterns across flows; the LSTM models how they evolve. In practice the CNN mainly
added robustness to out-of-range bursts on Abilene, and the hybrid was the best model on GÉANT at 60 minutes.

**7. What is "causal" padding?**
Each convolution output uses only the current and earlier steps, so nothing later in the window leaks in.

**8. What is the residual model?**
Forecast = last value + learned correction. The correction layer starts at zero, so an untrained model is
exactly persistence, and training only moves away from it where that lowers the error.

**9. Why a custom loss?**
On the log scale every flow counts equally, so the model optimised small flows that hardly matter in Mbit/s.
`mae_raw` undoes the scaling inside TensorFlow and trains on MAE in Mbit/s — the reported metric.
Validation MAE: residual + MSE 2.777, residual + MAE in Mbit/s 2.693, persistence 2.742.

**10. How did you prevent overfitting and leakage?**
Dropout 0.2, early stopping with best-weight restore, learning-rate halving; scaler fitted on train only;
chronological split; no window crosses a gap; no decision ever made on the test split.

**11. How were hyper-parameters tuned?**
One at a time (filters, kernel, LSTM units, dropout, learning rate) on validation, keeping a change only if it
improved MAE by more than 0.5%. None did, so the planned settings stayed.

## Implementation decisions

**12. What was the forecast blow-up?**
One validation input exceeded the training range; the model output 1.99 in scaled units and the inverse
exponential turned it into about 25 million Mbit/s. `predict()` now clips forecasts to 0 … fastest link capacity.

**13. Why 50% and not 80% as the congestion threshold?**
No Abilene link reaches 80% (the busiest reaches 75%). 50% was fixed before seeing any warning result, with
40% and 60% as checks, and is documented as a deviation from the plan.

**14. Why three seeds?**
Gains are about 1% and the seed spread is 0.5–1%; without seeds a real gain cannot be told from luck.

## Results and testing

**15. How much better than persistence is the model?**
Abilene: +0.4% (5 min), +0.8% (30 min), +1.1% (60 min). GÉANT: +0.3% (15 min), +2.7% (60 min).
Every seed beats persistence in every case.

**16. Isn't a 0.4% gain just noise?**
At 5 minutes the seed spread is ±0.005 Mbit/s on 3.2 and all seeds win — consistent but tiny. The 60-minute
GÉANT gain (+2.7%, better on 459 of 529 flows) is clearer.

**17. Did the model give earlier congestion warnings?**
No. Abilene F1 at 50%: 0.97 / 0.94 / 0.91 vs persistence 0.97 / 0.93 / 0.90, and the same number of onsets
warned 30 minutes ahead (5 of 35). On GÉANT it gave fewer false alarms but missed more events.

**18. Why is persistence's F1 so high?**
Congestion episodes last many steps; persistence is right on every step except the first and last.

**19. Why did LSTM-only fail at 30/60 minutes on Abilene?**
Entirely one flow, LOSA→CHIN, whose test bursts exceed anything in training. Without it LSTM-only is as good
as the CNN-LSTM.

**20. If the model is nearly persistence, what did it learn?**
Small systematic corrections: it forecasts only 1–5% of each real change (damping 0.01–0.05), gains on large
moves (up to +7% at 60 min) and loses on small ones.

**21. How was the code tested?**
`self_test()` runs the data pipeline on synthetic data with assertions; `00_setup.py` checks the model has
exactly 143,608 parameters; `01_data.py` checks that undoing the scaling reproduces the raw data; ARIMA and
persistence are asserted to use identical truth values; an untrained residual model equals persistence.

## Novelty, limitations, future

**22. What is novel?**
Not the architecture. The strict leak-free evaluation against persistence on two networks with seeds; evidence
that output form and loss matter more than architecture; the end-to-end link-level test; and the diagnosis of
why flow gains do not become better warnings.

**23. Main limitations?**
Small gains; no early-warning advantage; only the last hour as input (no time-of-day); 2004–05 data; 3 seeds and
one split; GÉANT link level uses approximate routing; offline only.

**24. What would you do next?**
Seasonal baseline and time-of-day inputs, longer windows, more seeds with confidence intervals, probabilistic
forecasts for exceedance warnings, a third dataset, and simulating rerouting on warnings.

**25. Is it publishable or patentable?**
The evaluation findings could support a short paper after adding seeds and significance tests; acceptance is not
guaranteed. No patent claim — the CNN-LSTM architecture is not new.
