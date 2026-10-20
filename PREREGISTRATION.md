# PREREGISTRATION

Written and committed **before any market data was downloaded or any return computed**. The git commit timestamp of this file is the proof. Contract: `Financedashboard/research/strategy-validation-plan-2026-10-18.md` §3 (lines 57–68), metrics (84–95), verdicts (109–113), line 152. It is applied as written, not loosened. A change to any rule after the freeze is a new version, not an edit.

Registered 2026-10-20. Machine-readable copy: `config/primary.json`.

## 1. Universe and data
- **Risk assets (16 ETFs):** SPY, EFA, EEM, IWM, QQQ, EWJ (equity); TLT, IEF, LQD, TIP (bonds); GLD, SLV, DBC (commodities); VNQ (real estate); UUP, FXE (FX). Cash proxy: BIL.
- **Source:** yfinance daily, `auto_adjust=True` (dividends and splits). Fallback: Stooq CSV if Yahoo returns 429 or gaps. The source per ticker, the download UTC time, the URL, the row count, first/last date and sha256 go in `data/manifest.json`.
- **Alignment:** intersection of sessions across all 17 tickers; no forward-fill.
- **Evaluation period:** 2008-07-01 (13-month warm-up) to the last complete month before the download.

## 2. Split (contract line 60)
Chronological by session count over the evaluation period: 60% train, 20% validation, 20% sealed test. Boundaries snap to month-ends. Dates are computed from the session calendar only and committed to `config/split.json` before any performance is computed. Signals may use history from before a boundary (data up to t). The test set stays sealed until the strategy and parameters are frozen on train and validation.

## 3. Primary strategy: cross-asset time-series momentum, volatility targeted
- **Signal:** on the last session of each month t, `mom = P[t-21] / P[t-21*L] - 1`, where P is the adjusted close, indices count sessions back from t, and L is the lookback in months (L=12: return from t-252 to t-21, i.e. 12 months excluding the last month). The sign of `mom` is the position direction; exactly 0 means flat.
- **Sizing:** `w_i = sign_i * (target_vol / vol_i) / N`, vol_i = 63-session realised daily vol * sqrt(252), data up to t. The portfolio is then scaled so ex-ante portfolio vol (63-session covariance) equals the target. Gross exposure cap 2.0x. Long/short.
- **Execution:** signal at the close of t, fill at the close of t+1. Weights are held to the next rebalance and drift with prices.
- **Returns:** `r_p = rf + sum w_i (r_i - rf) - costs - borrow`, rf = BIL daily return. Borrow 25 bps/yr on short notional.
- **Costs:** 5 bps commission + 5 bps slippage = 10 bps on |change in weight| per fill, including drift-correction trades. Stress: 2x (20 bps) and 3x (30 bps).
- **Frozen primary:** lookback 12 months skipping the last month, vol target 10%, monthly rebalance. It stays so regardless of validation results.
- **Round trip (contract line 59):** open from flat or the opposite sign, later close or reverse. A reversal closes one and opens another. Resizing is not a new trade. Fills and turnover are reported separately.

## 4. Trial accounting (contract line 62)
Grid: lookback {3, 6, 9, 12} months x vol target {6, 8, 10, 12, 15}% = 20 trials, used only for the sensitivity heatmap and the Deflated Sharpe trial count. Every run (grid, cost variants, reruns, failures) is appended to `trials/ledger.csv`. The Deflated Sharpe uses N = all unique registered trials. The radar study has its own ledger.

## 5. Benchmarks
Buy-and-hold SPY; 60/40 SPY/IEF rebalanced monthly. Same dates, fill convention and cost model.

## 6. Metrics (headline numbers: sealed OOS only)
Annualised return, volatility, Sharpe (252 sessions; excess over BIL), Deflated Sharpe (probability), Sortino, max drawdown, Calmar, hit rate, round trips, turnover, profit factor. Sharpe 95% CI: stationary block bootstrap of daily excess returns, mean block 21 sessions, 10,000 resamples, seed 20261020. Crisis windows: 2008 (Jul 2008–Mar 2009, partial due to warm-up), 2020 (Feb–Apr), 2022 (calendar year), each labelled with its segment. Under this split they fall in train/validation, not OOS; the report says so.

## 7. Overfitting flags and verdict
Automatic flags: OOS Sharpe > 2; Deflated Sharpe probability < 0.95; fewer than 60% of grid cells with positive validation Sharpe; primary Sharpe above the median of its adjacent grid cells by > 0.3 (hinges on one setting); fewer than 100 OOS round trips (forces WATCH, contract line 152).
Verdict per contract lines 109–113. PROMOTE needs: net OOS gain and PF > 1 at 1x and 2x costs, bootstrap CI for the Sharpe above 0, >= 100 OOS round trips, >= 2 complete OOS years, regime-cell coverage (>= 20 trades per joint cell; trend/range from the 63-session efficiency ratio at 0.30, high/low vol vs the median of 252 earlier estimates, on SPY) and no hard gate failing. Otherwise WATCH, or REJECT for verified leakage or negative net OOS economics under the cost model. A good Sharpe cannot offset a failed gate.

## 8. Test lock
`02_run_dev.py` slices data to the validation end before computing anything. `03_freeze.py` records the config hash, commit, trial count and validation summary, and requires a clean tree. `04_run_sealed_test.py` refuses to run without a freeze, with a hash mismatch or on a dirty tree, and runs once; reruns must reproduce identical numbers.
Report whatever comes out. Nothing is tuned on the test set.

## 8a. Addendum 1 (2026-10-20, written after data download and split, before any performance was computed)
- **Deflated Sharpe inputs:** the chosen strategy's daily excess returns on the sealed test window; trial Sharpe variance from the daily excess Sharpes of the 20 grid cells over train+validation combined (the selection sample, no test data); trial count N = unique (lookback, vol target, cost multiplier) combinations in `trials/ledger.csv` at freeze time, which covers the 20 grid cells plus the primary at 2x and 3x costs. Costs of 10 bps are the "1x" case.
- **Sealed-test run:** the engine runs on the full history and the OOS window is sliced out; trades are counted by exit date inside the window, and a position carried in counts as entered at the window start. Benchmarks buy at the close before the window starts, with the entry cost booked on the first window day.
- **Dev run:** `02_run_dev.py` drops all rows after the validation end before computing anything.

## 9. Secondary study: do the radar break rules predict next-period returns?
- **Rules** (ported unchanged from `Financedashboard/server/routine_radar.py:46-83`): 20-session extreme divergence (one leg breaks its prior-20 high/low, the sign-adjusted other leg does not); 60-session correlation sign flip vs the prior day, or a > 0.5 shift from the prior-252 mean; ratio z-score beyond +/-2 over 252 sessions. 313 aligned observations minimum, no forward-fill.
- **Pairs** (free daily legs only): SPY/TLT, RSP/SPY, QQQ/SPY, SOXX/SPY, SPY/^VIX, BZ=F/SPY, BZ=F/XLE, BZ=F/^VIX, ZW=F/DX-Y.NYB, GC=F/DFII10, BAMLH0A0HYM2/SPY. Pairs needing MATIF/TTF/JP2Y and crypto are excluded.
- **Test:** month-end state vs next-month outcome (non-overlapping). Outcomes: next-month return of SPY and of the left leg; next-month realised vol. Break vs hold, per rule type, pooled across pairs. Difference in means with a stationary-bootstrap 95% CI; Benjamini-Hochberg over all registered tests. Same 60/20/20 split; headline numbers OOS only. Reported separately.
