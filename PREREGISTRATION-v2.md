# PREREGISTRATION v2

Registered 2026-10-20 and committed **before any pre-2008 data or any new ticker was downloaded**. v1 (`PREREGISTRATION.md`, tag `v1.0`) is unchanged: its sealed test (2023-02 to 2026-09) gave Sharpe 0.25, Deflated Sharpe probability 0.60, 78 round trips, verdict WATCH. That test set is spent and is never used for selection or tuning again.
Machine-readable copy: `config/v2.json`. Contract: `Financedashboard/research/strategy-validation-plan-2026-10-18.md`, applied as written except the one departure listed in section 2.

## 1. Lineage and honesty
- v2 is chosen using 2008-07 to 2026-09, which includes the v1 test period. That whole range is therefore **development** data, not a test.
- The only test is a **pre-sample holdout**, 2002-09-01 to 2008-06-30. v1 never computed a return before 2008-07-01, and no v2 data has been downloaded yet.
- A good holdout number would partly reflect 2002-2008 being a favourable era for trend following. A poor one would not prove the idea is dead: published SG Trend Index results show a 20.4% drawdown from May 2024 to May 2025. Either is reported as it comes out.

## 2. Departure from the contract (stated, not hidden)
Contract line 60 puts the sealed test as the *final* 20% of history. That slice is already spent, so v2 uses an earlier block instead: [holdout 2002-09 to 2008-06] then [train, validation 2008-07 onward]. Every holdout number is labelled "pre-sample holdout". The contract-compliant check is prospective (contract line 66): `scripts/07_forward_check.py` logs a paper trade of the frozen rule from the freeze date; no forward numbers are claimed until months have passed.

## 3. Split
- **Holdout (sealed):** 2002-09-01 to 2008-06-30.
- **Development:** 2008-07-01 to the last complete month before download, chronological 75% train / 25% validation by session count, boundaries snapped to month-end, computed from the calendar only.
- Warm-up prices before a window start are used for signals only. No return or statistic is computed on dates outside the window being evaluated (positions are zero before the window's first fill).

## 4. Universe, point-in-time
- **U16:** the v1 ETFs: SPY, EFA, EEM, IWM, QQQ, EWJ, TLT, IEF, LQD, TIP, GLD, SLV, DBC, VNQ, UUP, FXE.
- **U24:** U16 plus SHY, EWG, EWZ, FXY, FXB, FXA, FXC, DBA.
- An asset becomes tradable only once it has 273 sessions (13 months) of price history, and at least `vol window + 1` sessions. Before that its weight is 0 and it is excluded from covariance. N in the sizing formula is the number of tradable assets on that date.
- **Calendar:** SPY's sessions. Gaps in another ETF of up to 2 sessions are forward-filled (stale price, return 0) and logged; a ticker with a longer gap or more than 5 gap days in total is excluded and the exclusion is logged.

## 5. Cash
FRED `DTB3` (3-month T-bill), daily rate `DTB3[t-1] / 100 / 252` using the previous available observation (forward-filled across holidays), the same series for holdout and development. Replaces v1's BIL, which does not exist before 2007.

## 6. Candidate designs (12, all registered, all logged)
signal {S1, S2, S3} x universe {U16, U24} x vol window {63, 126 sessions}.
- **S1:** v1 signal, `sign(P[t-21] / P[t-252] - 1)`.
- **S2:** mean of `sign(P[t] / P[t-21L] - 1)` over L = 1, 3, 12 months (no skip). Values in {-1, -1/3, 1/3, 1}.
- **S3:** mean of `sign(P[t-21] / P[t-21L] - 1)` over L = 3, 6, 12 months (skip the last month).
- **Common to all:** raw weight `w_i = s_i * (target / vol_i) / N`, vol_i annualised from the vol window, then the portfolio is scaled to a 10% ex-ante vol target using that window's covariance, gross cap 2x, long/short, monthly rebalance, signal at close t / fill at close t+1, costs 5 bps commission + 5 bps slippage on turnover, borrow 25 bps a year on shorts.
- **Selection rule:** the design with the highest validation Sharpe. Nothing else decides it.

## 7. Trial count (fixed in advance)
N = 49 = 22 (v1 trials) + 12 designs + 12 sensitivity cells (3 signals x vol targets 6, 8, 12, 15% on the chosen universe and vol window) + 2 cost-stress runs + 1 (v1's primary rule on the holdout). Deflated Sharpe uses `max(49, unique ledger combinations)`. Trial-Sharpe variance is taken from the union of the v1 grid's and the 12 designs' daily development Sharpes (the Financedashboard `previous_sharpes` approach).

## 8. Report and verdict
Headline numbers come from the holdout only, for the selected design at 10 bps: annualised return, volatility, Sharpe, Deflated Sharpe, Sortino, max drawdown, Calmar, hit rate, round trips, turnover, profit factor; stationary bootstrap 95% CI for the Sharpe (block 21, 10,000 resamples, seed 20261020); benchmarks SPY and 60/40 SPY/IEF on the same dates and costs; cost stress at 2x and 3x; the sensitivity heatmap (signal x vol target); crisis windows that fall in the holdout or development; v1's primary rule on the same holdout as a comparison.
Overfit flags and gates are unchanged from v1 (Sharpe above 2, Deflated Sharpe probability under 0.95, hinges on one setting, fewer than 100 round trips, fewer than 2 full years, sparse regime cells, profit factor below 1, CI including zero). Verdict by contract lines 109-113. A 5.8-year holdout with a small early universe may fall short of 100 round trips, which would keep the verdict at WATCH.

## 9. Test lock
`09_dev_v2.py` computes nothing on holdout dates. `10_freeze_v2.py` records the config hash, v2 code hash, selected design and trial count, needs a clean git tree and green tests. `11_run_holdout_v2.py` refuses without a matching freeze, runs once, and any rerun must reproduce identical numbers. A change to v2 rules after the freeze is a new version (v3), not an edit.
