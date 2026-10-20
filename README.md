# systematic-trend-backtest

Cross-asset time-series momentum with volatility targeting on 16 liquid ETFs, validated under a pre-registered contract
(60/20/20 chronological split, 100+ out-of-sample round trips, Deflated Sharpe, cost stress). Rules were written in
`PREREGISTRATION.md` and committed before any data was downloaded. The sealed test was opened once.

<!-- summary:start -->
## Summary across versions (all net of costs, each test opened once)
| Version | Test window | Years | Sharpe | 95% CI | Deflated Sharpe | Round trips | Verdict |
|---|---|---|---|---|---|---|---|
| v1 | 2023-02-01 to 2026-09-30 (final 20%) | 3.6 | 0.25 | -0.52 to 1.09 | 0.60 | 78 | WATCH |
| v2 | 2002-09-01 to 2008-06-30 (pre-sample) | 5.8 | 0.65 | -0.13 to 1.46 | 0.84 | 53 | WATCH |
| v3 | 1988-01-04 to 2002-08-30 (pre-sample) | 14.7 | 0.59 | 0.05 to 1.15 | 0.91 | 208 | WATCH |

The strongest evidence is v3: the rule frozen in v2, unchanged, on 14.7 never-seen years. Its Sharpe CI excludes zero, but it is still WATCH:
every joint regime cell has enough trades, and the Deflated Sharpe (0.91) is below 0.95. Nothing was tuned on any test set.
<!-- summary:end -->

## v1 result (sealed out-of-sample, 2023-02-01 to 2026-09-30, 10 bps costs)
**Verdict: WATCH.** The strategy did not earn a statistically distinguishable edge, and it lagged both benchmarks.

| | Strategy | SPY | 60/40 |
|---|---|---|---|
| Annualised return | 6.7% | 20.4% | 12.2% |
| Volatility | 9.7% | 14.9% | 9.4% |
| Sharpe | 0.25 | 1.02 | 0.79 |
| Max drawdown | -12.3% | -18.8% | -10.6% |

- Sharpe 95% bootstrap CI: -0.52 to 1.09 (includes zero). Deflated Sharpe probability: 0.60 against 22 registered trials (needs 0.95).
- Sortino 0.35, Calmar 0.54, hit rate 35.9%, 78 round trips, turnover 7.64x equity a year.
- Costs: Sharpe 0.25 / 0.17 / 0.10 at 1x / 2x / 3x.
- Cash (BIL) earned 4.6% a year over the window, so most of the strategy's return is collateral yield, not trend profit.
- Flags: Deflated Sharpe probability 0.60 < 0.95; only 78 OOS round trips < 100.
- Failed gates: profit factor > 1 at 1x costs; profit factor > 1 at 2x costs; bootstrap 95% CI for Sharpe above 0; >= 100 OOS round trips; every joint regime cell has enough trades.
- Crises (2008, 2020, 2022) fall in train/validation, not out-of-sample; see the tearsheet table.

Full tables and charts: `results/tearsheet.html`. Machine-readable: `results/results.json`.

## Secondary study: radar break rules
Do the intermarket radar's break rules (20-session divergence, correlation sign flip, ratio z beyond ±2) predict next-month
returns? Test segment, pooled across 11 pairs: 2 of 12 tests have a 95% CI excluding zero
(divergence|spy_ret, any_break|spy_ret), and 0 survive Benjamini-Hochberg at 5%. No evidence the rules predict returns. Pair coverage differs:
BAMLH0A0HYM2/SPY has only 19 usable months (FRED returned data only from 2023-10-23), and GC=F/DFII10 has 159
(real yields cross zero, so the ratio z-score is undefined and the original rule gives "no read" unless another rule fires; those months skew toward breaks).
Only the 11 pairs registered in advance were tested; pairs needing derived or unavailable legs (cracks, MATIF, TTF, JP2Y, crypto) were not.

## Method
See `PREREGISTRATION.md` (rules, split, trial accounting, verdict logic, Addendum 1). Test lock: `src/stb/seal.py` refuses to open the
test without `freeze.json` matching config and engine code on a clean git tree; reruns must reproduce identical numbers.
No lookahead: signals use data up to day t and fill at the close of t+1. Costs are 5 bps commission + 5 bps slippage per unit of turnover; borrow 25 bps a year on shorts.

## Data
yfinance daily adjusted closes (dividends and splits), downloaded 2026-10-20 to 2026-10-20 UTC; per-ticker source, URL, row counts and sha256 in `data/manifest.json`
(radar series in `data/manifest_radar.json`; FRED for DFII10 and BAMLH0A0HYM2). Raw CSVs are not committed; rerun `scripts/01_download.py` (results will drift slightly with Yahoo revisions).

## Reproduce
```
python -m venv .venv && .venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python -m pytest -q
.venv\Scripts\python scripts\01_download.py ; scripts\02a_split.py ; scripts\02_run_dev.py ; scripts\03_freeze.py
.venv\Scripts\python scripts\04_run_sealed_test.py    # once; later runs only verify reproduction
.venv\Scripts\python scripts\05_radar.py ; scripts\06_report.py
```

## Limits
- The sealed test is about 3.7 years of a strong equity bull market; the 100-round-trip gate was not met.
- Profit factor and hit rate are per round trip on unweighted asset returns net of round-trip costs; they ignore position size and cash yield.
- Yahoo adjusted closes are free-data quality. Borrow and slippage are flat assumptions. Long/short at up to 2x is not available in an ISA.
- Validation favoured 9-month lookbacks over the pre-registered 12 months; the primary was not changed, by design.

## v2: pre-sample holdout (new registration, same contract gates)
After v1's test set was spent, v2 was registered in `PREREGISTRATION-v2.md` (committed before any new data was downloaded) and judged on a
**never-seen pre-sample holdout, 2002-09-01 to 2008-06-30**. Changes from v1: point-in-time universe widened to 24 ETFs, FRED DTB3 as cash, and 12 registered
designs (signal x universe x vol window), the best on validation chosen by a fixed rule: **S1|u24|126**. Deflated Sharpe uses 49 trials (v1's 22 included).

**Verdict: WATCH.**

| Holdout, 10 bps costs | v2 strategy | SPY | 60/40 | v1 rule (S1/U16/63) |
|---|---|---|---|---|
| Annualised return | 8.4% | 7.9% | 7.1% | 8.3% |
| Volatility | 8.9% | 15.1% | 8.4% | 10.2% |
| Sharpe | 0.65 | 0.39 | 0.53 | 0.57 |
| Max drawdown | -14.1% | -17.8% | -7.7% | -14.0% |

- Sharpe 95% bootstrap CI: -0.13 to 1.46 (includes zero). Deflated Sharpe probability 0.84 (needs 0.95).
- 53 round trips (the contract needs 100), hit rate 45.3%, profit factor 1.59, turnover 4.13x a year. Cost stress Sharpe: 0.65 / 0.60 / 0.55 at 1x / 2x / 3x.
- Cash earned 2.8% a year over the window.
- Flags: Deflated Sharpe probability 0.84 < 0.95; only 50% of grid cells have positive validation Sharpe; result hinges on one setting (validation: primary beats neighbour median by 0.31); only 53 OOS round trips < 100.
- Failed gates: bootstrap 95% CI for Sharpe above 0; >= 100 OOS round trips; every joint regime cell has enough trades.
- What did not help: the 1/3/12-month and 3/6/12-month ensemble signals scored lower on validation than the single 12-1 signal, and they also scored lower on the holdout (see the sensitivity heatmap), so the literature-motivated change was not an improvement here. Widening the universe raised round trips but lowered train Sharpe.
- Caveats: the holdout is *earlier* than the development data (contract line 60 wants the final 20%), 2002-2008 was a favourable era for trend following, and the selection rule picked a design whose train Sharpe was low. A contract-compliant check is prospective: `scripts/07_forward_check.py` logs the frozen rule from the freeze date; no forward numbers exist yet.

Tearsheet: `results/tearsheet_v2.html`. Machine-readable: `results/v2_results.json`. v2 data: yfinance and FRED, downloaded 2026-10-20 UTC (`data/manifest_v2.json`).

## v3: frozen rule on a long-history proxy universe, 1988-2002 holdout
Registered in `PREREGISTRATION-v3.md` before any v3 data was downloaded. The v2 rule (12-1 sign, 126-session vol window, 10% target, gross cap 2x)
was applied **unchanged**: nothing was selected. Universe: 18 long-history proxies (Vanguard equity, bond and REIT funds; FRED spot FX as excess-return proxies).
Excluded by the registered rules: XAUUSD: download failed: ValueError: stooq returned no usable rows f; DCOILWTICO: gap of 8 sessions in a row (limit 2). So v3 has **no commodities**.

**Verdict: WATCH.**

| Holdout 1988-01-04 to 2002-08-30, 10 bps | v3 strategy | S&P 500 fund (VFINX) | 60/40 VFINX/VBMFX |
|---|---|---|---|
| Annualised return | 8.8% | 11.9% | 10.6% |
| Volatility | 6.2% | 16.1% | 10.0% |
| Sharpe | 0.59 | 0.46 | 0.56 |
| Max drawdown | -10.9% | -46.3% | -25.5% |
| Calmar | 0.81 | 0.26 | 0.42 |

- Sharpe 95% bootstrap CI: **0.05 to 1.15, above zero.** Deflated Sharpe probability 0.91 against 67 trials (needs 0.95, flagged).
- 208 round trips (gate passed), hit rate 47.6%, profit factor 2.61, Sortino 0.84, turnover 4.56x a year.
- Cost stress Sharpe 0.59 / 0.51 / 0.44 at 1x / 2x / 3x.
- Dot-com bust (2000-03-24 to 2002-08-30): strategy 19.1%, VFINX -38.2%.
- Realised vol 6.2% is under the 10% target because the 2x gross cap binds on low-vol bonds and FX. Cash earned 5.2% a year.
- Flags: Deflated Sharpe probability 0.91 < 0.95; only 50% of grid cells have positive validation Sharpe; result hinges on one setting (validation: primary beats neighbour median by 0.31). The selection flags are inherited from v2, where the rule was chosen.
- Failed gate: every joint regime cell has enough trades (trend-regime cells: trend/high 5, trend/low 15, range/high 92, range/low 96).
- Caveats: the holdout is pre-sample, not the contract's final 20%. Mutual funds can't really be shorted. FX proxies omit carry. 1990s trend-following results are public, so this is an independent-data check, not proof.

Tearsheet: `results/tearsheet_v3.html`. Machine-readable: `results/v3_results.json`. Data: yfinance and FRED, downloaded 2026-10-20 UTC (`data/manifest_v3.json`).
Reproduce v2/v3: `scripts/08`-`12` (v2), `scripts/13`-`17` (v3); the holdout scripts refuse to run without a matching freeze, and a rerun must reproduce identical numbers.
