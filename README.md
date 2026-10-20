# systematic-trend-backtest

Cross-asset time-series momentum with volatility targeting on 16 liquid ETFs, validated under a pre-registered contract
(60/20/20 chronological split, 100+ out-of-sample round trips, Deflated Sharpe, cost stress). Rules were written in
`PREREGISTRATION.md` and committed before any data was downloaded. The sealed test was opened once.

## Result (sealed out-of-sample, 2023-02-01 to 2026-09-30, 10 bps costs)
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
