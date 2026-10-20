# PREREGISTRATION v3

Registered 2026-10-20 and committed **before any v3 data was downloaded**. v1 (`PREREGISTRATION.md`) and v2 (`PREREGISTRATION-v2.md`) are unchanged. Both ended WATCH, mainly because their tests were too short and too narrow to clear the contract's power gates (CI above zero, 100 round trips). Machine-readable copy: `config/v3.json`.

## 1. What v3 is
- The **already-frozen v2 rule, unchanged**: S1 signal `sign(P[t-21]/P[t-252]-1)`, 126-session vol window, 10% portfolio vol target, gross cap 2x, long/short, monthly rebalance, signal at close t / fill at close t+1, 5 bps commission + 5 bps slippage on turnover, 25 bps a year borrow on shorts.
- Applied to a **new long-history proxy universe** and judged on a **never-seen holdout**. Nothing is selected: development data is reported, never used to choose.

## 2. Windows
- **Holdout (sealed):** 1988-01-04 to 2002-08-30, about 14.7 years. It ends the day before v2's holdout begins; neither v1 nor v2 computed any return in it.
- **Development (report only, already seen):** 2002-09-03 to the last complete month before download.
- **Departure from contract line 60, stated:** the holdout is pre-sample, not the final 20%. 1990s trend-following results are public knowledge, so this is an independent-data check, not proof. Warm-up prices before a window are used for signals only; positions are zero before the window's first fill.

## 3. Universe (registered list, point-in-time)
An asset trades once it has 273 sessions of history (and at least 127). Before that its weight is 0.
- **Equity (Vanguard funds, yfinance adjusted close, distributions included):** VFINX, NAESX, VEURX, VPACX, VEIEX.
- **Bonds:** VFISX, VFITX, VUSTX, VWESX, VBMFX, VWEHX.
- **Real estate:** VGSIX.
- **Commodities, spot:** gold XAUUSD (Stooq), WTI `DCOILWTICO` (FRED).
- **FX vs USD, spot (FRED):** DEXUSUK, DEXUSAL, DEXUSEU (USD per unit as quoted); DEXJPUS, DEXCAUS, DEXSZUS (inverted to USD per unit).
- **Excess-return proxies:** spot commodities and FX carry no carry or roll in this data. They enter as synthetic indices `cumprod(1 + spot_return + rf)`, so the engine's excess return `r - rf` equals the spot move. The bias is stated: carry and roll are omitted.
- **Proxies, stated:** mutual funds cannot really be shorted and have trading limits; this is research, not a tradable book.
- **Drop rule:** a series that fails to download from every source, or has no data before the holdout end, is dropped and logged. No substitution after download.

## 4. Calendar, gaps, cash
- Calendar: VFINX sessions. Gaps of up to 2 sessions in a row are forward-filled (stale price, return 0), which covers FRED's federal-holiday gaps. A ticker with any longer gap after its first observation is excluded and logged.
- Cash: FRED DTB3, daily rate `DTB3[t-1] / 100 / 252`.

## 5. Report, trials, verdict
- **Headline (holdout only, 10 bps):** annualised return, volatility, Sharpe, Deflated Sharpe, Sortino, max drawdown, Calmar, hit rate, round trips, turnover, profit factor; stationary bootstrap 95% CI for the Sharpe (block 21, 10,000 resamples, seed 20261020); cost stress 2x/3x; benchmarks VFINX buy-and-hold and 60/40 VFINX/VBMFX rebalanced monthly on the same dates and costs; sensitivity heatmap (signals S1/S2/S3 from v2 x vol target 6/8/10/12/15%); holdout crisis windows 1998-08-01..1998-10-31 and 2000-03-24..2002-08-30; development crisis windows 2008, 2020, 2022.
- **Trial count, fixed now:** N = 67 = 49 (v1 + v2) + 1 (v3 rule) + 2 (cost stress) + 15 (sensitivity). The Deflated Sharpe uses `max(67, ledger)`. Trial-Sharpe variance: the union of the v1 and v2 development Sharpes.
- **Flags and gates:** unchanged from v1/v2, not loosened. Regimes on VFINX. Verdict by contract lines 109-113.

## 6. Test lock
`14_dev_v3.py` computes nothing on holdout dates. `15_freeze_v3.py` needs green tests and a clean tree, and records the config and code hashes. `16_run_holdout_v3.py` refuses without a matching freeze, runs once, and any rerun must reproduce identical numbers.
