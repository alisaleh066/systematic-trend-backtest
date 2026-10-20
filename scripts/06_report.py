"""Build tearsheet.html, README.md and CV bullets from results.json. Every number is read from results, never typed."""
import base64
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

RES = ROOT / "results"
r = json.loads((RES / "results.json").read_text())
radar = json.loads((RES / "radar_results.json").read_text())
freeze = json.loads((ROOT / "freeze.json").read_text())
manifest = json.loads((ROOT / "data" / "manifest.json").read_text())
daily = pd.read_csv(RES / "oos_daily.csv", parse_dates=["date"], index_col="date")
h, ci, dsr = r["headline_oos_1x_costs"], r["sharpe_bootstrap_95ci"], r["deflated_sharpe"]
spy, sf = r["benchmarks_oos"]["spy"], r["benchmarks_oos"]["sixty_forty"]
pct = lambda v, d=1: "n/a" if v is None else f"{v * 100:.{d}f}%"  # noqa: E731
num = lambda v, d=2: "n/a" if v is None else f"{v:.{d}f}"  # noqa: E731
cash_ann = float((1 + daily["rf"]).prod() ** (252 / len(daily)) - 1)
dl = sorted(t["downloaded_utc"] for t in manifest["tickers"])


def png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# charts
eq = (1 + daily[["strategy", "spy", "sixty_forty"]]).cumprod()
fig, ax = plt.subplots(figsize=(8, 3.6))
for c, label in (("strategy", "Trend strategy (10% vol target)"), ("spy", "Buy-and-hold SPY"), ("sixty_forty", "60/40 SPY/IEF")):
    ax.plot(eq.index, eq[c], label=label, lw=1.6 if c == "strategy" else 1.1)
ax.set_title("Out-of-sample equity curve (growth of 1, net of costs)")
ax.legend(frameon=False, fontsize=8)
ax.grid(alpha=0.2)
equity_png = png(fig)

dd = eq / eq.cummax() - 1
fig, ax = plt.subplots(figsize=(8, 2.8))
for c in ("strategy", "spy", "sixty_forty"):
    ax.plot(dd.index, dd[c], lw=1.4 if c == "strategy" else 1.0, label=c)
ax.fill_between(dd.index, dd["strategy"], 0, alpha=0.15)
ax.set_title("Drawdown")
ax.legend(frameon=False, fontsize=8)
ax.grid(alpha=0.2)
dd_png = png(fig)


def heat(cells: dict, title: str) -> str:
    lbs = sorted({int(k.split("|")[0]) for k in cells})
    vols = sorted({float(k.split("|")[1]) for k in cells})
    z = np.array([[cells[f"{a}|{b}"] for b in vols] for a in lbs])
    fig, ax = plt.subplots(figsize=(5, 3.2))
    im = ax.imshow(z, cmap="RdYlGn", vmin=-1.5, vmax=1.5, aspect="auto")
    ax.set_xticks(range(len(vols)), [f"{v:.0%}" for v in vols])
    ax.set_yticks(range(len(lbs)), [f"{v}m" for v in lbs])
    ax.set_xlabel("vol target")
    ax.set_ylabel("lookback (skip last month)")
    ax.set_title(title, fontsize=10)
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            ax.text(j, i, f"{z[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im)
    return png(fig)


heat_oos = heat(r["grid_oos_sharpe"], "Sharpe, sealed test (sensitivity only)")
heat_val = heat(r["grid_validation_sharpe"], "Sharpe, validation")

monthly = (1 + daily["strategy"]).resample("ME").prod() - 1
tab = monthly.to_frame("r")
tab["y"], tab["m"] = tab.index.year, tab.index.month
piv = tab.pivot(index="y", columns="m", values="r")
piv["Year"] = (1 + daily["strategy"]).groupby(daily.index.year).prod() - 1
months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def cell(v):
    if pd.isna(v):
        return "<td></td>"
    c = "pos" if v > 0 else "neg"
    return f'<td class="{c}">{v * 100:.1f}</td>'


monthly_html = "<table><tr><th></th>" + "".join(f"<th>{m}</th>" for m in months) + "<th>Year</th></tr>" + "".join(
    f"<tr><th>{y}</th>" + "".join(cell(piv.loc[y].get(m)) for m in list(range(1, 13)) + ["Year"]) + "</tr>" for y in piv.index) + "</table>"


def stat_rows(rows: dict) -> str:
    keys = [("ann_return", "Ann. return", pct), ("ann_vol", "Ann. vol", pct), ("sharpe", "Sharpe", num), ("sortino", "Sortino", num),
            ("max_drawdown", "Max drawdown", pct), ("calmar", "Calmar", num), ("hit_rate", "Hit rate (round trips)", pct),
            ("round_trips", "Round trips", lambda v, d=0: "n/a" if v is None else str(v)), ("turnover_per_year", "Turnover / yr (x equity)", num),
            ("profit_factor", "Profit factor (round trips)", num)]
    head = "<tr><th></th>" + "".join(f"<th>{n}</th>" for n in rows) + "</tr>"
    body = "".join(f"<tr><th>{lab}</th>" + "".join(f"<td>{f(v.get(k))}</td>" for v in rows.values()) + "</tr>" for k, lab, f in keys)
    return f"<table>{head}{body}</table>"


headline_tbl = stat_rows({"Strategy": h, "SPY": spy, "60/40": sf})
cost_tbl = stat_rows({"1x (10 bps)": r["oos_cost_stress"]["1x"], "2x (20 bps)": r["oos_cost_stress"]["2x"], "3x (30 bps)": r["oos_cost_stress"]["3x"]})
crisis_rows = "".join(
    f"<tr><th>{k}</th><td>{v['window'][0]} to {v['window'][1]}</td><td>{v['segment']}</td><td>{pct(v['strategy']['total_return'])}</td><td>{pct(v['spy']['total_return'])}</td><td>{pct(v['sixty_forty']['total_return'])}</td><td>{pct(v['strategy']['max_drawdown'])}</td></tr>"
    for k, v in r["crisis_periods"].items())
cov = r["regime_coverage"]["trades"]
gates = "".join(f"<li>{'PASS' if ok else 'FAIL'}: {g}</li>" for g, ok in r["verdict"]["gates"].items())
flags = "".join(f"<li>{f}</li>" for f in r["flags"]) or "<li>none</li>"
rt = radar["segments"]["test"]["tests"]
radar_rows = "".join(
    f"<tr><th>{k.replace('|', ' → ')}</th><td>{v['n_break']}/{v['n_hold']}</td><td>{pct(v['diff'], 2)}</td><td>[{pct(v['ci95'][0], 2)}, {pct(v['ci95'][1], 2)}]</td><td>{'yes' if v['bh_significant_5pct'] else 'no'}</td></tr>"
    for k, v in rt.items() if v)

verdict = r["verdict"]["verdict"]
html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Systematic trend tearsheet</title><style>
:root{{--bg:#fff;--fg:#1a1a1a;--mut:#666;--line:#ddd;--pos:#1a7f37;--neg:#c62828}}
@media(prefers-color-scheme:dark){{:root{{--bg:#16181c;--fg:#e8e8e8;--mut:#9aa;--line:#333;--pos:#5fd38a;--neg:#ff7b72}}}}
body{{font:15px/1.5 system-ui,sans-serif;background:var(--bg);color:var(--fg);max-width:980px;margin:0 auto;padding:16px}}
table{{border-collapse:collapse;margin:8px 0;font-size:13px;display:block;overflow-x:auto}}td,th{{border:1px solid var(--line);padding:3px 8px;text-align:right}}th{{text-align:left}}
.pos{{color:var(--pos)}}.neg{{color:var(--neg)}}img{{max-width:100%;height:auto}}.mut{{color:var(--mut)}}
.badge{{display:inline-block;padding:2px 10px;border:2px solid var(--line);border-radius:6px;font-weight:700}}</style></head><body>
<h1>Cross-asset trend following: out-of-sample tearsheet</h1>
<p class="mut">Sealed test {r['oos_window'][0]} to {r['oos_window'][1]} · 16 ETFs, 12-1 momentum, 10% vol target, long/short, gross cap 2x · costs 10 bps per trade · config {freeze['config_sha256'][:10]}</p>
<p>Verdict under the pre-registered contract: <span class="badge">{verdict}</span></p>
<p>Sharpe {num(h['sharpe'])} (bootstrap 95% CI {num(ci[0])} to {num(ci[1])}); Deflated Sharpe probability {num(dsr['probability']) if dsr else 'n/a'} over {r['trial_count']} trials. SPY returned Sharpe {num(spy['sharpe'])} over the same dates. Cash (BIL) returned {pct(cash_ann)} a year, so most of the strategy's {pct(h['ann_return'])} is the cash yield on collateral.</p>
<h2>Headline (out-of-sample only)</h2>{headline_tbl}
<h2>Equity curve and drawdown</h2><img src="{equity_png}" alt="equity curve"><img src="{dd_png}" alt="drawdown">
<h2>Monthly returns, % (strategy, net)</h2>{monthly_html}
<h2>Cost stress</h2>{cost_tbl}
<h2>Crisis periods (all before the sealed test)</h2>
<table><tr><th></th><th>Window</th><th>Segment</th><th>Strategy</th><th>SPY</th><th>60/40</th><th>Strategy max DD</th></tr>{crisis_rows}</table>
<p class="mut">2008 starts 2008-07-01 because the signal needs 13 months of history. None of these windows is out-of-sample.</p>
<h2>Parameter sensitivity (lookback × vol target)</h2><img src="{heat_val}" alt="validation heatmap"><img src="{heat_oos}" alt="test heatmap">
<p class="mut">The primary stayed at 12 months / 10% whatever these showed. Neighbour gap: validation {num(r['neighbour_gap']['validation'])}, test {num(r['neighbour_gap']['oos'])}.</p>
<h2>Overfitting flags</h2><ul>{flags}</ul>
<h2>Verdict gates</h2><ul>{gates}</ul>
<p>Round trips by joint SPY regime at entry: {', '.join(f'{k} {v}' for k, v in cov.items())}. Sparse cells: {', '.join(r['regime_coverage']['sparse']) or 'none'}.</p>
<h2>Secondary: radar break rules vs next-month outcomes (test segment)</h2>
<table><tr><th>Rule → outcome</th><th>n break / hold</th><th>Break minus hold</th><th>95% CI</th><th>Survives BH 5%</th></tr>{radar_rows}</table>
<p class="mut">Observations are pair-months pooled across 11 pairs; the CI resamples months. Reported separately from the strategy.</p>
</body></html>"""
(RES / "tearsheet.html").write_text(html, encoding="utf-8")

# README and CV bullets
bh = [k for k, v in rt.items() if v and v["bh_significant_5pct"]]
nominal = [k for k, v in rt.items() if v and (v["ci95"][0] > 0 or v["ci95"][1] < 0)]
readme = f"""# systematic-trend-backtest

Cross-asset time-series momentum with volatility targeting on 16 liquid ETFs, validated under a pre-registered contract
(60/20/20 chronological split, 100+ out-of-sample round trips, Deflated Sharpe, cost stress). Rules were written in
`PREREGISTRATION.md` and committed before any data was downloaded. The sealed test was opened once.

## Result (sealed out-of-sample, {r['oos_window'][0]} to {r['oos_window'][1]}, 10 bps costs)
**Verdict: {verdict}.** The strategy did not earn a statistically distinguishable edge, and it lagged both benchmarks.

| | Strategy | SPY | 60/40 |
|---|---|---|---|
| Annualised return | {pct(h['ann_return'])} | {pct(spy['ann_return'])} | {pct(sf['ann_return'])} |
| Volatility | {pct(h['ann_vol'])} | {pct(spy['ann_vol'])} | {pct(sf['ann_vol'])} |
| Sharpe | {num(h['sharpe'])} | {num(spy['sharpe'])} | {num(sf['sharpe'])} |
| Max drawdown | {pct(h['max_drawdown'])} | {pct(spy['max_drawdown'])} | {pct(sf['max_drawdown'])} |

- Sharpe 95% bootstrap CI: {num(ci[0])} to {num(ci[1])} (includes zero). Deflated Sharpe probability: {num(dsr['probability']) if dsr else 'n/a'} against {r['trial_count']} registered trials (needs 0.95).
- Sortino {num(h['sortino'])}, Calmar {num(h['calmar'])}, hit rate {pct(h['hit_rate'])}, {h['round_trips']} round trips, turnover {num(h['turnover_per_year'])}x equity a year.
- Costs: Sharpe {num(r['oos_cost_stress']['1x']['sharpe'])} / {num(r['oos_cost_stress']['2x']['sharpe'])} / {num(r['oos_cost_stress']['3x']['sharpe'])} at 1x / 2x / 3x.
- Cash (BIL) earned {pct(cash_ann)} a year over the window, so most of the strategy's return is collateral yield, not trend profit.
- Flags: {'; '.join(r['flags']) or 'none'}.
- Failed gates: {'; '.join(r['verdict']['failed']) or 'none'}.
- Crises (2008, 2020, 2022) fall in train/validation, not out-of-sample; see the tearsheet table.

Full tables and charts: `results/tearsheet.html`. Machine-readable: `results/results.json`.

## Secondary study: radar break rules
Do the intermarket radar's break rules (20-session divergence, correlation sign flip, ratio z beyond ±2) predict next-month
returns? Test segment, pooled across {len(radar['pairs'])} pairs: {len(nominal)} of {len([v for v in rt.values() if v])} tests have a 95% CI excluding zero
({', '.join(nominal) or 'none'}), and {len(bh)} survive Benjamini-Hochberg at 5%. No evidence the rules predict returns. Pair coverage differs:
BAMLH0A0HYM2/SPY has only {radar['pairs']['BAMLH0A0HYM2/SPY']['reads']} usable months (FRED returned data only from {radar['pairs']['BAMLH0A0HYM2/SPY']['first']}), and GC=F/DFII10 has {radar['pairs']['GC=F/DFII10']['reads']}
(real yields cross zero, so the ratio z-score is undefined and the original rule gives "no read" unless another rule fires; those months skew toward breaks).
Only the 11 pairs registered in advance were tested; pairs needing derived or unavailable legs (cracks, MATIF, TTF, JP2Y, crypto) were not.

## Method
See `PREREGISTRATION.md` (rules, split, trial accounting, verdict logic, Addendum 1). Test lock: `src/stb/seal.py` refuses to open the
test without `freeze.json` matching config and engine code on a clean git tree; reruns must reproduce identical numbers.
No lookahead: signals use data up to day t and fill at the close of t+1. Costs are 5 bps commission + 5 bps slippage per unit of turnover; borrow 25 bps a year on shorts.

## Data
yfinance daily adjusted closes (dividends and splits), downloaded {dl[0][:10]} to {dl[-1][:10]} UTC; per-ticker source, URL, row counts and sha256 in `data/manifest.json`
(radar series in `data/manifest_radar.json`; FRED for DFII10 and BAMLH0A0HYM2). Raw CSVs are not committed; rerun `scripts/01_download.py` (results will drift slightly with Yahoo revisions).

## Reproduce
```
python -m venv .venv && .venv\\Scripts\\pip install -r requirements.txt
.venv\\Scripts\\python -m pytest -q
.venv\\Scripts\\python scripts\\01_download.py ; scripts\\02a_split.py ; scripts\\02_run_dev.py ; scripts\\03_freeze.py
.venv\\Scripts\\python scripts\\04_run_sealed_test.py    # once; later runs only verify reproduction
.venv\\Scripts\\python scripts\\05_radar.py ; scripts\\06_report.py
```

## Limits
- The sealed test is about 3.7 years of a strong equity bull market; the 100-round-trip gate was not met.
- Profit factor and hit rate are per round trip on unweighted asset returns net of round-trip costs; they ignore position size and cash yield.
- Yahoo adjusted closes are free-data quality. Borrow and slippage are flat assumptions. Long/short at up to 2x is not available in an ISA.
- Validation favoured 9-month lookbacks over the pre-registered 12 months; the primary was not changed, by design.
"""
(ROOT / "README.md").write_text(readme, encoding="utf-8")

cv = f"""- Built and pre-registered a cross-asset trend-following backtest (16 ETFs, 12-1 momentum, 10% volatility target) with a sealed 60/20/20 split, no-lookahead and cost-application tests, and Deflated Sharpe over {r['trial_count']} registered trials.
- Out-of-sample ({r['oos_window'][0][:7]} to {r['oos_window'][1][:7]}, net of costs): Sharpe {num(h['sharpe'])} (95% bootstrap CI {num(ci[0])} to {num(ci[1])}), {pct(h['ann_return'])} annualised at {pct(h['ann_vol'])} volatility, max drawdown {pct(h['max_drawdown'])}, against SPY Sharpe {num(spy['sharpe'])}; Deflated Sharpe probability {num(dsr['probability']) if dsr else 'n/a'}, verdict {verdict} under my own validation contract.
"""
(RES / "cv_bullets.md").write_text(cv, encoding="utf-8")
print(cv)
