"""Build tearsheet_v3.html, add a cross-version summary and the v3 section to README.md, and write the CV bullets.
Run after 06_report.py and 12_report_v2.py. Every number is read from results JSON, never typed."""
import base64
import io
import json
import re
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
v1 = json.loads((RES / "results.json").read_text())
v2 = json.loads((RES / "v2_results.json").read_text())
r = json.loads((RES / "v3_results.json").read_text())
cfg = json.loads((ROOT / "config" / "v3.json").read_text())
manifest = json.loads((ROOT / "data" / "manifest_v3.json").read_text())
daily = pd.read_csv(RES / "v3_oos_daily.csv", parse_dates=["date"], index_col="date")
h, ci, dsr = r["headline_holdout_1x_costs"], r["sharpe_bootstrap_95ci"], r["deflated_sharpe"]
vf, sf = r["benchmarks_holdout"]["vfinx"], r["benchmarks_holdout"]["sixty_forty"]
pct = lambda v, d=1: "n/a" if v is None else f"{v * 100:.{d}f}%"  # noqa: E731
num = lambda v, d=2: "n/a" if v is None else f"{v:.{d}f}"  # noqa: E731
cash_ann = float((1 + daily["rf"]).prod() ** (252 / len(daily)) - 1)
ho = r["holdout_window"]
years = round(len(daily) / 252, 1)
dl = sorted(t["downloaded_utc"] for t in manifest["series"])
excluded = r["panel_log"]["excluded"]


def png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


eq = (1 + daily[["strategy", "vfinx", "sixty_forty"]]).cumprod()
fig, ax = plt.subplots(figsize=(8, 3.6))
for c, label in (("strategy", "v3 strategy (frozen v2 rule)"), ("vfinx", "S&P 500 fund (VFINX)"), ("sixty_forty", "60/40 VFINX/VBMFX")):
    ax.plot(eq.index, eq[c], label=label, lw=1.6 if c == "strategy" else 1.1)
ax.set_yscale("log")
ax.set_title("Holdout equity curve, log scale (growth of 1, net of costs)")
ax.legend(frameon=False, fontsize=8)
ax.grid(alpha=0.2)
equity_png = png(fig)
dd = eq / eq.cummax() - 1
fig, ax = plt.subplots(figsize=(8, 2.8))
for c in dd:
    ax.plot(dd.index, dd[c], lw=1.4 if c == "strategy" else 1.0, label=c)
ax.fill_between(dd.index, dd["strategy"], 0, alpha=0.15)
ax.set_title("Drawdown")
ax.legend(frameon=False, fontsize=8)
ax.grid(alpha=0.2)
dd_png = png(fig)

vols = cfg["sensitivity_vol_targets"]
z = np.array([[r["sensitivity_sharpe"][f"{k}|{v}"] for v in vols] for k in (1, 2, 3)])
fig, ax = plt.subplots(figsize=(5, 2.6))
im = ax.imshow(z, cmap="RdYlGn", vmin=-1.5, vmax=1.5, aspect="auto")
ax.set_xticks(range(len(vols)), [f"{v:.0%}" for v in vols])
ax.set_yticks(range(3), ["S1 12-1 (frozen)", "S2 1/3/12", "S3 3/6/12 skip"])
ax.set_xlabel("vol target")
ax.set_title("Holdout Sharpe (sensitivity only)", fontsize=10)
for i in range(3):
    for j in range(len(vols)):
        ax.text(j, i, f"{z[i, j]:.2f}", ha="center", va="center", fontsize=8)
fig.colorbar(im)
heat_png = png(fig)

monthly = (1 + daily["strategy"]).resample("ME").prod() - 1
tab = monthly.to_frame("r")
tab["y"], tab["m"] = tab.index.year, tab.index.month
piv = tab.pivot(index="y", columns="m", values="r")
piv["Year"] = (1 + daily["strategy"]).groupby(daily.index.year).prod() - 1
months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
cell = lambda v: "<td></td>" if pd.isna(v) else f'<td class="{"pos" if v > 0 else "neg"}">{v * 100:.1f}</td>'  # noqa: E731
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


def crisis_html(rows: dict, label: str) -> str:
    return "".join(f"<tr><th>{k}</th><td>{v['window'][0]} to {v['window'][1]}</td><td>{label}</td><td>{pct(v['strategy']['total_return'])}</td><td>{pct(v['vfinx']['total_return'])}</td><td>{pct(v['sixty_forty']['total_return'])}</td></tr>" for k, v in rows.items())


head_tbl = stat_rows({"v3 strategy": h, "VFINX": vf, "60/40": sf})
cost_tbl = stat_rows({"1x (10 bps)": r["holdout_cost_stress"]["1x"], "2x (20 bps)": r["holdout_cost_stress"]["2x"], "3x (30 bps)": r["holdout_cost_stress"]["3x"]})
crisis_rows = crisis_html(r["crisis_periods_holdout"], "holdout") + crisis_html(r["crisis_periods_development"], "development (seen)")
gates = "".join(f"<li>{'PASS' if ok else 'FAIL'}: {g}</li>" for g, ok in r["verdict"]["gates"].items())
flags = "".join(f"<li>{f}</li>" for f in r["flags"]) or "<li>none</li>"
cov = r["regime_coverage"]["trades"]
verdict = r["verdict"]["verdict"]
excl = "; ".join(f"{k}: {v[:60]}" for k, v in excluded.items()) or "none"

html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Systematic trend tearsheet v3</title><style>
:root{{--bg:#fff;--fg:#1a1a1a;--mut:#666;--line:#ddd;--pos:#1a7f37;--neg:#c62828}}
@media(prefers-color-scheme:dark){{:root{{--bg:#16181c;--fg:#e8e8e8;--mut:#9aa;--line:#333;--pos:#5fd38a;--neg:#ff7b72}}}}
body{{font:15px/1.5 system-ui,sans-serif;background:var(--bg);color:var(--fg);max-width:980px;margin:0 auto;padding:16px}}
table{{border-collapse:collapse;margin:8px 0;font-size:13px;display:block;overflow-x:auto}}td,th{{border:1px solid var(--line);padding:3px 8px;text-align:right}}th{{text-align:left}}
.pos{{color:var(--pos)}}.neg{{color:var(--neg)}}img{{max-width:100%;height:auto}}.mut{{color:var(--mut)}}
.badge{{display:inline-block;padding:2px 10px;border:2px solid var(--line);border-radius:6px;font-weight:700}}</style></head><body>
<h1>v3: frozen rule on a 1988-2002 holdout</h1>
<p class="mut">Holdout {ho[0]} to {ho[1]} ({years} years, pre-sample: departs from contract line 60) · frozen v2 rule, nothing re-selected · {len(r['assets'])} long-history proxies · 10 bps costs · {r['trial_count']} registered trials</p>
<p>Verdict under the pre-registered contract: <span class="badge">{verdict}</span> (failed: {'; '.join(r['verdict']['failed'])})</p>
<p>Sharpe {num(h['sharpe'])}, bootstrap 95% CI {num(ci[0])} to {num(ci[1])} (above zero); Deflated Sharpe probability {num(dsr['probability'])} (needs 0.95). Realised vol {pct(h['ann_vol'])} is below the 10% target because the 2x gross cap binds on low-vol bonds and FX. Cash returned {pct(cash_ann)} a year.</p>
<h2>Headline (holdout only)</h2>{head_tbl}
<h2>Equity curve and drawdown</h2><img src="{equity_png}" alt="equity curve"><img src="{dd_png}" alt="drawdown">
<h2>Monthly returns, % (net)</h2>{monthly_html}
<h2>Cost stress</h2>{cost_tbl}
<h2>Sensitivity</h2><img src="{heat_png}" alt="sensitivity heatmap">
<p class="mut">Neighbour gap on this heatmap {num(r['neighbour_gap']['holdout_heatmap'])}. The S1 rule was frozen in v2; S2/S3 are shown only as sensitivity.</p>
<h2>Crisis windows</h2>
<table><tr><th></th><th>Window</th><th>Data</th><th>Strategy</th><th>VFINX</th><th>60/40</th></tr>{crisis_rows}</table>
<h2>Overfitting flags</h2><ul>{flags}</ul>
<p class="mut">The two selection flags are inherited from v2, where this rule was chosen.</p>
<h2>Verdict gates</h2><ul>{gates}</ul>
<p>Round trips by joint VFINX regime at entry: {', '.join(f'{k} {v}' for k, v in cov.items())}. Sparse cells: {', '.join(r['regime_coverage']['sparse']) or 'none'}.</p>
<h2>Universe</h2><p>{', '.join(r['assets'])}. Excluded by registered rules: {excl}.</p>
</body></html>"""
(RES / "tearsheet_v3.html").write_text(html, encoding="utf-8")

h1, ci1 = v1["headline_oos_1x_costs"], v1["sharpe_bootstrap_95ci"]
h2, ci2 = v2["headline_holdout_1x_costs"], v2["sharpe_bootstrap_95ci"]
summary = f"""<!-- summary:start -->
## Summary across versions (all net of costs, each test opened once)
| Version | Test window | Years | Sharpe | 95% CI | Deflated Sharpe | Round trips | Verdict |
|---|---|---|---|---|---|---|---|
| v1 | {v1['oos_window'][0]} to {v1['oos_window'][1]} (final 20%) | {round(h1['sessions'] / 252, 1)} | {num(h1['sharpe'])} | {num(ci1[0])} to {num(ci1[1])} | {num(v1['deflated_sharpe']['probability'])} | {h1['round_trips']} | {v1['verdict']['verdict']} |
| v2 | {v2['holdout_window'][0]} to {v2['holdout_window'][1]} (pre-sample) | {round(h2['sessions'] / 252, 1)} | {num(h2['sharpe'])} | {num(ci2[0])} to {num(ci2[1])} | {num(v2['deflated_sharpe']['probability'])} | {h2['round_trips']} | {v2['verdict']['verdict']} |
| v3 | {ho[0]} to {ho[1]} (pre-sample) | {years} | {num(h['sharpe'])} | {num(ci[0])} to {num(ci[1])} | {num(dsr['probability'])} | {h['round_trips']} | {verdict} |

The strongest evidence is v3: the rule frozen in v2, unchanged, on {years} never-seen years. Its Sharpe CI excludes zero, but it is still WATCH:
{'; '.join(r['verdict']['failed'])}, and the Deflated Sharpe ({num(dsr['probability'])}) is below 0.95. Nothing was tuned on any test set.
<!-- summary:end -->

"""
sec = f"""
## v3: frozen rule on a long-history proxy universe, {ho[0][:4]}-{ho[1][:4]} holdout
Registered in `PREREGISTRATION-v3.md` before any v3 data was downloaded. The v2 rule (12-1 sign, 126-session vol window, 10% target, gross cap 2x)
was applied **unchanged**: nothing was selected. Universe: {len(r['assets'])} long-history proxies (Vanguard equity, bond and REIT funds; FRED spot FX as excess-return proxies).
Excluded by the registered rules: {excl}. So v3 has **no commodities**.

**Verdict: {verdict}.**

| Holdout {ho[0]} to {ho[1]}, 10 bps | v3 strategy | S&P 500 fund (VFINX) | 60/40 VFINX/VBMFX |
|---|---|---|---|
| Annualised return | {pct(h['ann_return'])} | {pct(vf['ann_return'])} | {pct(sf['ann_return'])} |
| Volatility | {pct(h['ann_vol'])} | {pct(vf['ann_vol'])} | {pct(sf['ann_vol'])} |
| Sharpe | {num(h['sharpe'])} | {num(vf['sharpe'])} | {num(sf['sharpe'])} |
| Max drawdown | {pct(h['max_drawdown'])} | {pct(vf['max_drawdown'])} | {pct(sf['max_drawdown'])} |
| Calmar | {num(h['calmar'])} | {num(vf['calmar'])} | {num(sf['calmar'])} |

- Sharpe 95% bootstrap CI: **{num(ci[0])} to {num(ci[1])}, above zero.** Deflated Sharpe probability {num(dsr['probability'])} against {r['trial_count']} trials (needs 0.95, flagged).
- {h['round_trips']} round trips (gate passed), hit rate {pct(h['hit_rate'])}, profit factor {num(h['profit_factor'])}, Sortino {num(h['sortino'])}, turnover {num(h['turnover_per_year'])}x a year.
- Cost stress Sharpe {num(r['holdout_cost_stress']['1x']['sharpe'])} / {num(r['holdout_cost_stress']['2x']['sharpe'])} / {num(r['holdout_cost_stress']['3x']['sharpe'])} at 1x / 2x / 3x.
- Dot-com bust ({r['crisis_periods_holdout']['2000-02']['window'][0]} to {r['crisis_periods_holdout']['2000-02']['window'][1]}): strategy {pct(r['crisis_periods_holdout']['2000-02']['strategy']['total_return'])}, VFINX {pct(r['crisis_periods_holdout']['2000-02']['vfinx']['total_return'])}.
- Realised vol {pct(h['ann_vol'])} is under the 10% target because the 2x gross cap binds on low-vol bonds and FX. Cash earned {pct(cash_ann)} a year.
- Flags: {'; '.join(r['flags'])}. The selection flags are inherited from v2, where the rule was chosen.
- Failed gate: {'; '.join(r['verdict']['failed'])} (trend-regime cells: {', '.join(f'{k} {v}' for k, v in r['regime_coverage']['trades'].items())}).
- Caveats: the holdout is pre-sample, not the contract's final 20%. Mutual funds can't really be shorted. FX proxies omit carry. 1990s trend-following results are public, so this is an independent-data check, not proof.

Tearsheet: `results/tearsheet_v3.html`. Machine-readable: `results/v3_results.json`. Data: yfinance and FRED, downloaded {dl[0][:10]} UTC (`data/manifest_v3.json`).
Reproduce v2/v3: `scripts/08`-`12` (v2), `scripts/13`-`17` (v3); the holdout scripts refuse to run without a matching freeze, and a rerun must reproduce identical numbers.
"""
gates_passed = sum(r["verdict"]["gates"].values())
limits = f"""
## Why the verdict is WATCH, and what a practitioner would object to
- **Why WATCH, not PROMOTE.** v3 passes {gates_passed} of {len(r['verdict']['gates'])} contract gates, including a Sharpe CI above zero and {h['round_trips']} round trips.
  It fails one: regime coverage. Only {r['regime_coverage']['trades']['trend/high']} and {r['regime_coverage']['trades']['trend/low']} trades entered in the two trending-market cells against 20 required, because an efficiency
  ratio of 0.30 or more is rare and a monthly strategy trades little inside it. The Deflated Sharpe ({num(dsr['probability'])}) is also under 0.95. The gate was not loosened after seeing results.
- **What would change it.** More years or markets inside trending regimes, or a change to the regime gate agreed and registered *before* a new version (v4) is run. A v4 needs fresh data: 1988-2008 and 2023-2026 have all been seen.
  The contract-compliant route is the forward paper-trade (`scripts/07_forward_check.py`), which needs months to years.
- **Data limits.** Mutual funds can't really be shorted; spot FX omits interest carry; v3 has no commodities (gold failed to download, oil was excluded by the registered gap rule). Effects on the Sharpe could go either way and are not measured here.
- **Originality.** Trend following is a well-established strategy and this project does not claim a new edge. What it offers is the validation discipline: pre-registration before data,
  sealed tests opened once, trial-counted Deflated Sharpe, cost stress, and a radar-rule study reported separately with no evidence of predictive power.
"""
readme = (ROOT / "README.md").read_text(encoding="utf-8")
readme = re.sub(r"<!-- summary:start -->.*?<!-- summary:end -->\n\n", "", readme, flags=re.S)
readme = readme.split("\n## v3: frozen rule")[0].rstrip("\n") + "\n" + sec + limits
readme = readme.replace("## Result (sealed out-of-sample", summary + "## v1 result (sealed out-of-sample", 1).replace(
    summary + "## v1 result (sealed out-of-sample", summary + "## v1 result (sealed out-of-sample")
(ROOT / "README.md").write_text(readme, encoding="utf-8")

cv = f"""- Pre-registered and backtested a cross-asset trend-following strategy across three versions with sealed, opened-once test sets, a Deflated Sharpe over {r['trial_count']} logged trials, cost stress to 3x, and automated no-lookahead, cost and test-lock checks; reported a weak first result (Sharpe {num(h1['sharpe'])}) instead of tuning it.
- On a never-seen {ho[0][:4]}-{ho[1][:4]} holdout ({years} years, {h['round_trips']} round trips) the frozen rule returned a Sharpe of {num(h['sharpe'])} net of costs (95% CI {num(ci[0])} to {num(ci[1])}) against {num(sf['sharpe'])} for a 60/40 portfolio, with a {pct(h['max_drawdown'])} max drawdown versus {pct(vf['max_drawdown'])} for the S&P 500; verdict WATCH under my own contract (Deflated Sharpe {num(dsr['probability'])}).
"""
(RES / "cv_bullets.md").write_text(cv, encoding="utf-8")
print(cv)
