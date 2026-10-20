"""Build tearsheet_v2.html, append the v2 section to README.md and write 3 CV bullets. Run after 06_report.py.
Every number is read from results JSON, never typed."""
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
v1 = json.loads((RES / "results.json").read_text())
r = json.loads((RES / "v2_results.json").read_text())
dev = json.loads((RES / "v2_dev_summary.json").read_text())
cfg = json.loads((ROOT / "config" / "v2.json").read_text())
manifest = json.loads((ROOT / "data" / "manifest_v2.json").read_text())
daily = pd.read_csv(RES / "v2_oos_daily.csv", parse_dates=["date"], index_col="date")
h, ci, dsr = r["headline_holdout_1x_costs"], r["sharpe_bootstrap_95ci"], r["deflated_sharpe"]
spy, sf, v1r = r["benchmarks_holdout"]["spy"], r["benchmarks_holdout"]["sixty_forty"], r["v1_primary_rule_on_holdout"]
pct = lambda v, d=1: "n/a" if v is None else f"{v * 100:.{d}f}%"  # noqa: E731
num = lambda v, d=2: "n/a" if v is None else f"{v:.{d}f}"  # noqa: E731
cash_ann = float((1 + daily["rf"]).prod() ** (252 / len(daily)) - 1)
ho = r["holdout_window"]
sel = r["selected_design"]
dl = sorted(t["downloaded_utc"] for t in manifest["series"])


def png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


eq = (1 + daily[["strategy", "spy", "sixty_forty"]]).cumprod()
fig, ax = plt.subplots(figsize=(8, 3.6))
for c, label in (("strategy", f"v2 strategy ({sel})"), ("spy", "Buy-and-hold SPY"), ("sixty_forty", "60/40 SPY/IEF")):
    ax.plot(eq.index, eq[c], label=label, lw=1.6 if c == "strategy" else 1.1)
ax.set_title("Pre-sample holdout equity curve (growth of 1, net of costs)")
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
ax.set_yticks(range(3), ["S1 12-1", "S2 1/3/12", "S3 3/6/12 skip"])
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


head_tbl = stat_rows({"v2 strategy": h, "SPY": spy, "60/40": sf, "v1 rule (S1/U16/63)": v1r})
cost_tbl = stat_rows({"1x (10 bps)": r["holdout_cost_stress"]["1x"], "2x (20 bps)": r["holdout_cost_stress"]["2x"], "3x (30 bps)": r["holdout_cost_stress"]["3x"]})
crisis_rows = "".join(f"<tr><th>{k}</th><td>{v['window'][0]} to {v['window'][1]}</td><td>{v['segment']}</td><td>{pct(v['strategy']['total_return'])}</td><td>{pct(v['spy']['total_return'])}</td><td>{pct(v['sixty_forty']['total_return'])}</td></tr>"
                      for k, v in r["crisis_periods_development"].items())
design_rows = "".join(f"<tr><th>{k}</th><td>{num(v['train_sharpe'])}</td><td>{num(v['validation_sharpe'])}</td><td>{v['validation']['round_trips']}</td></tr>" for k, v in dev["designs"].items())
gates = "".join(f"<li>{'PASS' if ok else 'FAIL'}: {g}</li>" for g, ok in r["verdict"]["gates"].items())
flags = "".join(f"<li>{f}</li>" for f in r["flags"]) or "<li>none</li>"
cov = r["regime_coverage"]["trades"]
verdict = r["verdict"]["verdict"]

html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Systematic trend tearsheet v2</title><style>
:root{{--bg:#fff;--fg:#1a1a1a;--mut:#666;--line:#ddd;--pos:#1a7f37;--neg:#c62828}}
@media(prefers-color-scheme:dark){{:root{{--bg:#16181c;--fg:#e8e8e8;--mut:#9aa;--line:#333;--pos:#5fd38a;--neg:#ff7b72}}}}
body{{font:15px/1.5 system-ui,sans-serif;background:var(--bg);color:var(--fg);max-width:980px;margin:0 auto;padding:16px}}
table{{border-collapse:collapse;margin:8px 0;font-size:13px;display:block;overflow-x:auto}}td,th{{border:1px solid var(--line);padding:3px 8px;text-align:right}}th{{text-align:left}}
.pos{{color:var(--pos)}}.neg{{color:var(--neg)}}img{{max-width:100%;height:auto}}.mut{{color:var(--mut)}}
.badge{{display:inline-block;padding:2px 10px;border:2px solid var(--line);border-radius:6px;font-weight:700}}</style></head><body>
<h1>v2: pre-sample holdout tearsheet</h1>
<p class="mut">Holdout {ho[0]} to {ho[1]} (earlier than the development data, so it departs from contract line 60) · selected design {sel} · 10 bps costs · {r['trial_count']} registered trials</p>
<p>Verdict under the pre-registered contract: <span class="badge">{verdict}</span></p>
<p>Sharpe {num(h['sharpe'])} (bootstrap 95% CI {num(ci[0])} to {num(ci[1])}, includes zero); Deflated Sharpe probability {num(dsr['probability']) if dsr else 'n/a'}. SPY Sharpe {num(spy['sharpe'])}, 60/40 {num(sf['sharpe'])}. Cash returned {pct(cash_ann)} a year over the window. 2002-2008 was a favourable era for trend following, so read this as a pre-sample check, not proof.</p>
<h2>Headline (holdout only)</h2>{head_tbl}
<h2>Equity curve and drawdown</h2><img src="{equity_png}" alt="equity curve"><img src="{dd_png}" alt="drawdown">
<h2>Monthly returns, % (net)</h2>{monthly_html}
<h2>Cost stress</h2>{cost_tbl}
<h2>Sensitivity</h2><img src="{heat_png}" alt="sensitivity heatmap">
<p class="mut">Neighbour gap on this heatmap {num(r['neighbour_gap']['holdout_heatmap'])}; selected design vs the 11 others on validation {num(r['neighbour_gap']['selected_vs_other_designs_validation'])}.</p>
<h2>The 12 registered designs (development)</h2>
<table><tr><th>signal | universe | vol window</th><th>Train Sharpe</th><th>Validation Sharpe</th><th>Validation trips</th></tr>{design_rows}</table>
<p class="mut">Selection rule: highest validation Sharpe. The ensemble signals (S2, S3) did worse on validation than the single 12-1 signal.</p>
<h2>Crisis windows (development data, not a test)</h2>
<table><tr><th></th><th>Window</th><th>Segment</th><th>Strategy</th><th>SPY</th><th>60/40</th></tr>{crisis_rows}</table>
<h2>Overfitting flags</h2><ul>{flags}</ul>
<h2>Verdict gates</h2><ul>{gates}</ul>
<p>Round trips by joint SPY regime at entry: {', '.join(f'{k} {v}' for k, v in cov.items())}. Sparse cells: {', '.join(r['regime_coverage']['sparse']) or 'none'}.</p>
</body></html>"""
(RES / "tearsheet_v2.html").write_text(html, encoding="utf-8")

sec = f"""
## v2: pre-sample holdout (new registration, same contract gates)
After v1's test set was spent, v2 was registered in `PREREGISTRATION-v2.md` (committed before any new data was downloaded) and judged on a
**never-seen pre-sample holdout, {ho[0]} to {ho[1]}**. Changes from v1: point-in-time universe widened to 24 ETFs, FRED DTB3 as cash, and 12 registered
designs (signal x universe x vol window), the best on validation chosen by a fixed rule: **{sel}**. Deflated Sharpe uses {r['trial_count']} trials (v1's 22 included).

**Verdict: {verdict}.**

| Holdout, 10 bps costs | v2 strategy | SPY | 60/40 | v1 rule (S1/U16/63) |
|---|---|---|---|---|
| Annualised return | {pct(h['ann_return'])} | {pct(spy['ann_return'])} | {pct(sf['ann_return'])} | {pct(v1r['ann_return'])} |
| Volatility | {pct(h['ann_vol'])} | {pct(spy['ann_vol'])} | {pct(sf['ann_vol'])} | {pct(v1r['ann_vol'])} |
| Sharpe | {num(h['sharpe'])} | {num(spy['sharpe'])} | {num(sf['sharpe'])} | {num(v1r['sharpe'])} |
| Max drawdown | {pct(h['max_drawdown'])} | {pct(spy['max_drawdown'])} | {pct(sf['max_drawdown'])} | {pct(v1r['max_drawdown'])} |

- Sharpe 95% bootstrap CI: {num(ci[0])} to {num(ci[1])} (includes zero). Deflated Sharpe probability {num(dsr['probability']) if dsr else 'n/a'} (needs 0.95).
- {h['round_trips']} round trips (the contract needs 100), hit rate {pct(h['hit_rate'])}, profit factor {num(h['profit_factor'])}, turnover {num(h['turnover_per_year'])}x a year. Cost stress Sharpe: {num(r['holdout_cost_stress']['1x']['sharpe'])} / {num(r['holdout_cost_stress']['2x']['sharpe'])} / {num(r['holdout_cost_stress']['3x']['sharpe'])} at 1x / 2x / 3x.
- Cash earned {pct(cash_ann)} a year over the window.
- Flags: {'; '.join(r['flags'])}.
- Failed gates: {'; '.join(r['verdict']['failed'])}.
- What did not help: the 1/3/12-month and 3/6/12-month ensemble signals scored lower on validation than the single 12-1 signal, and they also scored lower on the holdout (see the sensitivity heatmap), so the literature-motivated change was not an improvement here. Widening the universe raised round trips but lowered train Sharpe.
- Caveats: the holdout is *earlier* than the development data (contract line 60 wants the final 20%), 2002-2008 was a favourable era for trend following, and the selection rule picked a design whose train Sharpe was low. A contract-compliant check is prospective: `scripts/07_forward_check.py` logs the frozen rule from the freeze date; no forward numbers exist yet.

Tearsheet: `results/tearsheet_v2.html`. Machine-readable: `results/v2_results.json`. v2 data: yfinance and FRED, downloaded {dl[0][:10]} UTC (`data/manifest_v2.json`).
"""
readme = (ROOT / "README.md").read_text(encoding="utf-8")
readme = readme.split("\n## v2: pre-sample holdout")[0].rstrip("\n") + "\n" + sec
(ROOT / "README.md").write_text(readme, encoding="utf-8")

h1, ci1, spy1 = v1["headline_oos_1x_costs"], v1["sharpe_bootstrap_95ci"], v1["benchmarks_oos"]["spy"]
cv = f"""- Built and pre-registered a cross-asset trend-following research framework (16-24 ETFs, volatility targeting) with sealed chronological splits, a Deflated Sharpe over {r['trial_count']} registered trials, and no-lookahead, cost-application and test-lock tests.
- v1 out-of-sample ({v1['oos_window'][0][:7]} to {v1['oos_window'][1][:7]}, net of costs): Sharpe {num(h1['sharpe'])} (95% bootstrap CI {num(ci1[0])} to {num(ci1[1])}) against SPY {num(spy1['sharpe'])}, reported as an inconclusive WATCH rather than tuned.
- v2 on a never-seen pre-sample holdout ({ho[0][:7]} to {ho[1][:7]}, {h['round_trips']} round trips, net of costs): Sharpe {num(h['sharpe'])} (95% CI {num(ci[0])} to {num(ci[1])}) against SPY {num(spy['sharpe'])} and 60/40 {num(sf['sharpe'])}; Deflated Sharpe probability {num(dsr['probability']) if dsr else 'n/a'}, WATCH under my own contract.
"""
(RES / "cv_bullets.md").write_text(cv, encoding="utf-8")
print(cv)
