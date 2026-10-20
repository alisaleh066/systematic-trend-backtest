import pandas as pd

from stb.split import compute_split


def test_split_is_chronological_month_end_and_about_60_20_20():
    idx = pd.bdate_range("2007-06-01", "2026-09-30")
    s = compute_split(idx, "2008-07-01", 0.6, 0.2)
    tr, va, te = (pd.Timestamp(s[k][0]) for k in ("train", "validation", "test"))
    assert pd.Timestamp(s["train"][1]) < va <= pd.Timestamp(s["validation"][1]) < te
    for end in (s["train"][1], s["validation"][1]):
        e = pd.Timestamp(end)
        assert (idx[idx > e][0].month != e.month)  # last session of its month
    n = s["sessions"]
    assert n["train"] + n["validation"] + n["test"] == n["total"]
    assert abs(n["train"] / n["total"] - 0.6) < 0.03
    assert abs(n["validation"] / n["total"] - 0.2) < 0.03
