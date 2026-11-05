import os
import random
import sys
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ti_engine as e  # noqa: E402
import ti_synth as s  # noqa: E402

AS_OF = date(2026, 11, 5)
FROM = date(2021, 4, 1)


# ------------------------------------------------------------ INR formatting
@pytest.mark.parametrize("x,want", [
    (0, "\u20b90.00"), (999.5, "\u20b9999.50"), (1000, "\u20b91,000.00"), (99999, "\u20b999,999.00"),
    (100000, "\u20b91,00,000.00"), (1234567.5, "\u20b912,34,567.50"), (12345678, "\u20b91,23,45,678.00"),
    (-1234567.5, "-\u20b912,34,567.50"), (1e12, "\u20b910,00,00,00,00,000.00"), (-0.001, "\u20b90.00"),
])
def test_fmt_inr(x, want):
    assert e.fmt_inr(x) == want


def test_fmt_inr_bracket_and_nan():
    assert e.fmt_inr(-5000, style="bracket") == "(\u20b95,000.00)"
    assert e.fmt_inr(float("nan")) == "n/a" and e.fmt_inr(None) == "n/a"


def test_fmt_short_rollover():
    assert e.fmt_inr_short(9999999) == "\u20b91.00 Cr"      # not 100.00 L
    assert e.fmt_inr_short(9990000) == "\u20b999.90 L"
    assert e.fmt_inr_short(100000) == "\u20b91.00 L"
    assert e.fmt_inr_short(-12345678) == "-\u20b91.23 Cr"
    assert e.fmt_inr_short(950) == "\u20b9950"


def test_fmt_inr_roundtrip_fuzz():
    rnd = random.Random(5)
    for _ in range(2000):
        x = round(rnd.uniform(-1e11, 1e11), 2)
        back = float(e.fmt_inr(x).replace("\u20b9", "").replace(",", ""))
        assert abs(back - x) < 0.006


# ------------------------------------------------------------ financial year
def test_fy_edges():
    assert e.fy_start_year(date(2026, 3, 31)) == 2025 and e.fy_start_year(date(2026, 4, 1)) == 2026
    assert e.fy_label(2025) == "FY2025-26" and e.fy_label(2099) == "FY2099-00"


def test_full_fys():
    assert e.full_fys(FROM, AS_OF) == [2021, 2022, 2023, 2024, 2025]          # 2026 is a part year
    assert e.full_fys(date(2021, 7, 1), AS_OF) == [2022, 2023, 2024, 2025]      # first year partial -> excluded
    assert e.full_fys(FROM, date(2026, 3, 31)) == [2021, 2022, 2023, 2024, 2025]
    assert e.full_fys(FROM, date(2026, 3, 30)) == [2021, 2022, 2023, 2024]


def test_lfl_windows_leap():
    cur, pri = e.lfl_windows(date(2028, 2, 29))
    assert cur == (date(2027, 4, 1), date(2028, 2, 29)) and pri == (date(2026, 4, 1), date(2027, 2, 28))
    cur, pri = e.lfl_windows(date(2026, 11, 5))
    assert pri == (date(2025, 4, 1), date(2025, 11, 5))
    cur, pri = e.lfl_windows(date(2026, 4, 1))
    assert cur[0] == cur[1] and pri[0] == pri[1] == date(2025, 4, 1)


# ------------------------------------------------------------ Tally sign rule
def test_sign_convention():
    # opening 100 credit, sales credit 500, purchases debit stored negative -300 -> closing 300
    assert e.movement(-300, 500) == 200
    assert e.sign_convention_ok(100, 300, -300, 500)
    assert not e.sign_convention_ok(100, 300, 300, 500)   # the wrong (net_credit - net_debit) reading must fail
    assert not e.sign_convention_ok(100, 300, -300, -500)


# ------------------------------------------------------------ gate / lineage
def test_tie_and_gate():
    assert e.tie("x", 100.0, 100.005)["pass"]
    assert not e.tie("x", 100.0, 100.02)["pass"]
    assert e.gate([e.tie("a", 1, 1)]) == "OK"
    assert e.gate([e.tie("a", 1, 1), e.tie("b", 1, 2)]) == "DATA CHECK FAILED"
    assert e.gate([]) == "DATA CHECK FAILED"            # no checks = not verified


def test_lineage_requires_all_fields():
    with pytest.raises(ValueError):
        e.lineage("m", "", "p", 1, "x", "t")
    assert e.lineage("m", "s", "p", 3, "x", "t")["rows"] == 3


# ------------------------------------------------------------ prep
def test_prep_reports_exclusions():
    df = pd.DataFrame({"date": ["2024-05-01", "bad", "2024-06-01", "2024-06-01"], "voucher_no": [1, 2, 3, 3],
                       "customer": ["A  Ltd", "B", None, None], "product_line": ["X", "X", None, None], "amount": [100, 50, float("nan"), 70]})
    d, issues = e.prep_sales(df)
    assert len(d) == 2 and any("excluded" in i for i in issues)
    assert set(d["customer"]) == {"A Ltd", e.WALKIN}
    assert any("no product-line" in i for i in issues)


def test_prep_missing_columns():
    with pytest.raises(ValueError):
        e.prep_sales(pd.DataFrame({"date": [1]}))


def test_prep_infinite_amount():
    df = pd.DataFrame({"date": ["2024-05-01"] * 2, "voucher_no": [1, 2], "customer": ["A", "B"], "product_line": ["X", "X"], "amount": [float("inf"), 5]})
    d, issues = e.prep_sales(df)
    assert len(d) == 1 and np.isfinite(d["amount"]).all()


# ------------------------------------------------------------ growth
def test_growth_and_cagr_full_years_only():
    ser = {2021: 100.0, 2022: 110.0, 2023: 121.0, 2026: 5.0}
    full = [2021, 2022, 2023]
    g = e.yoy_growth(ser, full)
    assert g[2021] is None and abs(g[2022] - 0.10) < 1e-12 and abs(g[2023] - 0.10) < 1e-12
    assert abs(e.cagr(ser, full) - 0.10) < 1e-12
    assert e.cagr(ser, [2021]) is None and e.cagr({2021: 0, 2022: 5}, [2021, 2022]) is None
    assert e.cagr({2021: 5, 2022: -5}, [2021, 2022]) is None


# ------------------------------------------------------------ tiers
def test_tiers_ten_equal():
    t = e.abc_tiers(pd.Series({f"c{i}": 100.0 for i in range(10)}))
    assert t["tier"].value_counts().to_dict() == {"GOLD": 7, "SILVER": 2, "BRONZE": 1}


def test_tiers_one_whale_and_single():
    t = e.abc_tiers(pd.Series({"whale": 1000.0, "a": 1.0, "b": 1.0}))
    assert t.loc[t.customer == "whale", "tier"].item() == "GOLD"
    assert e.abc_tiers(pd.Series({"only": 5.0}))["tier"].tolist() == ["GOLD"]
    assert e.abc_tiers(pd.Series({"neg": -5.0, "zero": 0.0})).empty


def test_tiers_deterministic_ties():
    s1 = pd.Series({"b": 10.0, "a": 10.0, "c": 10.0})
    assert e.abc_tiers(s1)["customer"].tolist() == ["a", "b", "c"]


def test_top_n_share():
    assert abs(e.top_n_share(pd.Series({"a": 50.0, "b": 30.0, "c": 20.0, "d": -10.0}), 2) - 0.8) < 1e-12
    assert e.top_n_share(pd.Series({"a": -1.0}), 3) is None


# ------------------------------------------------------------ retention
def test_retention_known():
    rows = []

    def add(c, y, a):
        rows.append({"date": date(y, 6, 1), "voucher_no": f"{c}{y}", "customer": c, "product_line": "X", "amount": a})
    add("A", 2021, 100); add("A", 2022, 150)
    add("B", 2021, 100)
    add("C", 2022, 40)
    add("D", 2021, 50); add("D", 2023, 60)
    d, _ = e.prep_sales(pd.DataFrame(rows))
    r = e.retention(d, [2022, 2023]).set_index("fy")
    y22 = r.loc[2022]
    # FY2021 active: A,B,D. FY2022 active: A,C. Retained A; new C; churned B,D
    assert y22["active"] == 2 and y22["retained"] == 1 and y22["new"] == 1 and y22["churned"] == 2
    assert abs(y22["logo_retention"] - 1 / 3) < 1e-12 and abs(y22["nrr"] - 150 / 250) < 1e-12
    assert abs(y22["grr"] - 100 / 250) < 1e-12
    y23 = r.loc[2023]
    assert y23["reactivated"] == 1 and y23["active"] == 1      # D comes back


# ------------------------------------------------------------ white space / winback
def _small_sales():
    rows = [
        ("2026-06-01", "1", "A", "LIC", 100000), ("2026-06-02", "2", "A", "AMC", 10000),
        ("2026-06-03", "3", "B", "LIC", 50000), ("2026-06-04", "4", "C", "AMC", 30000),
        ("2026-06-05", "5", "D", "LIC", 900000),
        ("2022-06-05", "6", "D", "AMC", 40000),
        ("2023-01-05", "7", "OLD CLOSED", "AMC", 99999),
        ("2022-02-05", "8", "Gone", "AMC", 20000), ("2023-02-05", "9", "Gone", "AMC", 30000),
        ("2026-07-05", "10", "", "LIC", 1000),
    ]
    return pd.DataFrame([dict(zip(["date", "voucher_no", "customer", "product_line", "amount"], r)) for r in rows])


def test_white_space():
    d, _ = e.prep_sales(_small_sales())
    w = e.white_space(d, AS_OF)
    assert "B" in set(w.customer) and e.WALKIN not in set(w.customer)
    row = w[(w.customer == "B") & (w.missing_line == "AMC")].iloc[0]
    assert row.opportunity == 20000.0                    # trailing-12m AMC buyers: A 10000, C 30000 -> avg 20000
    assert not ((w.customer == "A") & (w.missing_line == "LIC")).any()    # already owns it
    assert w.iloc[0].customer == "D" or w.iloc[0].lifetime_value == w.lifetime_value.max()


def test_winback():
    d, _ = e.prep_sales(_small_sales())
    w = e.winback(d, AS_OF, years=[2021, 2022, 2023, 2024, 2025])
    names = w.customer.tolist()
    assert "Gone" in names and "OLD CLOSED" not in names and "A" not in names and e.WALKIN not in names
    assert w.set_index("customer").loc["Gone", "best_full_year"] == 30000.0   # best single year, not lifetime 50000


def test_winback_15_month_boundary():
    """14 months quiet is NOT win-back; 16 months quiet is. Guards the 15-month rule."""
    def mk(months_ago):
        dt = (pd.Timestamp(AS_OF) - pd.DateOffset(months=months_ago)).date().isoformat()
        rows = [(dt, "1", "Quiet", "X", 5000), ("2024-04-10", "2", "Anchor", "X", 100), ("2026-06-01", "3", "Anchor", "X", 100)]
        return pd.DataFrame([dict(zip(["date", "voucher_no", "customer", "product_line", "amount"], r)) for r in rows])
    d14, _ = e.prep_sales(mk(14))
    d16, _ = e.prep_sales(mk(16))
    assert "Quiet" not in e.winback(d14, AS_OF).customer.tolist()
    assert "Quiet" in e.winback(d16, AS_OF).customer.tolist()
    assert "Quiet" not in e.winback(d16, AS_OF, months=18).customer.tolist()


# ------------------------------------------------------------ price dispersion
def test_price_dispersion():
    ln = pd.DataFrame({"item": ["X"] * 5, "rate": [100, 100, 100, 80, 60], "qty": [1, 1, 1, 2, 3]})
    r = e.price_dispersion(ln).iloc[0]
    assert r["median"] == 100 and r["min"] == 60 and r["max"] == 100
    assert r["below_median"] == 2 and r["value_lost_to_median"] == (20 * 2 + 40 * 3)
    assert e.price_dispersion(pd.DataFrame({"item": ["Y"], "rate": [5], "qty": [0]})).empty


# ------------------------------------------------------------ receivables
def test_ageing_known():
    bal = pd.DataFrame({"customer": ["A", "B", "C", "D"], "balance": [150.0, 0.0, -20.0, 500.0]})
    bills = pd.DataFrame({"customer": ["A", "A", "B", "C", "D"], "bill_ref": list("12345"),
                          "bill_date": [AS_OF - timedelta(days=100), AS_OF - timedelta(days=10), AS_OF - timedelta(days=50), AS_OF - timedelta(days=5), AS_OF - timedelta(days=200)],
                          "pending": [100.0, 100.0, 70.0, 30.0, 300.0]})
    r = e.corrected_ageing(bal, bills, AS_OF)
    assert r["buckets"]["0-30"] == 100.0 and r["buckets"]["91-180"] == 50.0   # A: 150 newest-first
    assert r["buckets"]["Over 180"] == 300.0 and r["unallocated"] == 200.0    # D: balance beyond bills
    assert r["phantom"] == 150.0 and r["credit_balances"] == 20.0              # A 50, B 70, C 30
    assert r["phantom_by_customer"] == {"A": 50.0, "B": 70.0, "C": 30.0}
    assert r["receivable_total"] == 650.0


def test_ageing_boundaries():
    for age, lab in [(0, "0-30"), (30, "0-30"), (31, "31-60"), (60, "31-60"), (61, "61-90"), (90, "61-90"), (91, "91-180"), (180, "91-180"), (181, "Over 180")]:
        assert e._bucket(age) == lab


def test_ageing_bills_without_balance_row_and_future_dates():
    bal = pd.DataFrame({"customer": ["A"], "balance": [10.0]})
    bills = pd.DataFrame({"customer": ["A", "Z"], "bill_ref": ["1", "2"], "bill_date": [AS_OF + timedelta(days=3), AS_OF], "pending": [10.0, 99.0]})
    r = e.corrected_ageing(bal, bills, AS_OF)
    assert r["future_dated_bills"] == 1 and r["buckets"]["0-30"] == 10.0 and r["phantom"] == 99.0


def test_ageing_empty():
    r = e.corrected_ageing(pd.DataFrame({"customer": [], "balance": []}), pd.DataFrame({"customer": [], "bill_ref": [], "bill_date": [], "pending": []}), AS_OF)
    assert r["receivable_total"] == 0 and sum(r["buckets"].values()) == 0


def test_dso_and_release():
    assert e.dso(100, 1000) == 36.5 and e.dso(100, 0) is None and e.dso(100, -5) is None
    assert abs(e.cash_released(3650, 15) - 150) < 1e-9 and e.cash_released(-100, 15) == 0


def test_dso_basis_warning():
    one = {"date": ["2026-06-01"], "voucher_no": ["1"], "customer": ["A"], "product_line": ["X"], "amount": [1000.0]}
    d, _ = e.prep_sales(pd.DataFrame({**one, "gross_amount": [1180.0]}))
    v, basis = e.trailing_gross(d, AS_OF)
    assert v == 1180.0 and "gross" in basis
    d2, _ = e.prep_sales(pd.DataFrame(one))
    v, basis = e.trailing_gross(d2, AS_OF)
    assert v == 1000.0 and "overstated" in basis


# ------------------------------------------------------------ cost
def test_breakeven():
    assert e.breakeven(600000, 0.3) == 2000000 and e.breakeven(1, 0) is None and e.breakeven(1, -0.1) is None


def test_payments():
    p = pd.DataFrame({"date": ["2026-01-01", "2026-01-05", "2026-03-01", "2026-02-01", "2026-03-03", "2026-04-02"],
                      "supplier": ["S", "S", "S", "R", "R", "R"], "amount": [500, 500, 500, 1000, 1000, 1000]})
    dup = e.duplicate_payments(p)
    assert len(dup) == 1 and dup.iloc[0]["supplier"] == "S"
    assert e.recurring_payments(p)["supplier"].tolist() == ["R"]


def test_spikes():
    months = pd.period_range("2025-01", "2026-03", freq="M")
    rows = [{"date": m.to_timestamp(), "head": "Rent", "amount": 100.0} for m in months]
    rows[-1]["amount"] = 400.0
    sp = e.spikes(pd.DataFrame(rows))
    assert sp["month"].tolist() == ["2026-03"] and abs(sp.iloc[0]["ratio"] - 4.0) < 1e-9
    assert e.spikes(pd.DataFrame(rows[:5])).empty          # too little history


# ------------------------------------------------------------ forecast
def test_forecast_conservation_and_lowest():
    st = date(2026, 11, 5)
    rec = pd.DataFrame({"due_date": [st - timedelta(days=10), st + timedelta(days=3), st + timedelta(days=40), st + timedelta(days=200)], "amount": [100.0, 200.0, 300.0, 999.0]})
    pay = pd.DataFrame({"due_date": [st + timedelta(days=1), st + timedelta(days=15)], "amount": [500.0, 100.0]})
    r = e.forecast_13w(50.0, rec, pay, st, 0.5)
    w = r["weeks"]
    assert w["inflow"].tolist()[0] == 150.0                # (100 overdue + 200) * 0.5
    assert abs((50 + w["inflow"].sum() - w["outflow"].sum()) - w["closing"].iloc[-1]) < 1e-9
    assert r["overdue_receipts"] == 100.0 and r["lowest_week"] == 3 and r["lowest_cash"] == -400.0
    assert w["inflow"].sum() == (100 + 200 + 300) * 0.5    # the 200-day receipt is beyond 13 weeks
    with pytest.raises(ValueError):
        e.forecast_13w(0, rec, pay, st, 1.2)
    assert e.forecast_13w(10.0, None, None, st, 1.0)["lowest_cash"] == 10.0


# ------------------------------------------------------------ industries
def test_industry_config():
    assert len(e.INDUSTRIES) == 12 and "NGO / non-profit" in e.INDUSTRIES
    for k, v in e.INDUSTRIES.items():
        assert len(v["kpis"]) >= 1 and len(set(v["kpis"])) == len(v["kpis"]) and v["sub"]
    assert any("Programme" in k for k in e.INDUSTRIES["NGO / non-profit"]["kpis"])
    assert e.ENTITY_SCALES == ["MSME", "Mid-scale", "Large", "NGO / non-profit"]


# ------------------------------------------------------------ gated module
def _pl(sales):
    d, _ = e.prep_sales(sales)
    return {int(y): round(g.amount.sum(), 2) for y, g in d.groupby("fy")}


def test_module_ok_and_lineage_everywhere():
    sales = s.make_sales(seed=3)
    m = e.build_revenue_module(sales, AS_OF, FROM, _pl(sales), s.RECURRING, "2026-11-05 06:00")
    assert m["status"] == "OK"
    n = 0
    for k, v in m.items():
        if isinstance(v, dict) and "lineage" in v:
            n += 1
            assert all(v["lineage"][f] not in (None, "") for f in ("source", "period", "synced_at", "excluded"))
    assert n == 7
    assert m["tiers"]["value"]["share_of_revenue"].sum() == pytest.approx(1.0)


def test_module_withholds_numbers_when_tie_fails():
    sales = s.make_sales(seed=4)
    pl = _pl(sales)
    pl[2023] += 1.50
    m = e.build_revenue_module(sales, AS_OF, FROM, pl, s.RECURRING, "t")
    assert m["status"] == "DATA CHECK FAILED" and "revenue_by_line_year" not in m and "withheld" in m["message"]
    assert [c for c in m["checks"] if not c["pass"]][0]["variance"] == -1.5


def test_module_fails_closed_with_no_pl():
    assert e.build_revenue_module(s.make_sales(seed=4), AS_OF, FROM, {}, s.RECURRING, "t")["status"] == "DATA CHECK FAILED"


# ------------------------------------------------------------ FUZZ: invariants over many random books
@pytest.mark.parametrize("seed", range(300))
def test_fuzz_invariants(seed):
    rnd = random.Random(seed)
    sales = s.make_sales(seed=seed, n_customers=rnd.randint(1, 80), n_rows=rnd.randint(1, 400), credit_note_rate=rnd.random() * 0.3)
    d, _ = e.prep_sales(sales)
    assert abs(d.amount.sum() - e.revenue_by_line_year(d).to_numpy().sum()) < 0.05      # independent cuts agree
    assert abs(d.amount.sum() - e.customer_totals(d).sum()) < 0.05
    tot = e.customer_totals(d)
    t = e.abc_tiers(tot)
    assert len(t) == int((tot > 0).sum()) and t.customer.is_unique
    if len(t):
        assert t.revenue.sum() == pytest.approx(tot[tot > 0].sum())
        assert t.tier.iloc[0] == "GOLD" and set(t.tier) <= {"GOLD", "SILVER", "BRONZE"}
        assert t.tier.map({"GOLD": 0, "SILVER": 1, "BRONZE": 2}).is_monotonic_increasing
        assert t.loc[t.tier == "GOLD", "revenue"].sum() / t.revenue.sum() >= 0.70 - 1e-9   # gold covers at least 70%
        assert e.tier_summary(t).share_of_revenue.sum() == pytest.approx(1.0)
    r = e.retention(d, e.full_fys(FROM, AS_OF))
    for _, row in r.iterrows():
        assert row.active == row.retained + row.new + row.reactivated
        if row.logo_retention is not None:
            assert 0 <= row.logo_retention <= 1 and 0 <= row.grr <= 1 + 1e-12 and row.nrr >= 0
    x = e.lfl(d, AS_OF)
    assert x["current"]["days"] == (AS_OF - date(2026, 4, 1)).days + 1 and x["prior"]["to"] == date(2025, 11, 5)
    w = e.white_space(d, AS_OF)
    life = d.groupby(["customer", "product_line"]).amount.sum()
    for _, row in w.iterrows():
        assert life.get((row.customer, row.missing_line), 0) <= 0 and row.customer != e.WALKIN and row.opportunity > 0


@pytest.mark.parametrize("seed", range(200))
def test_fuzz_receivables_conservation(seed):
    bal, bills = s.make_receivables(seed=seed, n=random.Random(seed).randint(1, 60))
    r = e.corrected_ageing(bal, bills, AS_OF)
    pos = bal[bal.balance > 0].balance.sum()
    assert abs(sum(r["buckets"].values()) + r["unallocated"] - pos) < 0.05          # every ledger rupee lands once
    assert abs(r["receivable_total"] - pos) < 0.05
    assert abs(sum(r["buckets"].values()) + r["phantom"] - bills.pending.sum()) < 0.05   # every open-bill rupee: bucket or phantom
    assert all(v >= 0 for v in r["buckets"].values()) and r["phantom"] >= 0 and r["unallocated"] >= 0


@pytest.mark.parametrize("seed", range(100))
def test_fuzz_forecast(seed):
    rnd = random.Random(seed)
    st = date(2026, 11, 5)

    def mk(n):
        return pd.DataFrame({"due_date": [st + timedelta(days=rnd.randint(-30, 150)) for _ in range(n)], "amount": [round(rnd.uniform(1, 9999), 2) for _ in range(n)]})
    rec, pay = mk(rnd.randint(0, 30)), mk(rnd.randint(0, 30))
    cr = rnd.random()
    r = e.forecast_13w(rnd.uniform(-1000, 50000), rec, pay, st, cr)
    w = r["weeks"]
    assert len(w) == 13
    assert (w["opening"].iloc[1:].to_numpy() == w["closing"].iloc[:-1].to_numpy()).all()
    assert r["lowest_cash"] == w["closing"].min()
    end = pd.Timestamp(st) + pd.Timedelta(days=91)
    exp_in = sum(a for d_, a in zip(pd.to_datetime(rec.due_date), rec.amount) if d_ < end) * cr if len(rec) else 0
    assert abs(w["inflow"].sum() - exp_in) < 0.1


# ------------------------------------------------------------ ADVERSARIAL / burn tests
def test_adversarial_inputs():
    weird = pd.DataFrame({"date": ["2024-05-01", "2024-05-01", "1999-01-01", "2040-01-01"], "voucher_no": ["\u00e9", "\u00e9", "x", "y"],
                          "customer": ["\u0924\u0947\u0938\u094d\u091f  \u0932\u093f\u092e\u093f\u091f\u0947\u0921", "<script>alert(1)</script>", "A" * 5000, "  "],
                          "product_line": ["L", "L", "L", "L"], "amount": [1e13, -1e13, 0.0, 5.0]})
    d, _ = e.prep_sales(weird)
    assert len(d) == 4 and e.WALKIN in set(d.customer)
    m = e.build_revenue_module(weird, AS_OF, FROM, {2024: 0.0, 1998: 0.0, 2039: 5.0}, {"L"}, "t")
    assert m["status"] == "OK"
    assert e.fmt_inr(1e13).startswith("\u20b9") and e.fmt_inr_short(1e13).endswith("Cr")


def test_empty_inputs_do_not_crash():
    empty = pd.DataFrame(columns=["date", "voucher_no", "customer", "product_line", "amount"])
    d, _ = e.prep_sales(empty)
    assert e.abc_tiers(e.customer_totals(d)).empty and e.white_space(d, AS_OF).empty
    assert e.winback(d, AS_OF).empty and e.lfl(d, AS_OF)["current"]["revenue"] == 0
    assert e.retention(d, [2022]).empty


def test_all_credit_notes():
    df = pd.DataFrame({"date": ["2025-06-01"] * 2, "voucher_no": ["1", "2"], "customer": ["A", "B"], "product_line": ["X"] * 2, "amount": [-100.0, -50.0]})
    d, _ = e.prep_sales(df)
    assert e.abc_tiers(e.customer_totals(d)).empty and e.top_n_share(e.customer_totals(d), 5) is None


def test_duplicate_voucher_flagged_not_dropped():
    df = pd.DataFrame({"date": ["2025-06-01"] * 2, "voucher_no": ["1", "1"], "customer": ["A", "A"], "product_line": ["X"] * 2, "amount": [100.0, 100.0]})
    d, issues = e.prep_sales(df)
    assert len(d) == 2 and any("duplicate" in i for i in issues)


def test_regression_grr_never_negative_with_credit_notes():
    rows = [("A", 2021, 100.0), ("A", 2022, -40.0), ("B", 2021, 100.0), ("B", 2022, 100.0)]
    df = pd.DataFrame([{"date": date(y, 6, 1), "voucher_no": f"{c}{y}", "customer": c, "product_line": "X", "amount": a} for c, y, a in rows])
    d, _ = e.prep_sales(df)
    r = e.retention(d, [2022]).iloc[0]
    assert r["grr"] == 0.5 and r["nrr"] == 0.5 and r["churned"] == 1


def test_regression_retention_no_base_is_none_not_nan():
    df = pd.DataFrame([{"date": date(2022, 6, 1), "voucher_no": "1", "customer": "A", "product_line": "X", "amount": 10.0},
                       {"date": date(2023, 6, 1), "voucher_no": "2", "customer": "B", "product_line": "X", "amount": 10.0}])
    d, _ = e.prep_sales(df)
    r = e.retention(d, [2022, 2023])
    assert r.loc[r.fy == 2023, "logo_retention"].iloc[0] == 0.0
    d2 = d[d.fy == 2023]
    r2 = e.retention(d2, [2023])
    assert r2.empty or r2["logo_retention"].iloc[0] is None


def test_regression_ageing_with_no_bills_at_all():
    r = e.corrected_ageing(pd.DataFrame({"customer": ["A"], "balance": [100.0]}), pd.DataFrame(), AS_OF)
    assert r["unallocated"] == 100.0 and r["phantom"] == 0.0 and sum(r["buckets"].values()) == 0
    with pytest.raises(ValueError):
        e.corrected_ageing(pd.DataFrame({"customer": ["A"], "balance": [1.0]}), pd.DataFrame({"customer": ["A"]}), AS_OF)
