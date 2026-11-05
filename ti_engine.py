"""tally_insights.engine - deterministic metrics for the Tally Insights app.

Rules: no AI, no estimates. Every public function is pure (data in, numbers out).
Amounts are rupees (float, rounded to 2 dp at the edge). Sales are positive, credit notes negative.
Ties are checked to Rs 0.01.
"""
from __future__ import annotations

import math
import re
from datetime import date, timedelta

import numpy as np
import pandas as pd

WALKIN = "CASH/WALK-IN"
TOL = 0.01
EXCLUDE_NAME = re.compile(r"\b(CLOSED|DO NOT USE)\b", re.I)
BUCKET_LABELS = ["0-30", "31-60", "61-90", "91-180", "Over 180"]

# ----------------------------------------------------------------- INR formatting


def _group_indian(intpart: str) -> str:
    if len(intpart) <= 3:
        return intpart
    head, tail = intpart[:-3], intpart[-3:]
    parts = []
    while len(head) > 2:
        parts.insert(0, head[-2:])
        head = head[:-2]
    if head:
        parts.insert(0, head)
    return ",".join(parts + [tail])


def fmt_inr(x, decimals: int = 2, style: str = "minus") -> str:
    """Indian digit grouping: 1234567.5 -> Rs 12,34,567.50 (rupee sign). style='bracket' for accountants."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    q = round(float(x), decimals)
    neg = q < 0
    ip, _, fp = f"{abs(q):.{decimals}f}".partition(".")
    body = "\u20b9" + _group_indian(ip) + ("." + fp if fp else "")
    if neg:
        return f"({body})" if style == "bracket" else "-" + body
    return body


def fmt_inr_short(x) -> str:
    """Lakh / crore shorthand. Rolls over correctly: 99,99,999 -> 1.00 Cr, not 100.00 L."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    sign = "-" if x < 0 else ""
    a = abs(float(x))
    if round(a / 1e5, 2) >= 100:  # would print 100.00 L
        return f"{sign}\u20b9{a / 1e7:.2f} Cr"
    if a >= 1e7:
        return f"{sign}\u20b9{a / 1e7:.2f} Cr"
    if a >= 1e5:
        return f"{sign}\u20b9{a / 1e5:.2f} L"
    return sign + fmt_inr(a, 0)


# ----------------------------------------------------------------- financial year


def fy_start_year(d: date) -> int:
    return d.year if d.month >= 4 else d.year - 1


def fy_label(y: int) -> str:
    return f"FY{y}-{(y + 1) % 100:02d}"


def fy_bounds(y: int):
    return date(y, 4, 1), date(y + 1, 3, 31)


def full_fys(books_from: date, as_of: date) -> list[int]:
    """Financial years fully inside [books_from, as_of]. Growth and CAGR use only these."""
    out, y = [], fy_start_year(books_from)
    while True:
        s, e = fy_bounds(y)
        if e > as_of:
            break
        if s >= books_from:
            out.append(y)
        y += 1
    return out


def _shift_year(d: date, n: int) -> date:
    try:
        return d.replace(year=d.year + n)
    except ValueError:  # 29 Feb
        return d.replace(year=d.year + n, day=28)


def lfl_windows(as_of: date):
    """Current part-year vs the same dates last year."""
    s = fy_bounds(fy_start_year(as_of))[0]
    return (s, as_of), (_shift_year(s, -1), _shift_year(as_of, -1))


# ----------------------------------------------------------------- Tally sign helpers


def movement(net_debit: float, net_credit: float) -> float:
    """Tally connector stores net_debit NEGATIVE. Period movement = net_debit + net_credit."""
    return net_debit + net_credit


def sign_convention_ok(opening: float, closing: float, net_debit: float, net_credit: float) -> bool:
    return abs(movement(net_debit, net_credit) - (closing - opening)) <= TOL


# ----------------------------------------------------------------- integrity gate


def tie(name: str, a: float, b: float, tol: float = TOL) -> dict:
    var = round(float(a) - float(b), 2)
    return {"check": name, "a": round(float(a), 2), "b": round(float(b), 2), "variance": var, "pass": abs(var) <= tol}


def gate(checks: list[dict]) -> str:
    return "OK" if checks and all(c["pass"] for c in checks) else "DATA CHECK FAILED"


def lineage(metric: str, source: str, period: str, rows: int, excluded: str, synced_at: str) -> dict:
    for k, v in dict(metric=metric, source=source, period=period, excluded=excluded, synced_at=synced_at).items():
        if not v:
            raise ValueError(f"lineage field '{k}' is empty")
    return {"metric": metric, "source": source, "period": period, "rows": int(rows), "excluded": excluded, "synced_at": synced_at}


# ----------------------------------------------------------------- sales preparation

REQUIRED = ["date", "voucher_no", "customer", "product_line", "amount"]


def normalise_customer(name) -> str:
    s = " ".join(str(name).split()) if name is not None and not (isinstance(name, float) and math.isnan(name)) else ""
    return s if s else WALKIN


def prep_sales(df: pd.DataFrame):
    """Validate and clean. Returns (clean_df, issues). Never silently drops: every exclusion is reported."""
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    d = df.copy()
    issues: list[str] = []
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d["amount"] = pd.to_numeric(d["amount"], errors="coerce")
    bad = d["date"].isna() | d["amount"].isna() | ~np.isfinite(d["amount"].astype(float))
    if bad.any():
        issues.append(f"{int(bad.sum())} rows excluded: bad date or amount")
        d = d[~bad]
    d["customer"] = d["customer"].map(normalise_customer)
    d["product_line"] = d["product_line"].fillna("UNMAPPED").astype(str)
    d["fy"] = d["date"].map(lambda t: fy_start_year(t.date()))
    if "gross_amount" not in d.columns:
        d["gross_amount"] = np.nan
    dups = d.duplicated(subset=["voucher_no", "product_line", "customer", "amount", "date"], keep=False)
    if dups.any():
        issues.append(f"{int(dups.sum())} rows look like duplicate voucher lines (kept, please review)")
    um = (d["product_line"] == "UNMAPPED").sum()
    if um:
        issues.append(f"{int(um)} rows have no product-line mapping")
    return d.reset_index(drop=True), issues


# ----------------------------------------------------------------- revenue


def revenue_by_line_year(d: pd.DataFrame) -> pd.DataFrame:
    return d.pivot_table(index="product_line", columns="fy", values="amount", aggfunc="sum", fill_value=0.0).round(2)


def yoy_growth(series: dict, full: list[int]) -> dict:
    out = {}
    for y in full:
        prev = series.get(y - 1, 0.0)
        out[y] = (series.get(y, 0.0) / prev - 1) if (y - 1 in full and prev > 0) else None
    return out


def cagr(series: dict, full: list[int]):
    if len(full) < 2:
        return None
    a, b, n = series.get(full[0], 0.0), series.get(full[-1], 0.0), full[-1] - full[0]
    return None if (a <= 0 or b <= 0 or n <= 0) else (b / a) ** (1 / n) - 1


def recurring_share(d: pd.DataFrame, recurring_lines: set) -> pd.Series:
    tot = d.groupby("fy")["amount"].sum()
    rec = d[d["product_line"].isin(recurring_lines)].groupby("fy")["amount"].sum().reindex(tot.index, fill_value=0.0)
    return (rec / tot.where(tot > 0)).round(6)


def lfl(d: pd.DataFrame, as_of: date) -> dict:
    (cs, ce), (ps, pe) = lfl_windows(as_of)

    def win(s, e):
        w = d[(d["date"] >= pd.Timestamp(s)) & (d["date"] <= pd.Timestamp(e))]
        pos = w[w["amount"] > 0]
        return {"from": s, "to": e, "days": (e - s).days + 1, "revenue": round(w["amount"].sum(), 2),
                "invoices": int(pos["voucher_no"].nunique()), "customers": int(w.groupby("customer")["amount"].sum().gt(0).sum())}

    cur, pri = win(cs, ce), win(ps, pe)
    cur["growth"] = (cur["revenue"] / pri["revenue"] - 1) if pri["revenue"] > 0 else None
    return {"current": cur, "prior": pri}


def customer_totals(d: pd.DataFrame) -> pd.Series:
    return d.groupby("customer")["amount"].sum()


def abc_tiers(totals: pd.Series, gold: float = 0.70, silver: float = 0.90) -> pd.DataFrame:
    """Pareto tiers. A customer is GOLD while cumulative share BEFORE adding them is below 70%, SILVER below 90%, else BRONZE.
    Customers with zero/negative lifetime revenue are excluded from the ranking."""
    t = totals[totals > 0].rename("revenue").reset_index()
    t.columns = ["customer", "revenue"]
    if t.empty:
        return pd.DataFrame(columns=["customer", "revenue", "cum_share", "tier"])
    t = t.sort_values(["revenue", "customer"], ascending=[False, True]).reset_index(drop=True)
    total = t["revenue"].sum()
    cum = t["revenue"].cumsum()
    before = ((cum - t["revenue"]) / total).round(9)
    t["cum_share"] = (cum / total).round(9)
    t["tier"] = np.where(before < gold, "GOLD", np.where(before < silver, "SILVER", "BRONZE"))
    return t


def tier_summary(t: pd.DataFrame) -> pd.DataFrame:
    total = t["revenue"].sum()
    g = t.groupby("tier").agg(customers=("customer", "count"), revenue=("revenue", "sum"), cutoff=("revenue", "min"))
    g["share_of_revenue"] = g["revenue"] / total
    g["avg_per_customer"] = g["revenue"] / g["customers"]
    return g.reindex(["GOLD", "SILVER", "BRONZE"]).dropna(how="all")


def top_n_share(totals: pd.Series, n: int) -> float | None:
    pos = totals[totals > 0].sort_values(ascending=False)
    return None if pos.empty else float(pos.head(n).sum() / pos.sum())


def retention(d: pd.DataFrame, years: list[int]) -> pd.DataFrame:
    rev = d.pivot_table(index="customer", columns="fy", values="amount", aggfunc="sum", fill_value=0.0)
    if rev.empty:
        return pd.DataFrame()
    rev = rev.reindex(columns=range(int(rev.columns.min()), int(rev.columns.max()) + 1), fill_value=0.0)
    first = (rev > 0).apply(lambda r: next((c for c in rev.columns if r[c]), None), axis=1)
    rows = []
    for y in years:
        if y - 1 not in rev.columns or y not in rev.columns:
            continue
        prev, cur = rev[y - 1], rev[y]
        pa, ca = prev > 0, cur > 0
        cur_fl = cur.clip(lower=0.0)   # a customer's net-negative year (credit notes) retains nothing, never less
        pn = int(pa.sum())
        retained = int((pa & ca).sum())
        rows.append({
            "fy": y, "active": int(ca.sum()), "retained": retained, "new": int((first == y).sum()),
            "reactivated": int((ca & ~pa & (first < y)).sum()), "churned": int((pa & ~ca).sum()),
            "logo_retention": retained / pn if pn else None,
            "grr": float(np.minimum(cur_fl[pa], prev[pa]).sum() / prev[pa].sum()) if pn else None,
            "nrr": float(cur_fl[pa].sum() / prev[pa].sum()) if pn else None,
        })
    out = pd.DataFrame(rows)
    for c in ("logo_retention", "grr", "nrr"):
        if c in out:
            out[c] = out[c].astype(object).where(out[c].notna(), None)
    return out


def white_space(d: pd.DataFrame, as_of: date, active_days: int = 365) -> pd.DataFrame:
    """Active customers missing a product line they have never bought; opportunity = the client's own trailing average per buying customer."""
    ts = pd.Timestamp(as_of)
    recent = d[(d["date"] > ts - pd.Timedelta(days=active_days)) & (d["date"] <= ts)]
    act = recent.groupby("customer")["amount"].sum()
    active = [c for c in act[act > 0].index if c != WALKIN]
    life = d.groupby(["customer", "product_line"])["amount"].sum()
    owned = set(life[life > 0].index)
    lr = recent.groupby(["product_line", "customer"])["amount"].sum()
    avg = lr[lr > 0].groupby(level=0).mean()
    ltv = d.groupby("customer")["amount"].sum()
    rows = [{"customer": c, "missing_line": ln, "opportunity": round(float(avg[ln]), 2), "lifetime_value": round(float(ltv[c]), 2)}
            for c in active for ln in avg.index if (c, ln) not in owned]
    out = pd.DataFrame(rows, columns=["customer", "missing_line", "opportunity", "lifetime_value"])
    return out.sort_values(["lifetime_value", "opportunity", "customer"], ascending=[False, False, True]).reset_index(drop=True)


def winback(d: pd.DataFrame, as_of: date, months: int = 15, years: list[int] | None = None) -> pd.DataFrame:
    ts = pd.Timestamp(as_of)
    cutoff = ts - pd.DateOffset(months=months)
    pos = d[d["amount"] > 0]
    last = pos.groupby("customer")["date"].max()
    life = d.groupby("customer")["amount"].sum()
    fyrev = d.pivot_table(index="customer", columns="fy", values="amount", aggfunc="sum", fill_value=0.0)
    cols = [y for y in (years or []) if y in fyrev.columns] or list(fyrev.columns)
    best = fyrev[cols].max(axis=1)
    rows = []
    for c, ld in last.items():
        if ld < cutoff and life[c] > 0 and c != WALKIN and not EXCLUDE_NAME.search(c):
            rows.append({"customer": c, "last_purchase": ld.date(), "best_full_year": round(float(best[c]), 2), "lifetime": round(float(life[c]), 2)})
    out = pd.DataFrame(rows, columns=["customer", "last_purchase", "best_full_year", "lifetime"])
    return out.sort_values(["best_full_year", "customer"], ascending=[False, True]).reset_index(drop=True)


def price_dispersion(lines: pd.DataFrame) -> pd.DataFrame:
    """lines: item, rate, qty. Value lost = sum((median - rate) * qty) over invoices below the median."""
    rows = []
    for item, g in lines[lines["qty"] > 0].groupby("item"):
        r = g["rate"].astype(float)
        med = float(np.percentile(r, 50))
        below = g[g["rate"] < med]
        rows.append({"item": item, "invoices": len(g), "min": r.min(), "p25": float(np.percentile(r, 25)), "median": med,
                     "p75": float(np.percentile(r, 75)), "max": r.max(), "below_median": len(below),
                     "value_lost_to_median": round(float(((med - below["rate"]) * below["qty"]).sum()), 2)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- receivables


def _bucket(age: int) -> str:
    for lab, hi in zip(BUCKET_LABELS, [30, 60, 90, 180]):
        if age <= hi:
            return lab
    return BUCKET_LABELS[-1]


def corrected_ageing(balances: pd.DataFrame, bills: pd.DataFrame, as_of: date) -> dict:
    """balances: customer, balance (receivable +, credit -). bills: customer, bill_ref, bill_date, pending (+).
    Each customer's real ledger balance is allocated to bills NEWEST-FIRST (an assumption, labelled as such).
    Bills beyond the balance are PHANTOM (paid, not knocked off). Balance beyond bills is UNALLOCATED."""
    ts = pd.Timestamp(as_of)
    buckets = {k: 0.0 for k in BUCKET_LABELS}
    phantom = credit = unalloc = 0.0
    phantom_by, future = {}, 0
    cols = ["customer", "bill_ref", "bill_date", "pending"]
    b = pd.DataFrame(columns=cols) if bills is None or bills.empty else bills.copy()
    miss = [c for c in cols if c not in b.columns]
    if miss:
        raise ValueError(f"bills missing columns: {miss}")
    if not {"customer", "balance"} <= set(balances.columns):
        raise ValueError("balances needs columns customer, balance")
    b["bill_date"] = pd.to_datetime(b["bill_date"])
    b["pending"] = pd.to_numeric(b["pending"])
    grouped = {c: g for c, g in b.groupby("customer")}
    for _, row in balances.iterrows():
        c, bal = row["customer"], round(float(row["balance"]), 2)
        g = grouped.get(c)
        pend_total = round(float(g["pending"].sum()), 2) if g is not None else 0.0
        if bal <= 0:
            credit += -bal
            if pend_total > 0:
                phantom += pend_total
                phantom_by[c] = pend_total
            continue
        remaining = bal
        if g is not None:
            for _, bl in g.sort_values("bill_date", ascending=False).iterrows():
                if remaining <= 0:
                    break
                take = min(float(bl["pending"]), remaining)
                age = (ts - bl["bill_date"]).days
                if age < 0:
                    future += 1
                    age = 0
                buckets[_bucket(age)] += take
                remaining -= take
        remaining = round(remaining, 2)
        unalloc += remaining
        ph = round(pend_total - (bal - remaining), 2)
        if ph > 0:
            phantom += ph
            phantom_by[c] = ph
    # bills for customers with no balance row at all are phantom too
    known = set(balances["customer"])
    for c, g in grouped.items():
        if c not in known:
            v = round(float(g["pending"].sum()), 2)
            phantom += v
            phantom_by[c] = v
    return {"buckets": {k: round(v, 2) for k, v in buckets.items()}, "unallocated": round(unalloc, 2),
            "phantom": round(phantom, 2), "credit_balances": round(credit, 2), "phantom_by_customer": phantom_by,
            "future_dated_bills": future, "receivable_total": round(float(balances.loc[balances["balance"] > 0, "balance"].sum()), 2)}


def dso(receivables: float, revenue_ttm: float, days: int = 365):
    """Receivables and revenue MUST be on the same tax basis (both GST-inclusive or both exclusive)."""
    return None if revenue_ttm <= 0 else receivables / revenue_ttm * days


def cash_released(revenue_ttm: float, days_improvement: int) -> float:
    return max(revenue_ttm, 0.0) / 365 * days_improvement


def trailing_gross(d: pd.DataFrame, as_of: date) -> tuple[float, str]:
    """Trailing-12-month revenue on the same basis as receivables (GST-inclusive) when gross_amount exists."""
    ts = pd.Timestamp(as_of)
    w = d[(d["date"] > ts - pd.Timedelta(days=365)) & (d["date"] <= ts)]
    if w["gross_amount"].notna().all() and len(w):
        return round(float(w["gross_amount"].sum()), 2), "gross (GST-inclusive)"
    return round(float(w["amount"].sum()), 2), "net (GST-exclusive) - DSO overstated if receivables include GST"


# ----------------------------------------------------------------- cost


def breakeven(fixed_cost: float, gross_margin_pct: float):
    return None if gross_margin_pct <= 0 else fixed_cost / gross_margin_pct


def duplicate_payments(pay: pd.DataFrame, days: int = 7) -> pd.DataFrame:
    p = pay.copy()
    p["date"] = pd.to_datetime(p["date"])
    p = p.sort_values(["supplier", "amount", "date"])
    gap = p.groupby(["supplier", "amount"])["date"].diff().dt.days
    return p[gap.notna() & (gap <= days)].reset_index(drop=True)


def recurring_payments(pay: pd.DataFrame, min_count: int = 3, lo: int = 25, hi: int = 35) -> pd.DataFrame:
    rows = []
    p = pay.copy()
    p["date"] = pd.to_datetime(p["date"])
    for (sup, amt), g in p.groupby(["supplier", "amount"]):
        ds = sorted(g["date"])
        if len(ds) >= min_count and all(lo <= (b - a).days <= hi for a, b in zip(ds, ds[1:])):
            rows.append({"supplier": sup, "amount": amt, "occurrences": len(ds), "last_date": ds[-1].date()})
    return pd.DataFrame(rows, columns=["supplier", "amount", "occurrences", "last_date"])


def spikes(exp: pd.DataFrame, factor: float = 1.5, min_history: int = 6) -> pd.DataFrame:
    e = exp.copy()
    e["month"] = pd.to_datetime(e["date"]).dt.to_period("M")
    rows = []
    for head, g in e.groupby("head"):
        m = g.groupby("month")["amount"].sum()
        m = m.reindex(pd.period_range(m.index.min(), m.index.max(), freq="M"), fill_value=0.0)
        for i in range(min_history, len(m)):
            prior = m.iloc[max(0, i - 12):i]
            avg = prior.mean()
            if avg > 0 and m.iloc[i] > factor * avg:
                rows.append({"head": head, "month": str(m.index[i]), "amount": float(m.iloc[i]), "avg_prior": float(avg), "ratio": float(m.iloc[i] / avg)})
    return pd.DataFrame(rows, columns=["head", "month", "amount", "avg_prior", "ratio"])


# ----------------------------------------------------------------- cash forecast


def forecast_13w(opening_cash: float, receipts: pd.DataFrame, payments: pd.DataFrame, start: date, collection_rate: float) -> dict:
    """receipts/payments: due_date, amount (positive). collection_rate (0..1) is the OWNER'S assumption, never ours.
    Items due before `start` fall into week 1 and are counted as overdue."""
    if not 0 <= collection_rate <= 1:
        raise ValueError("collection_rate must be between 0 and 1")
    wk = pd.DataFrame({"week": range(1, 14)})
    s = pd.Timestamp(start)

    def bucket(df):
        if df is None or df.empty:
            return pd.Series(0.0, index=range(1, 14)), 0.0
        idx = ((pd.to_datetime(df["due_date"]) - s).dt.days // 7 + 1).clip(lower=1)
        overdue = float(df.loc[pd.to_datetime(df["due_date"]) < s, "amount"].sum())
        idx = idx[idx <= 13]
        return df.loc[idx.index].groupby(idx)["amount"].sum().reindex(range(1, 14), fill_value=0.0), overdue

    r, r_over = bucket(receipts)
    p, p_over = bucket(payments)
    wk["inflow"] = (r.values * collection_rate).round(2)
    wk["outflow"] = p.values.round(2)
    close, bal = [], float(opening_cash)
    opens = []
    for i in range(13):
        opens.append(bal)
        bal = bal + wk.loc[i, "inflow"] - wk.loc[i, "outflow"]
        close.append(round(bal, 2))
    wk["opening"] = [round(o, 2) for o in opens]
    wk["closing"] = close
    low = wk.loc[wk["closing"].idxmin()]
    return {"weeks": wk, "lowest_week": int(low["week"]), "lowest_cash": float(low["closing"]), "overdue_receipts": r_over, "overdue_payments": p_over}


# ----------------------------------------------------------------- industries

ENTITY_SCALES = ["MSME", "Mid-scale", "Large", "NGO / non-profit"]
INDUSTRIES = {
    "Trading / distribution": {"sub": ["Wholesale", "Distributor", "Import-export"], "kpis": ["Stock turns", "Margin per SKU", "Credit days to retailers", "Supplier credit days"]},
    "Manufacturing": {"sub": ["Discrete", "Process", "Job work"], "kpis": ["Raw material cost %", "Conversion cost", "WIP value", "Scrap %"]},
    "Retail": {"sub": ["Grocery", "Apparel", "Electronics", "Pharmacy"], "kpis": ["Daily sales", "Basket value", "Stock ageing", "Cash vs UPI mix"]},
    "Restaurants / food": {"sub": ["Dine-in", "Cloud kitchen", "Catering"], "kpis": ["Food cost %", "Aggregator commission %", "Daily cash"]},
    "Services / professional": {"sub": ["CA", "Consultants", "Agencies", "IT services"], "kpis": ["Billable revenue", "Retainer share", "People cost %", "Receivable days"]},
    "Construction / real estate": {"sub": ["Contractors", "Developers"], "kpis": ["Project-wise cost", "Retention money", "Customer advances"]},
    "Healthcare": {"sub": ["Clinics", "Diagnostics", "Pharmacy"], "kpis": ["Revenue per patient", "Consumables %"]},
    "Education": {"sub": ["Schools", "Coaching"], "kpis": ["Fee collection %", "Term seasonality", "Fees in advance"]},
    "Logistics / transport": {"sub": ["Fleet", "Courier"], "kpis": ["Fuel cost %", "Per-trip margin", "Vehicle utilisation"]},
    "Jewellery / gems": {"sub": ["Retail", "Wholesale", "Manufacturing"], "kpis": ["Metal-weighted stock", "Making charges %", "Hedge position"]},
    "NGO / non-profit": {"sub": ["Trust", "Society", "Section 8 company"], "kpis": ["Programme cost %", "Admin cost %", "Restricted vs unrestricted funds", "Grant utilisation", "Donor concentration"]},
    "Other": {"sub": ["Free text"], "kpis": ["Standard KPI set"]},
}


# ----------------------------------------------------------------- module builders (gated)


def build_revenue_module(sales: pd.DataFrame, as_of: date, books_from: date, pl_revenue_by_fy: dict, recurring_lines: set, synced_at: str) -> dict:
    d, issues = prep_sales(sales)
    full = full_fys(books_from, as_of)
    checks = []
    for y in sorted(pl_revenue_by_fy):
        checks.append(tie(f"{fy_label(y)} revenue vs P&L", d.loc[d["fy"] == y, "amount"].sum(), pl_revenue_by_fy[y]))
    status = gate(checks)
    out = {"status": status, "checks": checks, "issues": issues}
    if status != "OK":
        out["message"] = "Data check failed. Figures withheld until the variance is resolved."
        return out
    by = revenue_by_line_year(d)
    tot = d.groupby("fy")["amount"].sum().to_dict()
    tiers = abc_tiers(customer_totals(d))
    period = f"{fy_label(full[0])} to {fy_label(full[-1])} + part year" if full else "part year only"
    L = lambda m, s, ex: lineage(m, s, period, len(d), ex, synced_at)
    out.update({
        "revenue_by_line_year": {"value": by, "lineage": L("Revenue by line", "Sales ledgers via ledger-account", "opening stubs, GST, round-off")},
        "growth": {"value": yoy_growth(tot, full), "lineage": L("YoY growth (full years only)", "Sales ledgers", "part year")},
        "cagr": {"value": cagr(tot, full), "lineage": L("CAGR (full years only)", "Sales ledgers", "part year")},
        "recurring_share": {"value": recurring_share(d, recurring_lines), "lineage": L("Recurring share", "Sales ledgers + mapping", "unmapped lines")},
        "like_for_like": {"value": lfl(d, as_of), "lineage": L("Like-for-like part year", "Sales ledgers", "none")},
        "tiers": {"value": tier_summary(tiers), "lineage": L("Gold/Silver/Bronze 70/20/10", "Customer totals", "customers with revenue <= 0")},
        "top10_share": {"value": top_n_share(customer_totals(d), 10), "lineage": L("Top-10 share", "Customer totals", "customers with revenue <= 0")},
    })
    return out
