"""Deterministic SYNTHETIC test data. Not real. Used only for tests and demos; every output must be labelled DEMO."""
import random
from datetime import date, timedelta

import pandas as pd

LINES = ["LICENCE", "AMC", "CLOUD", "TSS", "SERVICES"]
RECURRING = {"AMC", "CLOUD", "TSS"}


def make_sales(seed=1, n_customers=60, start=date(2021, 4, 1), end=date(2026, 11, 5), n_rows=900, credit_note_rate=0.04):
    rnd = random.Random(seed)
    days = (end - start).days
    rows = []
    for i in range(n_rows):
        c = f"Customer {rnd.randint(1, n_customers):03d}"
        ln = rnd.choice(LINES)
        amt = round(rnd.lognormvariate(9.5, 1.0), 2)
        if rnd.random() < credit_note_rate:
            amt = -round(amt * rnd.uniform(0.1, 0.6), 2)
        dt = start + timedelta(days=rnd.randint(0, days))
        rows.append({"date": dt, "voucher_no": f"V{i:05d}", "customer": c, "product_line": ln, "amount": amt, "gross_amount": round(amt * 1.18, 2)})
    return pd.DataFrame(rows)


def make_receivables(seed=2, n=40, as_of=date(2026, 11, 5)):
    rnd = random.Random(seed)
    bal, bills = [], []
    for i in range(n):
        c = f"Customer {i + 1:03d}"
        k = rnd.randint(0, 5)
        pend = []
        for j in range(k):
            amt = round(rnd.uniform(1000, 90000), 2)
            pend.append(amt)
            bills.append({"customer": c, "bill_ref": f"B{i}-{j}", "bill_date": as_of - timedelta(days=rnd.randint(0, 500)), "pending": amt})
        mode = rnd.random()
        if mode < 0.2:
            b = 0.0                      # paid, bills not knocked off
        elif mode < 0.3:
            b = -round(rnd.uniform(100, 5000), 2)  # in credit
        elif mode < 0.4:
            b = round(sum(pend) + rnd.uniform(1, 5000), 2)  # on-account balance beyond bills
        else:
            b = round(sum(pend) * rnd.uniform(0.2, 1.0), 2)
        bal.append({"customer": c, "balance": b})
    return pd.DataFrame(bal), pd.DataFrame(bills, columns=["customer", "bill_ref", "bill_date", "pending"])
