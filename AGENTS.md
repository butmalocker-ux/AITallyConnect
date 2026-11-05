# Rules for Claude Code / Codex working in this repo (Tally Insights)

1. Every number shown to a client is computed by code in `ti_engine.py` (or a module built the same way). The AI never computes or estimates a figure. AI writes commentary only, from computed tables.
2. Money is shown in rupees with Indian grouping via `fmt_inr` / `fmt_inr_short`. No other formatter.
3. Growth and CAGR use full financial years only (1 Apr - 31 Mar). The current year is a part year and is compared like-for-like.
4. Tally sign convention: net_debit is stored negative; movement = net_debit + net_credit. Proven per client at onboarding (`sign_convention_ok`).
5. Every module goes through `gate()`. If any integrity tie fails (or there are no checks), show "DATA CHECK FAILED" and no numbers.
6. Every metric carries `lineage()` (source, period, rows, exclusions, synced_at). It feeds the hover tooltip.
7. Read-only on Tally. The sync agent never writes to Tally.
8. Tenant isolation: a query without a tenant id must fail, never return all tenants.
9. Compliance rules are versioned by period (Income-tax Act 1961 vs 2025 numbering). Never hard-code a section number without a version.
10. Definition of done for any change: `./run_tests.sh` passes, including the mutation test with kill rate 100%. Add a test for every bug you fix and a mutant for every new rule.
