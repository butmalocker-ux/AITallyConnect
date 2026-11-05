# Tally Insights - calculation engine and test harness

Pure-Python engine that computes every client-facing number (revenue, tiers, retention, win-back, ageing, DSO, 13-week forecast, break-even, spikes) from synced Tally tables. It contains no UI and no AI. Verified on synthetic data only: run it on a real client's export before trusting any figure.

## Run
    pip install -r requirements.txt
    ./run_tests.sh            # 653 tests, then mutation test (20 planted bugs, all must be caught)

## Files
- ti_engine.py      engine (formatting, FY maths, sign convention, integrity gate, lineage, revenue, receivables, cost, forecast, industries incl. NGO)
- ti_synth.py       synthetic book generator for tests
- tests/test_engine.py   unit, fuzz (300 books), receivables conservation fuzz (200), forecast fuzz (100), adversarial, regression
- mutation_test.py  plants one bug at a time and confirms tests fail
- AGENTS.md         rules for Claude Code / Codex

## Entry point for a module
`build_revenue_module(sales_df, ...)` returns metrics with lineage, or "DATA CHECK FAILED" with no numbers when a tie fails.

## Known limits
- Cost-to-serve, headcount and customer phone numbers are not in Tally; the user must enter them.
- Newest-first bill allocation in the corrected ageing is an assumption and is labelled as such in output.
- Compliance (GST, TDS, 40A(3), 43B) is not in this engine yet (Phase 2).
