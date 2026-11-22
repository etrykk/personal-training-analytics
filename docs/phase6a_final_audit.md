# V2 Phase 6A — Release Functionality Audit

This phase does not change `dashboard_v2.py` or the business analytics models.

## Automated checks

Run from **Anaconda Prompt**:

```cmd
cd /d "C:\Users\Ethan\Downloads\personal-training-analytics"
python -m pytest tests/test_release_audit.py -v
python -m pytest tests/ -v
python -m src.release_audit
```

Review `outputs/release/v2_release_audit.csv`.

- `PASS` = check succeeded.
- `WARNING` = investigate before deployment; sometimes expected (e.g. generated research reports aren't present locally).
- `FAIL` = required code or dataset structure is missing or invalid; address before release.

**Limitations:** This audit checks expected paths, Python syntax, processed CSV headers, dependency declarations, and common local credential filenames. It does **not** assess business correctness, malicious code, deployment security, missing secrets in source code, or real-world model validity. A passing audit is not a security certification. Do not upload private client data, model artifacts, local generated outputs, credentials, or third-party datasets without appropriate permissions.

## Manual browser checklist

1. **Overview:** verify KPI values, monthly cash chart, partial-month labels, and date range controls.
2. **Clients:** verify the 30/60/90-day activity table, engagement categories, client search, and chart labels (no underscores).
3. **Revenue:** compare completed payments minus refunds with the displayed net cash; confirm partial months are distinguished.
4. **Sessions:** verify attendance and no-show percentages and weekday charts.
5. **Memberships:** check cancellations, retention definition, and empty expiry watchlist behavior.
6. **Trainers:** filter trainers, edit a display name, and confirm that trainer IDs/financial totals remain unchanged.
7. **Data Import:** upload *non-sensitive test CSVs*, verify column mapping, error feedback, switching Sample/Imported Data, and that an invalid import does not replace the approved dataset.
8. **Business Reports:** export a report based on each approved source; verify the CSV/ZIP opens and references the correct source.
9. **Business Indicators:** check readable alerts and that they are not represented as verified churn predictions.
10. **Predictive Lab:** confirm churn results are described as external retrospective research; business-specific predictions stay disabled; revenue graphs are historical and no live forecasts are claimed.
11. **Theme and layout:** switch light/dark, check sidebar alignment, narrow the browser, confirm no text clipping at top, verify graph labels have spaces.
12. **Privacy check:** ensure no genuine client data, API keys, `.env`, Kaggle redistribution-restricted files, or private `outputs/` have been committed to public GitHub.

## Release note

V2 should be released only after automated tests and manual checks are reviewed. Predictions remain experimental. Dashboard visuals that use locally generated CSV research reports may be missing on the hosted Streamlit app if those reports have not been made available there via an approved, non-sensitive release process.
