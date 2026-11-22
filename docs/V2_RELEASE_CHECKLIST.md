# V2 — Release Checklist

## 1. Local correctness

- [ ] Run `python -m pytest tests/ -v` after final changes and verify **zero failures**.
- [ ] Run `python -m src.release_audit` and inspect `outputs/release/v2_release_audit.csv` for any `FAIL` rows.
- [ ] Confirm `requirements.txt` includes all application and ML imports, especially `pandas`, `plotly`, `streamlit`, `numpy`, and `scikit-learn`. Keep other dependencies already used.
- [ ] Launch `python -m streamlit run app/dashboard_v2.py` from the project root.
- [ ] Confirm both dark and light mode, date inputs, and the sidebar work without clipping.

## 2. Manual feature audit

- [ ] **Dashboard:** cash totals, attendance KPIs and month labels render correctly.
- [ ] **Clients:** 30/60/90-day activity, categories and charts are readable (without underscores in display labels).
- [ ] **Revenue / Memberships:** completed cash, refunds, payment status and expiration analysis display as intended.
- [ ] **Sessions / Trainers:** attendance trends and trainer comparisons; trainer name customization and filters preserve IDs.
- [ ] **Data Import:** upload all five CSVs; validate and map unfamiliar columns; activate imported data; return to sample data.
- [ ] **Business Reports:** successfully download the intended report and confirm it contains selected data.
- [ ] **Business Indicators:** separate sidebar tab and descriptive labels, not prediction probabilities.
- [ ] **Predictive Lab:** no individual churn scores or live forecasts; research results clearly identified as external/historical.
- [ ] **Missing files:** when research CSVs are absent, Predictive Lab explains the limitation rather than crashing or inventing values.
- [ ] **Import isolation:** switching to an imported business never causes the sample-business backtest to be labeled as a personalized forecast.

## 3. Public-repository review

- [ ] Preserve V1 on `main` and upload V2 changes to `v2-development`.
- [ ] Add updated `README.md` and `docs/` files, plus source code and tests that changed.
- [ ] **Never upload** `.env`, tokens, real client data, uploaded CSVs, local credential files, or unreviewed raw research datasets.
- [ ] Inspect `gitignore` rules; `outputs/`, `data/raw/`, `data/private/`, and `data/imported/` should remain excluded from public commits.
- [ ] Inspect screenshots for identifiable information; add **new V2 screenshots** if desired, e.g. `screenshots/v2_overview.png` and `screenshots/v2_predictive_lab.png`. Do not relabel existing V1 screenshots.
- [ ] Note that files generated under `outputs/` will **not** be published just because you upload source code.
- [ ] Verify the updated branch visibly contains `app/dashboard_v2.py`, `src/ml/`, and the V2 test files.

## 4. Streamlit deployment

- [ ] Keep the existing V1 app pointing to V1 unless you intentionally want to replace it.
- [ ] For a separate V2 deployment, configure a new Streamlit app pointing at the **`v2-development` branch**, with **`app/dashboard_v2.py`** as the entry point.
- [ ] Confirm dependencies install in the cloud environment from `requirements.txt`.
- [ ] Determine how to provide **privacy-safe aggregate research reports** if Predictive Lab should display them on the hosted V2 site. Locally ignored `outputs/` files are not automatically deployed; without them, a clear missing-report notice is expected.
- [ ] Verify every tab on the public URL in both appearance modes and repeat a five-CSV import test with synthetic data.
- [ ] Confirm there is no claim of commercial-grade multi-business security or verified predictive accuracy.

## 5. Final release

- [ ] Refresh the README with the public V2 URL **only after successful deployment**.
- [ ] Record the final test count and audit results after the *last* file change.
- [ ] Commit the tested release to `v2-development` and leave V1's `main` unchanged until a deliberate release/merge decision.
- [ ] Decide whether to merge into `main`, tag a V2 version, or retain V2 as a separate public demonstration.

## Existing release-audit evidence

On November 21, 2026, the uploaded `v2_release_audit(1).csv` reported **14 PASS / 0 FAIL**, including 59 Python files parsed, dependency declarations, expected dataset headers, and aggregate experiment report availability. The user's prior screenshot confirmed 368 tests passing; a *fresh* test run is still required after any remaining edits.
