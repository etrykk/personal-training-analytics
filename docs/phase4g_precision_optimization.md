# V2 — Phase 4G: Experimental Churn Precision Optimization

## Goal

Improve the **precision of research-only churn classification**, while aiming for at least 60% recall. The original Phase 4E Logistic Regression achieved **44.9% precision and 80.4% recall** on a DIFFERENT 75/25 retrospective split; the new results will not be directly comparable to those numbers.

This is **not** a business-specific churn predictor. The public gym dataset lacks verified feature measurement dates and 30-day cancellation targets. Phase 4C training remains blocked for the small business example.

## Method

1. Use only the six pre-specified allowlisted Phase 4E features, never outcome proxies or future-looking fields.
2. Stratified random **60% training / 20% validation / 20% untouched test** (the dataset has no suitable timestamps for temporal validation).
3. Fit six Logistic Regression configurations, using two class-weight choices (`None`, `balanced`) and three `C` values (0.1, 1, 10). Imputation and scaling are fitted **only on training data**.
4. On the **validation partition only**, evaluate thresholds from 0.30 to 0.85 in 0.05 increments. Select the highest validation precision among configurations achieving validation recall >=0.60; use F1 and recall to resolve ties.
5. Assess the selected model **once** on the final untouched test partition. Compare with a majority-class baseline and the default balanced Logistic Regression at 0.50 **on that same test partition**.
6. Report precision, recall, F1, balanced accuracy, ROC-AUC, confusion-matrix counts, and Wilson confidence intervals. Optimization may or may not improve test precision.

**Caution:** selecting settings on one validation sample can itself overfit that sample. The final test remains untouched for a cleaner estimate, but even good test results would not validate prospective churn predictions, business deployment, calibration, or fairness.

## Files

- `src/ml/churn_precision.py` — research-only tuning and aggregate reports.
- `tests/test_churn_precision.py` — 19 tests, including checks that test labels never affect the selected validation threshold.
- This documentation in `docs/phase4g_precision_optimization.md`.

The original external CSV is **not included**. Keep your existing `data/external/gym_churn_us.csv` from Phase 4E.

## Run in Windows Anaconda Prompt

```cmd
cd /d "C:\Users\Ethan\Downloads\personal-training-analytics"
python -m pytest tests/test_churn_precision.py -v
python -m pytest tests/ -v
python -m src.ml.churn_precision
```

Results are saved at `outputs/predictive/precision_optimization/`:

- `churn_precision_checks.csv`
- `churn_precision_metrics.csv`
- `churn_precision_thresholds.csv` (validation-only threshold curve)
- `churn_precision_decision.csv`

If the external dataset is missing, the reports will clearly report `external_dataset_missing` instead of showing fake results. No row-level client data, probabilities, or fitted model files are exported. Do **not** upload the external dataset, outputs, or real client information to public GitHub.

## Subsequent work

After reviewing the actual report results, continue to V2 Phase 5 (experimental revenue forecasting), then dashboard integration, regression tests, README/deployment review and a final V2 release. Keep the main business analytics as the platform's focus.
