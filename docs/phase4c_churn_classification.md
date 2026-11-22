# V2 Phase 4C — Experimental Membership Cancellation Classification

## Objective

Evaluate a majority-class dummy classifier and logistic regression on **historically known 30-day membership cancellation outcomes**, but only when enough independent events are available to support a meaningful split. The existing V1/V2 project sample is **not approved for model training**.

## Inputs

- `outputs/predictive/historical/historical_churn_snapshots.csv` from Phase 4B
- Historical labels: `1` = recorded cancellation within the next 30 days; `0` = no recorded cancellation in the full observed window; missing = censored
- Eight historical numerical features defined in `src/ml/historical_churn.py`

## Precautions

1. **Temporal validation:** Models train only on snapshots whose outcome windows end strictly before the holdout cutoff. The holdout contains later snapshots.
2. **Client isolation:** If a client appears in the holdout, all their historical rows are removed from training. This can reduce data substantially.
3. **Outcome definitions:** Membership expiry, inactivity, and prepaid-package nonrenewal are not labeled cancellations.
4. **No future data in features:** Only point-in-time features from Phase 4B are used. IDs, final status, label reasons, outcome dates, and future data are excluded.
5. **Insufficient events:** A hard readiness gate requires >=30 distinct cancellation memberships in training, >=10 in holdout, and at least 30/10 non-cancellation memberships respectively. This is a *screening floor*, not a claim of statistical reliability.
6. **No misleading output:** When a gate fails, model fitting is skipped and the metrics report is empty. No client-specific scores or trained model files are exported.
7. **No tuning on holdout:** The positive classification threshold is fixed at 0.5 in this early experiment. Tuning may be added later using a separate validation set.
8. **Synthetic examples only:** Tests train on handcrafted synthetic examples to validate *software behavior*. Their metrics are **not business performance estimates**.

## Evaluation metrics

Accuracy, balanced accuracy, precision, recall, F1, ROC-AUC, average precision, and confusion-matrix counts. Compare logistic regression to the majority-class baseline. The user's exploratory goal of at least 60% recall is **not a minimum approval criterion by itself**: false positives, data sufficiency, and baseline performance matter.

## Install

Add `scikit-learn` to the project's `requirements.txt` (the ZIP includes an updated copy). No changes to the V1 application or current V2 dashboard are required.

## Commands (Anaconda Prompt)

```cmd
cd /d "C:\Users\Ethan\Downloads\personal-training-analytics"
python -m pytest tests/test_churn_classifier.py -v
python -m pytest tests/ -v
python -m src.ml.churn_classifier --cutoff 2026-09-01
```

## Exports

`outputs/predictive/classification/` contains:

- `churn_classification_checks.csv` — readiness checks
- `churn_classification_decision.csv` — screening decision and split sizes
- `churn_classification_metrics.csv` — aggregate metrics (empty when blocked)

## Next

Phase 4D can develop a separately sourced experimental dataset and stronger validation. Keep this project dataset's individualized churn model unavailable until sufficient, independently observed events exist. A future trainer's data must never be pooled with another business's data for model training.
