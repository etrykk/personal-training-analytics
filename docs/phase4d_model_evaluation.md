# Personal Training Analytics — V2 Phase 4D

## Purpose

Evaluate **aggregate** churn-classification reports from Phase 4C without retraining models or exporting any individual client's data. Churn means a **recorded membership cancellation within the next 30 days**, not inactivity, package expiration, or nonrenewal.

The model is **experimental**. Passing a screening threshold is not proof of accuracy or commercial reliability.

## Files

- `src/ml/evaluation_review.py`: Validates Phase 4C outputs, computes baseline comparisons and confidence intervals, and creates evaluation decisions.
- `tests/test_evaluation_review.py`: Tests the blocked workflow, example model metrics, safety checks, export behavior, and consistency checks.

## Workflow

Run Phase 4C first to generate `outputs/predictive/classification/`:

```cmd
cd /d "C:\Users\Ethan\Downloads\personal-training-analytics"
python -m src.ml.churn_classifier --cutoff 2026-09-01
```

Run the Phase 4D tests:

```cmd
cd /d "C:\Users\Ethan\Downloads\personal-training-analytics"
python -m pytest tests/test_evaluation_review.py -v
python -m pytest tests/ -v
```

Generate evaluation reports:

```cmd
cd /d "C:\Users\Ethan\Downloads\personal-training-analytics"
python -m src.ml.evaluation_review
```

Results are saved to `outputs/predictive/evaluation/`:

1. `churn_evaluation_decision.csv` — screening status and whether model metrics are available.
2. `churn_evaluation_checks.csv` — initial targets: recall >= 60%, precision >= 30%, and improvements over the baseline in F1 and balanced accuracy.
3. `churn_evaluation_comparison.csv` — aggregate model metrics, performance changes vs. baseline, and approximate 95% Wilson intervals for recall and precision. Empty when training was blocked.

## Current dataset result

For the sample data evaluated at the September 1, 2026 cutoff, the Phase 4C report has 34 training snapshots and 43 testing snapshots. It has too few distinct cancellation memberships for either meaningful training or a stable held-out evaluation. **The classifier was not fitted**, and there is no valid model accuracy or recall to report.

Phase 4D accordingly writes a blocked decision, unevaluated metric checks, and an empty comparison report. That is expected and correct.

## Interpretation

- A 60% recall goal is an *exploratory* target, not approval to deploy.
- Precision and F1 must also be evaluated, and the model should outperform a majority-class baseline.
- Confidence intervals can be wide when the test set has few actual cancellations.
- Independent historical outcomes, time-aware validation, probability calibration, and real-world validation remain necessary.
- **`business_predictions_enabled` is always False** in Phase 4D. The dashboard should continue providing descriptive client engagement signals instead.
- The sample tests include made-up examples to check the software's behavior. They are **not** evidence that the model works on real fitness-business data.

## Next phase

Locate a legally usable experimental dataset with recorded timestamps and confirmed cancellation outcomes. Verify provenance and definitions before constructing historical snapshots. Keep external experimentation separate from any future business's individual models.
