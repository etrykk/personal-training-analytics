# V2 — Methods and Limitations

## Purpose and scope

PT Analytics Studio V2 is primarily a **descriptive business analytics** application. It calculates cash, activity, retention, session and trainer metrics from standardized data. The Predictive Lab demonstrates reproducible machine-learning research, but it does not issue individual churn predictions or approved future revenue forecasts.

## Data sources

- **Synthetic project business data:** Five linked datasets—clients, memberships, packages, sessions, and payments—in `data/processed/` (with sample data under `data/sample/`).
- **Imported business CSVs:** Five datasets accepted through Streamlit in the current browser session; columns can be mapped and validated. Session imports do not overwrite stored sample CSVs.
- **External gym membership dataset:** Used for a separate retrospective churn classification experiment. Its provenance and the timing of individual features may not establish prospective prediction validity. Do not merge it with client information from another business.

## Descriptive analytics and definitions

- **Net cash collected** uses completed payments less completed refunds. It is neither accounting profit nor necessarily accrued revenue.
- **Membership retention** refers to continuing membership contracts in eligible cohorts; it is not interchangeable with overall client retention or attendance continuation.
- **Engagement labels** are deterministic classifications based on observed session activity. Low engagement is not proof of cancellation or an individual churn probability.
- **Training sessions** are filtered by recorded status; completed sessions are distinct from canceled bookings and no-shows.
- **Monthly cash comparisons** distinguish completed months from partial months; incomplete months are not presented as like-for-like full-month totals.

## Churn research methodology

1. The internal business dataset has too few independently observed cancellations for reliable business-specific model fitting. The readiness checks prevent activation.
2. Historical membership snapshots define a candidate observation date, prediction horizon and outcome; future information is excluded from features. Unobservable outcomes are censored.
3. The external gym experiment compares Logistic Regression with a majority-class baseline and reports held-out accuracy, precision, recall, F1, balanced accuracy and ROC-AUC.
4. Precision optimization uses **separate training, validation and final test partitions**. The threshold is selected using validation data, not the reserved test set.
5. Different data splits can produce different scores; do not compare numbers across splits as if the samples were identical.

**Observed experimental results:** initial holdout precision ≈44.9% and recall ≈80.4%; separately optimized holdout precision ≈49.6% and recall ≈65.1%. A high recall can still create many false-positive outreach recommendations. These scores do not validate a 30-day individualized cancellation prediction model.

**Limitations:** possible feature timing ambiguity, dataset shift between gyms and personal-training businesses, class imbalance, synthetic or unverified provenance, and insufficient labeled business events. Models remain research-only.

## Revenue forecasting methodology

1. Construct monthly net cash totals from completed payment and refund transactions.
2. Exclude the uncertain first month and incomplete reporting month.
3. Compare a last-month baseline, three-month mean, and six-month linear trend using only past observations.
4. Rank methods on development-history validation and **freeze the selected model** before scoring the six reserved target months (May–October 2026).
5. Report MAE, RMSE, and historical one- to three-month backtests. Holdout results assess a preselected method; they are not used to pick a new winner.

**Observed results:** last-month baseline won on development data (development MAE ≈$691); holdout MAE ≈$2,202 and RMSE ≈$2,744. The linear trend's lower holdout errors are documented but do not change pre-holdout selection.

**Limitations:** 33 complete usable monthly points in this sample, six held-out target months, overlapping forecast horizons, structural changes and insufficient evidence for dependable prediction intervals. Live forecasts are disabled.

## Reporting and security boundaries

- The research views display **aggregate metrics only** and do not present client-specific model outputs.
- Model training and prediction are not triggered by dashboard navigation.
- Generated outputs are ignored by Git and must not be assumed present on hosted servers.
- Local source checks cannot replace credential scanning, secure deployment configuration, access control or a privacy review before processing real customer data.
- V2 is not a secure multi-tenant business product and does not implement external app/API synchronization.

## Reproduce the studies locally

From the project directory, using Anaconda Prompt:

```cmd
cd /d "C:\Users\Ethan\Downloads\personal-training-analytics"
python -m pytest tests/ -v
python -m src.ml.churn_readiness --as-of 2026-11-20
python -m src.ml.historical_churn --as-of 2026-11-20
python -m src.ml.external_churn_experiment
python -m src.ml.churn_precision
python -m src.ml.revenue_forecast_readiness --as-of 2026-11-20
python -m src.ml.revenue_forecast_evaluation
```

The external research commands require the separately obtained dataset at `data/external/gym_churn_us.csv`. See the corresponding phase-specific documentation for detailed methodology and input requirements. The audit command is `python -m src.release_audit`.
