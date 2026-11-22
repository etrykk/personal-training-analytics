# PT Analytics Studio — Personal Training Business Analytics (V2)

**Python business intelligence and analytics for personal trainers and fitness studios**, with optional, clearly labeled experimental machine-learning research.

V2 expands the original Personal Training Analytics Platform with a YouTube Studio-inspired Streamlit interface, advanced engagement and revenue reporting, business health indicators, and a separate Predictive Lab. **Business analytics are the primary product; churn classification and revenue forecasting remain research-only.**

> **Status:** V2 in release preparation. The `main` branch preserves V1; `v2-development` contains the ongoing V2 work. The hosted V1 app should not be assumed to serve V2 until its deployment is updated.

## Highlights

| Area | Capabilities |
| --- | --- |
| Dashboard | Business overview, monthly cash charts, client and session KPIs, interactive reporting-date and appearance controls |
| Clients | 30/60/90-day attendance, activity continuation, engagement categories, inactivity and follow-up indicators |
| Revenue & memberships | Gross payments, refunds, net cash collected, payment statuses, membership retention, expiration watchlists, monthly trends |
| Sessions & trainers | Completion/no-show trends, weekday patterns, training hours, trainer comparisons, editable trainer display names |
| Business indicators | Separate operational insights and follow-up indicators; rules-based, not churn probabilities |
| Data import | Upload five CSV datasets, map columns, validate, switch between sample and imported data; imports remain in app session memory |
| Reports | Download CSV/HTML reporting outputs and business summaries |
| Predictive Lab | Aggregate external churn research and historical revenue forecasting backtests, **not personalized predictions** |

### Dashboard structure

The Streamlit sidebar contains **Dashboard**, **Clients**, **Revenue**, **Sessions**, **Memberships**, **Trainers**, **Data Import**, **Business Reports**, **Business Indicators**, and **Predictive Lab**. The interface supports dark and light appearance options.

V1 reference screenshots remain in `screenshots/`. Updated V2 screenshots can be added during the release process; do not label the existing V1 images as V2.

## Quick start (Windows / Anaconda Prompt)

1. Open **Anaconda Prompt** and navigate to the project directory:

   ```cmd
   cd /d "C:\Users\Ethan\Downloads\personal-training-analytics"
   ```

2. Install the dependencies:

   ```cmd
   python -m pip install -r requirements.txt
   ```

3. Launch the V2 application:

   ```cmd
   python -m streamlit run app/dashboard_v2.py
   ```

4. To use the original V1 dashboard instead:

   ```cmd
   python -m streamlit run app/dashboard.py
   ```

The application reads sample data from `data/processed/`. Imported CSVs are kept only in the current Streamlit session and do not overwrite these sample files.

### Tests and release checks

```cmd
cd /d "C:\Users\Ethan\Downloads\personal-training-analytics"
python -m pytest tests/ -v
python -m src.release_audit
```

The release audit checks file presence, syntax, schema headers, dependencies and common accidental-secret patterns. It **does not** replace browser testing, security review or verification on a fresh deployment.

## Input datasets and method

The five canonical CSV datasets are `clients`, `memberships`, `packages`, `sessions`, and `payments`. `src/data_schema.py` defines required fields, identifiers, date/numeric columns, and accepted categories. `src/data_input/` handles CSV importing and column mapping; `src/validation.py` checks formatting and relationships.

The standard analytical path is:

**CSV import or processed sample → mapping and validation → descriptive analytics → business indicators → Streamlit charts/reports**

Analytics logic is separated into `src/analytics/`. Machine learning research and model-readiness guards are in `src/ml/`. Dashboard code is in `app/dashboard_v2.py`, with shared data and view helpers in `src/dashboard_data.py` and `src/dashboard_views.py`.

## Predictive Lab: experimental results only

### External gym churn classification

A separate external gym membership dataset of approximately 4,000 records was used to evaluate Logistic Regression against a majority-class baseline. In the initial retrospective holdout, Logistic Regression achieved approximately **68.7% accuracy**, **80.4% churn recall**, **44.9% precision**, and **0.791 ROC-AUC**.

Subsequent threshold and regularization research on a *different* held-out split increased precision to approximately **49.6%**, while recall fell to approximately **65.1%**. These results are **not directly interchangeable** with the initial split.

The external dataset does not establish verified timing of all features before cancellation. Accordingly, these are retrospective research metrics—not validated 30-day cancellation predictions. Individual business predictions remain **disabled** when historical outcomes or data quality are insufficient.

### Revenue forecasting backtests

The revenue experiment compares last-month persistence, three-month average, and six-month linear trend models using historical completed-month net cash. Partial months and a potentially incomplete initial month are excluded. Development-period selection is performed **before** the six-month May–October 2026 holdout is scored.

The **last-month baseline** won development selection. On the reserved holdout it had a mean absolute error (MAE) of approximately **$2,202** and root mean squared error (RMSE) of approximately **$2,744**. The linear trend scored better on that reserved period, but we do not retroactively switch winners using test results. Live revenue forecasts and dependable prediction intervals are **not enabled**.

Research CSV outputs are generated locally and read by Predictive Lab if available; if they are absent, the page displays a missing-report message rather than inventing results. See [Methods and limitations](docs/V2_METHODS_AND_LIMITATIONS.md).

## Project layout

```text
personal-training-analytics/
├── app/
│   ├── dashboard.py                  # Original V1 dashboard
│   └── dashboard_v2.py               # V2 Studio interface
├── data/
│   ├── sample/
│   ├── processed/
│   └── external/                     # Optional research dataset (local)
├── docs/
│   ├── data_dictionary.md
│   ├── kpi_definitions.md
│   ├── V2_METHODS_AND_LIMITATIONS.md
│   └── V2_RELEASE_CHECKLIST.md
├── src/
│   ├── analytics/                    # Business reporting and engagement
│   ├── data_input/                   # CSV import and column mapping
│   ├── ml/                           # Experimental models and evaluations
│   ├── dashboard_data.py
│   ├── dashboard_views.py
│   ├── data_schema.py
│   ├── validation.py
│   └── release_audit.py
├── tests/
├── screenshots/                      # Existing V1 reference images
├── requirements.txt
└── README.md
```

`outputs/` is generated locally and excluded from Git by `.gitignore`. Public deployments do not automatically receive locally generated research reports. Consult the [release checklist](docs/V2_RELEASE_CHECKLIST.md) before deploying V2.

## Privacy and deployment boundaries

- The bundled business sample is synthetic. **Do not commit real client records, secrets, imported CSVs, or individual prediction rows to a public repository.**
- Session-memory CSV upload is not a secure multi-business database or a guarantee of enterprise-grade data protection.
- No commercial multi-tenant authentication, individualized model training, automatic retraining or Trainerize API synchronization is claimed for V2.
- Predictive Lab must not imply that its experimental research results apply to an uploaded business.
- Avoid financial decisions based solely on experimental forecasting scores.

## Further documentation

- [V2 methods and limitations](docs/V2_METHODS_AND_LIMITATIONS.md)
- [V2 release checklist](docs/V2_RELEASE_CHECKLIST.md)
- [Data dictionary](docs/data_dictionary.md)
- [KPI definitions](docs/kpi_definitions.md)

**Stack:** Python, pandas, NumPy, scikit-learn, Streamlit, Plotly, pytest.
