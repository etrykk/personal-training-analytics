# Personal Training Analytics Platform

## Software Requirements Specification (SRS)

**Author:** etrykk  

**Version:** 1.0  

**Date:** November 20, 2026  

**Status:** Planning

---

## 1. Project Overview

### 1.1 Project Description

The Personal Training Analytics Platform is a Python-based application designed to help personal trainers, gym owners, and fitness businesses analyze client and business performance.

The application will allow users to upload CSV files containing client, subscription, session, payment, and prepaid-package information. The data will be cleaned, validated, and analyzed to generate business performance metrics and interactive visualizations.

### 1.2 Project Objectives

- Provide accessible business analytics for fitness professionals.

- Automate calculations of important business metrics.

- Identify trends in client retention, attendance, and revenue.

- Generate interactive dashboards and downloadable reports.

- Design a scalable architecture for future integrations.

### 1.3 Target Users

- Independent personal trainers

- Personal training businesses

- Gym owners and managers

- Fitness studios

---

## 2. Functional Requirements

### FR-01: Data Import

- FR-01.1: Accept CSV files containing client information.

- FR-01.2: Accept CSV files containing membership information.

- FR-01.3: Accept CSV files containing session information.

- FR-01.4: Accept CSV files containing payment information.
- FR-01.6: Accept CSV files containing prepaid package purchases.

- FR-01.5: Support column mapping for different CSV formats.

### FR-02: Data Validation

- FR-02.1: Identify missing required values.

- FR-02.2: Detect duplicate records.

- FR-02.3: Validate dates and numerical values.

- FR-02.4: Verify relationships between datasets.

- FR-02.5: Display understandable error messages.

### FR-03: Business Analytics

- FR-03.1: Calculate monthly revenue.

- FR-03.2: Calculate active client counts.

- FR-03.3: Calculate client retention rates.

- FR-03.4: Calculate client churn rates.

- FR-03.5: Calculate attendance rates.

- FR-03.6: Calculate session utilization.

- FR-03.7: Compare monthly subscription plan performance.
- FR-03.8: Calculate net receipts separately for monthly subscriptions, individual sessions, prepaid packages, and other sales.
- FR-03.9: Track prepaid package purchases and session redemptions.
- FR-03.10: Distinguish active monthly subscribers from clients active through completed sessions.
- FR-03.11: Calculate subscription retention and churn without treating pay-per-session gaps as automatic churn.
- FR-03.12: Support businesses using multiple pricing models simultaneously.

### FR-04: Dashboard

- FR-04.1: Display key performance indicators.

- FR-04.2: Generate interactive visualizations.

- FR-04.3: Support date and membership filters.

- FR-04.4: Display data quality warnings.

- FR-04.5: Handle invalid or empty datasets.

### FR-05: Reporting

- FR-05.1: Generate downloadable business summaries.

- FR-05.2: Export analytics results.

- FR-05.3: Identify reporting periods.

- FR-05.4: Document incomplete data.

---

## 3. Non-Functional Requirements

### 3.1 Usability

- The application must be usable without programming knowledge.

- The interface must provide clear navigation and instructions.

### 3.2 Reliability

- Invalid CSV files must not crash the application.

- Business calculations must be verified through automated testing.

### 3.3 Security

- The application must not publicly expose client information.

- Real client data and API credentials must not be uploaded to the public GitHub repository.

- Real client data must not be used in the public demonstration.

### 3.4 Maintainability

- Python functions must be organized into separate modules.

- Data processing and analytics must operate independently of the dashboard.

- Code must contain appropriate documentation.

### 3.5 Scalability

- The application must support future API integrations.

- The architecture should allow additional analytics features.

- Data import methods must use a standardized internal structure.

---

## 4. Technology Requirements

| Technology | Purpose |

|---|---|

| Python | Main programming language |

| Pandas | Data processing and analytics |

| NumPy | Numerical operations |

| Streamlit | Interactive application |

| Plotly | Interactive visualizations |

| Matplotlib / Seaborn | Exploratory visualizations |

| pytest | Automated testing |

| GitHub | Version control and distribution |

---

## 5. Version 1 Limitations

The following features will not be included in Version 1:

- Trainerize API integration

- Automatic data synchronization

- Predictive machine learning models

- Revenue forecasting

- Multi-business user accounts

- Cloud-based client data storage

- Automated billing and scheduling

These features may be considered for future releases.

---

## 6. Release Acceptance Criteria

Version 1 will be considered complete when:

- [ ] Users can upload all four required CSV datasets.

- [ ] Uploaded data is validated and standardized.

- [ ] Business analytics calculations produce verified results.

- [ ] Interactive dashboard visualizations function correctly.

- [ ] Users can filter analytics by reporting period.

- [ ] Users can download business reports.

- [ ] Invalid files produce understandable error messages.

- [ ] Automated tests pass.

- [ ] Documentation and installation instructions are complete.

- [ ] The application can run locally using synthetic data.

---

## 7. Future Development

### Version 2 — Predictive Analytics

- Client churn prediction

- Revenue forecasting

- Model performance evaluation

### Version 3 — Business Integrations

- Trainerize API connection

- Automated data synchronization

- Additional data source integrations

### Version 4 — Model Improvement

- Model version tracking

- Performance monitoring

- Retraining using authorized historical data

- Business-specific predictive models

