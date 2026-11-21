# Personal Training Analytics Platform
## Key Performance Indicator (KPI) Definitions

**Author:** etrykk  
**Version:** 1.0  
**Date:** November 20, 2026  
**Status:** Planning

---

## 1. Overview

### 1.1 Purpose

This document defines the business metrics calculated by the Personal Training Analytics Platform.

Each KPI includes its purpose, calculation method, required datasets, and validation considerations.

The objective is to ensure that calculations remain accurate and consistent across different fitness businesses.

### 1.2 General Calculation Standards

- All financial calculations will use USD.
- Reporting periods will be selected by the user.
- Dates will follow the `YYYY-MM-DD` format.
- Missing or invalid records must be identified before calculations.
- Historical calculations must use information relevant to the selected reporting period.
- Metrics with insufficient data must be labeled as unavailable rather than reported as zero.
- Percentage calculations will be displayed to two decimal places.
- Financial calculations will be displayed to two decimal places.
- Membership start and end dates are treated as inclusive.

---

## 2. Revenue Analytics

### KPI-01: Monthly Net Revenue

**Purpose:** Measure net cash revenue collected during a selected calendar month.

**Formula:**

Monthly Net Revenue = Completed Payments − Completed Refunds

**Required Dataset:** `payments.csv`

**Calculation Rules:**
- Include transactions processed within the selected reporting month.
- Include only transactions with `payment_status = completed`.
- Add amounts where `transaction_type = payment`.
- Subtract amounts where `transaction_type = refund`.
- Exclude pending and failed transactions.
- Record refunds in the month they were processed, even when the original payment occurred in a previous month.
- Support full and partial refunds.
- Ensure total completed refunds do not exceed the original payment amount.
- Do not subtract a refund twice by also removing the original completed payment.
- Treat the metric as cash-based net receipts, not accrual accounting revenue.

**Output:** USD

### KPI-02: Average Revenue Per Client

**Purpose:** Measure average monthly net revenue generated per active client.

**Formula:**

Average Revenue Per Client = Monthly Net Revenue / Average Active Clients

**Required Datasets:** `payments.csv`, `memberships.csv`

**Calculation Rules:**
- This metric uses active monthly subscribers as its denominator, but all-category net receipts as its numerator; label it clearly and do not interpret it as subscriber-only revenue.
- Calculate average active clients using daily active membership counts during the month.
- Count each client only once per day.
- Use monthly net revenue after refunds.
- Return unavailable if average active clients equals zero.

**Output:** USD per client

---

## 3. Client Analytics

### KPI-03: Active Subscription Clients

**Purpose:** Determine how many clients have active monthly subscriptions on a selected date.

**Formula:**

Active Subscription Clients = Number of Unique Clients with an Active Monthly Subscription

**Required Datasets:** `clients.csv`, `memberships.csv`

**Calculation Rules:**
- Membership start date must be on or before the selected date.
- Membership end date must be on or after the selected date, or be empty.
- Count unique client IDs.
- Use membership history rather than the current client status field.
- Do not count clients whose memberships have not yet started.

**Output:** Integer

### KPI-04: Subscription Retention Rate

**Purpose:** Measure the percentage of starting monthly subscribers still subscribed at period end.

**Formula:**

Retention Rate = (Starting Clients Still Active at Period End / Starting Clients) × 100

**Required Dataset:** `memberships.csv`

**Calculation Rules:**
- Identify clients active at the beginning of the reporting period.
- Determine which of those clients remain active at the end.
- Exclude newly acquired clients from the starting population.
- Count each client only once.
- Return unavailable if there are no starting clients.

**Output:** Percentage

### KPI-05: Subscription Churn Rate

**Purpose:** Measure the percentage of starting monthly subscribers no longer subscribed at period end.

**Formula:**

Churn Rate = (Starting Clients No Longer Active at Period End / Starting Clients) × 100

**Required Dataset:** `memberships.csv`

**Calculation Rules:**
- Use the same starting client population as the retention calculation.
- Count clients who are no longer active at period end.
- Clients who cancel and reactivate before the period ends are considered retained.
- New clients are excluded from the starting population.
- Return unavailable if there are no starting clients.

**Output:** Percentage

**Relationship:**

For the defined starting population:

Retention Rate + Churn Rate = 100%

---

### Membership Lifecycle Rules

The following rules apply to active client, retention, and churn calculations.

**Cancellations**
- Clients remain active until their membership coverage ends.
- A cancellation request does not automatically count as churn.
- Clients whose membership coverage ends before the reporting period closes are considered churned if they have not reactivated by period end.

**Reactivations**
- Returning clients retain their original client ID.
- A reactivation creates a new membership record.
- Clients who cancel and reactivate before the reporting period ends count as retained if they belonged to the starting cohort.
- Clients who return after a previous reporting period are counted as reactivated clients rather than first-time clients.

**Plan Changes**
- A membership plan change creates a new membership record.
- Continuous coverage across a plan change does not count as churn.
- Clients must not be counted more than once when calculating active membership totals.

**Historical Reporting**
- Historical membership activity must be determined from membership start and end dates.
- The current membership status must not override historical coverage information.
- Retention and churn must use the same starting client population.
- Starting and ending populations are measured on the first and last calendar days of the reporting period.

---

## 4. Session Analytics

### KPI-06: Attendance Rate

**Purpose:** Measure the percentage of eligible training sessions that clients attend.

**Formula:**

Attendance Rate = (Completed Sessions / Eligible Sessions) × 100

**Required Dataset:** `sessions.csv`

**Calculation Rules:**
- Completed sessions count as attended.
- No-show sessions count as missed.
- Canceled sessions are excluded from the denominator.
- Future scheduled sessions are excluded.
- Sessions still marked as scheduled after their scheduled date must be flagged for review.
- Return unavailable when no eligible sessions exist.

**Output:** Percentage

### KPI-07: Session Capacity Utilization Rate

**Purpose:** Measure how effectively a business uses its available training capacity.

**Formula:**

Session Utilization Rate = (Booked Training Capacity / Available Training Capacity) × 100

**Required Datasets:** `sessions.csv` and a future trainer availability or capacity dataset.

**Calculation Rules:**
- Available capacity must be defined for each reporting period.
- Booked and available capacity must use the same units.
- Canceled appointments must be handled according to the business's booking policy.
- Return unavailable when available capacity equals zero.
- Do not calculate utilization from attendance records alone.

**Output:** Percentage

**Version 1 Note:** This metric will remain unavailable until reliable trainer capacity information is provided. It is not a required release-blocking metric for Version 1.

---

## 5. Membership Analytics

### KPI-08: Monthly Subscription Plan Distribution

**Purpose:** Identify how active clients are distributed across membership plans.

**Formula:**

Plan Distribution = (Active Clients in Plan / Total Active Clients) × 100

**Required Dataset:** `memberships.csv`

**Calculation Rules:**
- Use membership records active on the selected date.
- Count unique clients.
- Assign each client to one plan for reporting.
- Flag conflicting overlapping memberships.
- Return unavailable when no active clients exist.

**Output:** Percentage by membership plan

### KPI-09: Revenue by Monthly Subscription Plan

**Purpose:** Compare net revenue generated by different membership plans.

**Formula:**

Plan Revenue = Completed Membership Payments − Completed Membership Refunds

**Required Datasets:** `payments.csv`, `memberships.csv`

**Calculation Rules:**
- Include only transactions with `payment_type = membership`.
- Include only completed transactions.
- Associate payments and refunds with their membership IDs.
- Subtract completed refunds from the appropriate membership plan.
- Use transaction dates to determine the reporting period.
- Exclude failed and pending transactions.
- Flag transactions that cannot be assigned to a membership.

**Output:** USD by membership plan

---

## 6. Business Growth Analytics

### KPI-10: Monthly Revenue Growth

**Purpose:** Measure changes in net revenue between consecutive months.

**Formula:**

Revenue Growth Rate = ((Current Month Net Revenue − Previous Month Net Revenue) / Previous Month Net Revenue) × 100

**Required Dataset:** `payments.csv`

**Calculation Rules:**
- Use net revenue after refunds for both months.
- Compare consecutive calendar months.
- Return unavailable when the previous month's net revenue is zero or negative.
- Allow negative growth values when the previous month's net revenue is positive.

**Output:** Percentage

### KPI-11: Subscription Client Growth Rate

**Purpose:** Measure changes in the number of active clients between reporting dates.

**Formula:**

Client Growth Rate = ((Current Active Clients − Previous Active Clients) / Previous Active Clients) × 100

**Required Dataset:** `memberships.csv`

**Calculation Rules:**
- Compare active client counts at equivalent reporting dates.
- Use unique client IDs.
- Return unavailable when the previous client count equals zero.
- Allow negative growth values.

**Output:** Percentage

---

## 7. Additional Pricing-Model Metrics

### KPI-12: Revenue by Payment Category

**Purpose:** Compare net cash receipts from monthly subscriptions, individual sessions, prepaid packages, and other sales.

**Formula:** Category Net Receipts = Completed Category Payments − Completed Category Refunds

**Required Dataset:** `payments.csv`

**Rules:** Group completed transactions by `payment_type` (`membership`, `session`, `package`, `other`); use transaction processing dates; do not count a package redemption as a new sale.

**Output:** USD by category

### KPI-13: Active Training Clients

**Purpose:** Count distinct clients who completed at least one session during the reporting period, regardless of pricing model.

**Formula:** Active Training Clients = Distinct Client IDs with Completed Sessions

**Required Dataset:** `sessions.csv`

**Output:** Integer

### KPI-14: Package Redemption Rate

**Purpose:** Measure the portion of prepaid package sessions used.

**Formula:** Package Redemption Rate = Completed Package-Linked Sessions / Purchased Package Sessions × 100

**Required Datasets:** `packages.csv`, `sessions.csv`

**Rules:** Count completed sessions linked to valid packages; exclude canceled and no-show sessions; apply a consistent package purchase cohort or selected package set to numerator and denominator; return unavailable when no package sessions were purchased.

**Output:** Percentage

### KPI-15: Repeat Session Client Rate

**Purpose:** Measure the share of pay-per-session clients with repeat completed training sessions in the selected reporting period.

**Formula:** Repeat Session Client Rate = Pay-per-Session Clients with at Least Two Completed Sessions / Pay-per-Session Clients with at Least One Completed Session × 100

**Required Datasets:** `sessions.csv`, `payments.csv`

**Rules:** Identify pay-per-session clients from session-type payments linked to sessions; do not classify prepaid package redemptions as pay-per-session transactions; return unavailable when the denominator is zero.

**Output:** Percentage

---

## 8. Data Quality Requirements

Before calculating KPIs, the application must:

- Validate required columns and data types.
- Identify duplicate records.
- Check relationships between datasets.
- Identify invalid membership periods.
- Identify missing or inconsistent payment information.
- Validate refund references and amounts.
- Validate session and package payment links and package redemption limits.
- Identify refunds exceeding their original payments.
- Flag sessions with invalid attendance statuses.
- Report records excluded from calculations.

Invalid records must not silently affect reported business metrics.

---

## 9. Reporting Standards

Each generated report should include:

- Business reporting period
- KPI names and calculated values
- Relevant units and percentages
- Visualizations of important trends
- Data quality warnings
- Unavailable metrics and their explanations
- Date of report generation

Calculations must be consistent between downloadable reports and the Streamlit dashboard.

---

## 10. Future KPI Extensions

Future versions may introduce:

- Client lifetime value
- Customer acquisition cost
- Revenue forecasting
- Client churn probability
- Trainer performance metrics
- Client engagement scores
- Membership renewal predictions

Predictive metrics must be evaluated using appropriate historical data before being presented as reliable business insights.
