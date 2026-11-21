# Personal Training Analytics Platform
## Data Dictionary

**Author:** etrykk  
**Version:** 1.0  
**Date:** November 20, 2026  
**Status:** Planning

---

## 1. Overview

### 1.1 Purpose

This document defines the structure, data types, relationships, and validation requirements for the datasets used in the Personal Training Analytics Platform.

The application will process five primary CSV datasets:

- `clients.csv` — Client information
- `memberships.csv` — Membership plans and history
- `sessions.csv` — Training session records
- `payments.csv` — Financial transactions
- `packages.csv` — Prepaid session package purchases

### 1.2 Data Standards

- CSV files must contain a header row.
- Column names must be standardized before analysis.
- Dates must use the `YYYY-MM-DD` format.
- Identifiers must be unique within their respective datasets where specified.
- Missing required values must be identified during validation.
- Monetary amounts must be represented in USD for Version 1.
- Personal identifying information should not be included in demonstration datasets.

---

## 2. Client Dataset — clients.csv

### 2.1 Description

Stores basic information about clients registered with a personal training business.

| Column | Data Type | Required | Description |
|---|---|---|---|
| client_id | String | Yes | Unique client identifier |
| join_date | Date | Yes | Date client first joined |
| status | String | Yes | Current client status |
| trainer_id | String | No | Assigned trainer identifier |

### 2.2 Validation Rules

- `client_id` must be unique and cannot be missing.
- `join_date` must contain a valid date.
- `status` must be either `active` or `inactive`.
- `trainer_id` may be empty.
- Client status represents the current record, not historical membership activity.

---

## 3. Membership Dataset — memberships.csv

### 3.1 Description

Stores recurring monthly subscription plans, pricing, and historical subscription periods. Pay-per-session clients and package clients do not need a monthly subscription.

Each membership record represents a continuous period of membership coverage. Clients may have multiple records due to cancellations, reactivations, or plan changes.

Historical records must be preserved to support accurate retention and churn calculations.

### 3.2 Dataset Structure

| Column | Data Type | Required | Description |
|---|---|---|---|
| membership_id | String | Yes | Unique membership identifier |
| client_id | String | Yes | Associated client identifier |
| plan_name | String | Yes | Membership plan name |
| start_date | Date | Yes | First day of membership coverage |
| end_date | Date | No | Last day of membership coverage |
| monthly_fee | Float | Yes | Recurring monthly subscription fee in USD |
| status | String | Yes | Current state of this membership record |
| cancellation_date | Date | No | Date cancellation was requested |
| cancellation_reason | String | No | Reason for cancellation |

### 3.3 Membership Status Values

| Status | Description |
|---|---|
| active | Membership currently active |
| canceled | Membership ended following cancellation |
| expired | Membership ended without renewal |
| scheduled | Membership begins on a future date |

Membership status describes the current state of the record. Historical activity must be determined using membership coverage dates.

### 3.4 Membership Cancellation Rules

- A canceled membership must retain its original start date.
- The end date must represent the final day of membership coverage.
- The cancellation date represents when the cancellation was requested.
- A cancellation request does not necessarily terminate membership coverage immediately.
- Canceled membership records must remain in the dataset.
- Cancellation reasons may be missing.
- Cancellations must not automatically delete associated payments or training sessions.

### 3.5 Membership Reactivation Rules

- A client returning after a membership gap must receive a new membership record.
- The new membership must have a unique membership ID.
- The original client ID must be preserved.
- Previous membership records must not be overwritten.
- Reactivated clients count as active beginning on the new membership start date.
- Reactivations must be distinguishable from first-time clients using membership history.

### 3.6 Membership Plan Changes

- Plan changes must create a new membership record when the plan or price changes.
- The previous membership period must receive an appropriate end date.
- The new membership period must begin after the previous period ends.
- Plan changes without a gap in coverage must not count as client churn.
- Historical payments must remain associated with their original membership IDs.

### 3.7 Validation Rules

- `membership_id` must be unique.
- `client_id` must exist in `clients.csv`.
- `start_date` must contain a valid date.
- `end_date` must be on or after `start_date`, if provided.
- `monthly_fee` must be greater than or equal to zero.
- Only recurring monthly subscriptions are recorded in this dataset. Clients can have sessions or packages without a membership.
- `status` must be `active`, `canceled`, `expired`, or `scheduled`.
- `cancellation_date` must be a valid date when provided.
- `cancellation_date` cannot precede `start_date`.
- `cancellation_date` must not be later than `end_date` for a canceled membership.
- Canceled and expired memberships must have an end date.
- Scheduled memberships must have a future start date.
- Overlapping membership coverage periods for the same client must be flagged for review.
- Missing end dates represent open-ended membership periods.
- A membership record with a future start date must not be counted as currently active.

### 3.8 Historical Reporting

Membership analytics must use coverage dates rather than relying solely on the status column.

A client is considered active on a selected date when at least one valid membership period covers that date.

Membership history must support:

- Historical active client counts
- Retention and churn calculations
- Membership reactivations
- Membership plan changes
- Client growth reporting

--

## 4. Session Dataset — sessions.csv

### 4.1 Description

Stores information about scheduled personal training sessions and their attendance outcomes.

| Column | Data Type | Required | Description |
|---|---|---|---|
| session_id | String | Yes | Unique session identifier |
| client_id | String | Yes | Associated client identifier |
| trainer_id | String | Yes | Trainer responsible for session |
| session_date | Date | Yes | Scheduled session date |
| session_status | String | Yes | Session attendance outcome |
| duration_minutes | Integer | Yes | Scheduled session duration |
| session_fee | Float | No | Agreed session price for a pay-per-session booking; not necessarily cash received |
| package_id | String | No | Package used to cover this session, if applicable |

### 4.2 Validation Rules

- `session_id` must be unique.
- `client_id` must exist in `clients.csv`.
- `session_date` must contain a valid date.
- `session_status` must be `scheduled`, `completed`, `canceled`, or `no_show`.
- `duration_minutes` must be greater than zero.
- `session_fee`, when provided, must be greater than or equal to zero.
- `package_id`, when provided, must reference a package belonging to the same client.
- A session must not be charged as a new pay-per-session sale when already covered by a prepaid package.
- Sessions associated with inactive or canceled memberships should be flagged for review rather than automatically deleted.

---

## 5. Payment Dataset — payments.csv

### 5.1 Description

Stores financial transactions, including successful payments, failed payments, pending payments, full refunds, and partial refunds.

Each payment or refund is recorded as a separate transaction to preserve historical financial information.

### 5.2 Dataset Structure

| Column | Data Type | Required | Description |
|---|---|---|---|
| payment_id | String | Yes | Unique transaction identifier |
| client_id | String | Yes | Associated client identifier |
| payment_date | Date | Yes | Transaction processing date |
| amount | Float | Yes | Transaction amount in USD |
| payment_status | String | Yes | Processing status |
| transaction_type | String | Yes | Payment or refund |
| payment_type | String | Yes | Business category of transaction |
| membership_id | String | No | Associated membership identifier |
| original_payment_id | String | No | Original transaction associated with a refund |
| session_id | String | No | Associated training session for an individual-session sale |
| package_id | String | No | Associated package purchase for a package sale |

### 5.3 Payment Status Values

| Status | Description |
|---|---|
| completed | Transaction successfully processed |
| pending | Transaction awaiting processing |
| failed | Transaction unsuccessful |

Refunds use the same processing statuses. A completed refund is recorded as a separate transaction with `transaction_type = refund`.

### 5.4 Transaction Type Values

| Type | Description |
|---|---|
| payment | Money received from a client |
| refund | Money returned to a client |

### 5.5 Payment Type Values

| Type | Description |
|---|---|
| membership | Membership subscription payment |
| session | Individual training session payment |
| package | Prepaid training package |
| other | Other business revenue |

### 5.6 Refund Handling

Refunds must be stored as separate transactions.

- Full refunds return the entire original payment amount.
- Partial refunds return only a portion of the original payment.
- Refunds must reference the original payment using `original_payment_id`.
- Refund amounts must be recorded as positive values.
- The transaction type determines whether an amount is added to or subtracted from revenue.
- Refunds are assigned to the month in which they were processed.
- Multiple partial refunds may reference the same original payment.
- Total completed refunds for a payment must not exceed its original completed payment amount.

### 5.7 Validation Rules

- `payment_id` must be unique.
- `client_id` must exist in `clients.csv`.
- `payment_date` must contain a valid date.
- `amount` must be greater than zero.
- `payment_status` must be `completed`, `pending`, or `failed`.
- `transaction_type` must be `payment` or `refund`.
- `payment_type` must be `membership`, `session`, `package`, or `other`.
- Refund transactions must reference an existing original payment.
- The original payment must belong to the same client.
- The original payment must be a completed payment before its refund is completed.
- Completed refunds must not exceed the original payment amount.
- If provided, `membership_id` must exist in `memberships.csv` and belong to the same client.
- A refund must retain the business category of its original payment.
- `session_id`, when provided, must reference an existing session for the same client.
- `package_id`, when provided, must reference an existing package purchase for the same client.
- Completed package purchases are counted as cash received once; package redemptions do not create new revenue.
- A payment should reference the applicable membership, session, or package record when the category requires it.

---

## 6. Package Dataset — packages.csv

### 6.1 Description

Stores each prepaid session package purchased by a client. A client can buy multiple packages over time. Package usage is linked through `sessions.csv` using `package_id`.

### 6.2 Dataset Structure

| Column | Data Type | Required | Description |
|---|---|---|---|
| package_id | String | Yes | Unique package purchase identifier |
| client_id | String | Yes | Client who purchased the package |
| purchase_date | Date | Yes | Date the package was purchased |
| package_name | String | Yes | Package offering name |
| package_sessions | Integer | Yes | Number of included sessions |
| package_price | Float | Yes | Agreed total purchase price in USD |
| expiration_date | Date | No | Final date for package redemption |

### 6.3 Validation Rules

- `package_id` must be unique and nonempty.
- `client_id` must exist in `clients.csv`.
- `purchase_date` must be valid.
- `package_sessions` must be a positive integer.
- `package_price` must be greater than zero.
- `expiration_date`, if provided, must not precede `purchase_date`.
- A package-linked session must belong to the package's client.
- Completed sessions redeemed against a package must not exceed `package_sessions`.
- Package sales and refunds are recorded in `payments.csv`; the package price is not added to cash revenue separately.

---

## 7. Dataset Relationships

### 7.1 Primary Keys

| Dataset | Primary Key |
|---|---|
| clients.csv | client_id |
| memberships.csv | membership_id |
| sessions.csv | session_id |
| payments.csv | payment_id |
| packages.csv | package_id |

### 7.2 Foreign Keys

| Dataset | Foreign Key | References |
|---|---|---|
| memberships.csv | client_id | clients.csv |
| sessions.csv | client_id | clients.csv |
| payments.csv | client_id | clients.csv |
| payments.csv | membership_id | memberships.csv |
| payments.csv | original_payment_id | payments.csv |
| payments.csv | session_id | sessions.csv |
| payments.csv | package_id | packages.csv |
| sessions.csv | package_id | packages.csv |
| packages.csv | client_id | clients.csv |

### 7.3 Relationship Structure

- One client can have multiple membership records.
- One client can have multiple training sessions.
- One client can have multiple financial transactions.
- One membership can be associated with multiple payments.
- One original payment can be associated with multiple refund transactions.
- One client can purchase multiple prepaid packages.
- One package can cover multiple sessions.
- Per-session clients may have no membership record.
- Each session is associated with one client and one trainer.

---

## 8. Missing Data Handling

The application must distinguish between required and optional information.

### 8.1 Required Fields

Records missing required identifiers or essential dates must be flagged as invalid.

### 8.2 Optional Fields

Missing optional fields may be retained when they do not prevent accurate calculations.

`original_payment_id` is optional for payments but required for refunds.

### 8.3 Data Quality Reporting

The application should report:

- Number of records processed
- Number of invalid records
- Number of duplicate records
- Number of missing values
- Number of records excluded from calculations

---

## 9. Data Privacy

The application will initially use synthetic datasets for development and public demonstrations.

Real client information must not be committed to the public GitHub repository.

The initial data structure will avoid unnecessary personal identifiers, including names, email addresses, phone numbers, and health information.

Any future handling of real business records must follow applicable privacy, security, and data retention requirements.

---

## 10. Future Data Extensions

Future versions may support additional datasets or fields, including:

- Trainer information and performance metrics
- Membership package changes
- Session capacity and availability
- Historical client engagement
- Trainerize API data
- Additional financial information
- Predictive modeling features

New data sources must be converted into the standardized internal structure before being processed by the analytics engine.
