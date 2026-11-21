#Author: etrykk
#Date: 11/18/2026
#Personal Training Analytics Platform

import pandas as pd
import random
import datetime as dt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
SAMPLE_DIR = DATA_DIR / "sample"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

REFERENCE_DATE = dt.datetime(2026, 11, 20)


def generate_clients(num_clients):
    """
    Generate client records.
    """
    client_records = []
    start_date = dt.datetime(2024, 1, 1)
    trainer_ids = [f"T{i:03d}" for i in range(1, 11)]

    for client in range(1, num_clients + 1):
        client_id = f"C{client:03d}"
        random_days = random.randint(0, (REFERENCE_DATE - start_date).days)
        join_date = start_date + dt.timedelta(days=random_days)

        client_records.append({
            "client_id": client_id,
            "join_date": join_date.strftime("%Y-%m-%d"),
            "status": "inactive",  #Updated after membership and session generation
            "trainer_id": random.choice(trainer_ids)
        })

    client_data = pd.DataFrame(client_records, columns=[
        "client_id", "join_date", "status", "trainer_id"
    ])
    client_data.to_csv(SAMPLE_DIR / "clients.csv", index=False)
    return client_data


def generate_memberships(client_data):
    """
    Generate monthly memberships with consistent cancellation dates
    """
    membership_records = []
    membership_plans = {"Basic": 75, "Standard": 125, "Premium": 200}
    cancellation_reasons = [
        "Too expensive", "Scheduling conflicts", "Relocation",
        "Personal reasons", "No longer interested"
    ]

    for client in client_data.itertuples(index=False):
        if random.random() >= 0.40:
            continue

        join_date = pd.Timestamp(client.join_date).to_pydatetime()
        start_date = join_date + dt.timedelta(
            days=random.randint(0, (REFERENCE_DATE - join_date).days)
        )
        plan_name = random.choice(list(membership_plans))
        has_ended = random.choice([True, False])
        end_date = None
        cancellation_date = None
        cancellation_reason = None
        status = "active"

        if has_ended:
            end_date = start_date + dt.timedelta(
                days=random.randint(0, (REFERENCE_DATE - start_date).days)
            )
            status = random.choice(["canceled", "expired"])
            if status == "canceled":
                #Cancellation takes effect on the membership end date
                cancellation_date = end_date
                cancellation_reason = random.choice(cancellation_reasons)

        membership_records.append({
            "membership_id": f"M{len(membership_records) + 1:03d}",
            "client_id": client.client_id,
            "plan_name": plan_name,
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d") if end_date else None,
            "monthly_fee": membership_plans[plan_name],
            "status": status,
            "cancellation_date": (
                cancellation_date.strftime("%Y-%m-%d")
                if cancellation_date else None
            ),
            "cancellation_reason": cancellation_reason
        })

    membership_data = pd.DataFrame(membership_records, columns=[
        "membership_id", "client_id", "plan_name", "start_date",
        "end_date", "monthly_fee", "status", "cancellation_date",
        "cancellation_reason"
    ])
    membership_data.to_csv(SAMPLE_DIR / "memberships.csv", index=False)
    return membership_data


def generate_packages(client_data, membership_data):
    """
    Generate prepaid training packages with nonoverlapping validity periods.
    """
    package_records = []
    package_options = {5: 275, 10: 500, 20: 900}
    membership_clients = set(membership_data["client_id"])

    for client in client_data.itertuples(index=False):
        purchase_probability = 0.15 if client.client_id in membership_clients else 0.40
        if random.random() >= purchase_probability:
            continue

        join_date = pd.Timestamp(client.join_date).to_pydatetime()
        desired_packages = random.randint(1, 3)
        earliest_purchase = join_date

        for _ in range(desired_packages):
            if earliest_purchase > REFERENCE_DATE:
                break

            #Allow repeat purchases, but do not overlap package validity periods
            purchase_date = earliest_purchase + dt.timedelta(
                days=random.randint(0, (REFERENCE_DATE - earliest_purchase).days)
            )
            expiration_date = purchase_date + dt.timedelta(days=180)
            package_sessions = random.choice(list(package_options))

            package_records.append({
                "package_id": f"P{len(package_records) + 1:03d}",
                "client_id": client.client_id,
                "purchase_date": purchase_date.strftime("%Y-%m-%d"),
                "expiration_date": expiration_date.strftime("%Y-%m-%d"),
                "package_sessions": package_sessions,
                "package_price": package_options[package_sessions],
                "sessions_redeemed": 0,
                "sessions_remaining": package_sessions,
                "status": "active" if expiration_date >= REFERENCE_DATE else "expired"
            })
            earliest_purchase = expiration_date + dt.timedelta(days=1)

    package_data = pd.DataFrame(package_records, columns=[
        "package_id", "client_id", "purchase_date", "expiration_date",
        "package_sessions", "package_price", "sessions_redeemed",
        "sessions_remaining", "status"
    ])
    package_data.to_csv(SAMPLE_DIR / "packages.csv", index=False)
    return package_data


def generate_sessions(client_data, membership_data, package_data):
    """
    Generate chronologically priced sessions without trainer booking conflicts.
    """
    session_records = []
    candidate_sessions = []
    session_durations = [30, 45, 60, 90]
    session_fees = [40, 50, 60, 75, 90]
    trainer_bookings = {}
    client_bookings = {}

    #Create candidate sessions before assigning pricing or package credits
    for client in client_data.itertuples(index=False):
        join_date = pd.Timestamp(client.join_date).to_pydatetime()
        for _ in range(random.randint(5, 30)):
            session_date = join_date + dt.timedelta(
                days=random.randint(0, (REFERENCE_DATE - join_date).days)
            )
            candidate_sessions.append({
                "client_id": client.client_id,
                "trainer_id": client.trainer_id,
                "session_date": session_date,
                "duration_minutes": random.choice(session_durations),
                "session_status": random.choices(
                    ["completed", "canceled", "no_show"],
                    weights=[0.80, 0.10, 0.10], k=1
                )[0]
            })

    candidate_sessions.sort(key=lambda record: record["session_date"])
    membership_lookup = {
        client_id: group.to_dict("records")
        for client_id, group in membership_data.groupby("client_id")
    }
    package_lookup = {
        client_id: sorted(group.to_dict("records"), key=lambda p: p["purchase_date"])
        for client_id, group in package_data.groupby("client_id")
    }
    package_balances = {
        row.package_id: int(row.package_sessions)
        for row in package_data.itertuples(index=False)
    }

    for candidate in candidate_sessions:
        session_date = candidate["session_date"]
        duration_minutes = candidate["duration_minutes"]
        trainer_id = candidate["trainer_id"]
        client_id = candidate["client_id"]
        booking_key = (trainer_id, session_date.date())
        session_start = None

        #Find an appointment slot between 8 AM and 6 PM
        for _ in range(100):
            start_minutes = random.randrange(
                8 * 60, 18 * 60 - duration_minutes + 1, 15
            )
            proposed_start = session_date + dt.timedelta(minutes=start_minutes)
            proposed_end = proposed_start + dt.timedelta(minutes=duration_minutes)
            bookings = trainer_bookings.get(booking_key, [])
            client_key = (client_id, session_date.date())
            client_appointments = client_bookings.get(client_key, [])

            trainer_available = all(
                proposed_end <= booked_start or proposed_start >= booked_end
                for booked_start, booked_end in bookings
            )
            client_available = all(
                proposed_end <= booked_start or proposed_start >= booked_end
                for booked_start, booked_end in client_appointments
            )

            if trainer_available and client_available:
                session_start = proposed_start
                trainer_bookings.setdefault(booking_key, []).append(
                    (proposed_start, proposed_end)
                )
                client_bookings.setdefault(client_key, []).append(
                    (proposed_start, proposed_end)
                )
                break

        if session_start is None:
            continue

        #Membership coverage takes priority over package or session pricing
        has_membership = any(
            pd.Timestamp(membership["start_date"]) <= session_date
            and (
                pd.isna(membership["end_date"])
                or session_date <= pd.Timestamp(membership["end_date"])
            )
            for membership in membership_lookup.get(client_id, [])
        )

        pricing_model = "per_session"
        session_fee = random.choice(session_fees)
        package_id = None

        if has_membership:
            pricing_model = "monthly"
            session_fee = 0
        else:
            #Redeem the earliest eligible package with available credits
            for package in package_lookup.get(client_id, []):
                if (
                    pd.Timestamp(package["purchase_date"]) <= session_date
                    <= pd.Timestamp(package["expiration_date"])
                    and package_balances[package["package_id"]] > 0
                ):
                    package_id = package["package_id"]
                    pricing_model = "package"
                    session_fee = 0
                    if candidate["session_status"] == "completed":
                        package_balances[package_id] -= 1
                    break

        session_records.append({
            "session_id": f"S{len(session_records) + 1:04d}",
            "client_id": client_id,
            "trainer_id": trainer_id,
            "session_date": session_date.strftime("%Y-%m-%d"),
            "session_time": session_start.strftime("%H:%M"),
            "session_status": candidate["session_status"],
            "duration_minutes": duration_minutes,
            "pricing_model": pricing_model,
            "session_fee": session_fee,
            "package_id": package_id
        })

    session_data = pd.DataFrame(session_records, columns=[
        "session_id", "client_id", "trainer_id", "session_date",
        "session_time", "session_status", "duration_minutes",
        "pricing_model", "session_fee", "package_id"
    ])
    session_data.to_csv(SAMPLE_DIR / "sessions.csv", index=False)
    return session_data


def update_package_balances(package_data, session_data):
    """
    Calculate redeemed credits and current package statuses.
    """
    package_data = package_data.copy()
    completed_sessions = session_data.loc[
        (session_data["pricing_model"] == "package")
        & (session_data["session_status"] == "completed")
    ]
    redeemed_counts = completed_sessions.groupby("package_id").size()
    package_data["sessions_redeemed"] = (
        package_data["package_id"].map(redeemed_counts).fillna(0).astype(int)
    )
    package_data["sessions_remaining"] = (
        package_data["package_sessions"] - package_data["sessions_redeemed"]
    )

    def get_package_status(package):
        if pd.Timestamp(package["expiration_date"]) < REFERENCE_DATE:
            return "expired"
        if package["sessions_remaining"] == 0:
            return "depleted"
        return "active"

    package_data["status"] = package_data.apply(get_package_status, axis=1)
    package_data.to_csv(SAMPLE_DIR / "packages.csv", index=False)
    return package_data


def update_client_statuses(client_data, membership_data, package_data, session_data):
    """
    Classify clients by current memberships, packages, and recent activity.
    """
    client_data = client_data.copy()
    active_members = set(membership_data.loc[
        membership_data["status"] == "active", "client_id"
    ])
    active_package_clients = set(package_data.loc[
        package_data["status"] == "active", "client_id"
    ])
    recent_cutoff = REFERENCE_DATE - dt.timedelta(days=60)
    recent_clients = set(session_data.loc[
        (session_data["session_status"] == "completed")
        & (pd.to_datetime(session_data["session_date"]) >= recent_cutoff),
        "client_id"
    ])
    active_clients = active_members | active_package_clients | recent_clients
    client_data["status"] = client_data["client_id"].apply(
        lambda client_id: "active" if client_id in active_clients else "inactive"
    )
    client_data.to_csv(SAMPLE_DIR / "clients.csv", index=False)
    return client_data

def generate_payments(client_data, membership_data, package_data, session_data):
    """
    Generate payment transactions for memberships,
    individual sessions, prepaid packages, and refunds.

    Returns
    -------
    pd.DataFrame
        Generated payment information.
    """

    payment_records = []
    reference_date = pd.Timestamp(REFERENCE_DATE)

    payment_statuses = ["completed", "pending", "failed"]
    payment_weights = [0.95, 0.03, 0.02]

    client_ids = set(client_data["client_id"])

    def add_payment(client_id, payment_date, amount, payment_type,
                    membership_id=None, session_id=None, package_id=None):

        if client_id not in client_ids:
            raise ValueError(f"Unknown client: {client_id}")

        payment_id = f"PAY{len(payment_records) + 1:05d}"

        #Memberships and prepaid packages are activated only after payment.
        #Individual sessions can have outstanding or failed charges.
        payment_status = (
            "completed" if payment_type in ("membership", "package")
            else random.choices(payment_statuses, weights=payment_weights, k=1)[0]
        )

        payment_record = {
            "payment_id": payment_id,
            "client_id": client_id,
            "payment_date": pd.Timestamp(payment_date).strftime("%Y-%m-%d"),
            "amount": round(float(amount), 2),
            "payment_status": payment_status,
            "transaction_type": "payment",
            "payment_type": payment_type,
            "membership_id": membership_id,
            "session_id": session_id,
            "package_id": package_id,
            "original_payment_id": None
        }

        payment_records.append(payment_record)

    #Generate recurring monthly membership payments
    for _, membership in membership_data.iterrows():

        client_id = membership["client_id"]
        membership_id = membership["membership_id"]
        monthly_fee = membership["monthly_fee"]

        start_date = pd.Timestamp(membership["start_date"])

        if pd.notna(membership["end_date"]):
            end_date = min(
                pd.Timestamp(membership["end_date"]),
                reference_date
            )
        else:
            end_date = reference_date

        billing_month = 0

        while True:
            payment_date = start_date + pd.DateOffset(months=billing_month)
            
            if payment_date > end_date:
                break
            add_payment(
                client_id,
                payment_date,
                monthly_fee,
                "membership",
                membership_id=membership_id
            )

            billing_month += 1

    #Generate individual session payments
    individual_sessions = session_data[
        (session_data["pricing_model"] == "per_session")
        & (session_data["session_status"] == "completed")
    ]

    for _, session in individual_sessions.iterrows():

        add_payment(
            session["client_id"],
            session["session_date"],
            session["session_fee"],
            "session",
            session_id=session["session_id"]
        )

    #Generate prepaid package purchase payments
    for _, package in package_data.iterrows():

        add_payment(
            package["client_id"],
            package["purchase_date"],
            package["package_price"],
            "package",
            package_id=package["package_id"]
        )

    #Generate refunds from completed payments
    completed_payments = [
        record for record in payment_records
        if record["payment_status"] == "completed"
    ]

    for payment in completed_payments:

        #Assign a 3% probability of a refund
        #Refund only standalone sessions; refunding an active entitlement
        #would require revoking coverage and recalculating session pricing.
        if payment["payment_type"] != "session" or random.random() >= 0.03:
            continue

        original_date = pd.Timestamp(payment["payment_date"])

        #Generate refund within 30 days of original payment
        latest_refund_date = min(
            original_date + pd.Timedelta(days=30),
            reference_date
        )

        random_days = random.randint(
            0, (latest_refund_date - original_date).days
        )

        refund_date = original_date + pd.Timedelta(days=random_days)

        refund_id = f"PAY{len(payment_records) + 1:05d}"

        refund_record = {
            "payment_id": refund_id,
            "client_id": payment["client_id"],
            "payment_date": refund_date.strftime("%Y-%m-%d"),
            "amount": payment["amount"],
            "payment_status": "completed",
            "transaction_type": "refund",
            "payment_type": payment["payment_type"],
            "membership_id": payment["membership_id"],
            "session_id": payment["session_id"],
            "package_id": payment["package_id"],
            "original_payment_id": payment["payment_id"]
        }

        payment_records.append(refund_record)

    #Create payment dataframe
    payment_data = pd.DataFrame(
        payment_records,
        columns=[
            "payment_id",
            "client_id",
            "payment_date",
            "amount",
            "payment_status",
            "transaction_type",
            "payment_type",
            "membership_id",
            "session_id",
            "package_id",
            "original_payment_id"
        ]
    )

    #Sort transactions chronologically
    payment_data = payment_data.sort_values(
        ["payment_date", "payment_id"]
    ).reset_index(drop=True)

    #Export payment data
    payment_data.to_csv(
        SAMPLE_DIR / "payments.csv",
        index=False
    )

    return payment_data

    

def validate_data(client_data, membership_data, package_data, session_data, payment_data):
    """
    Check identifiers, dates, package balances, and appointment conflicts.
    """
    assert client_data["client_id"].is_unique
    assert membership_data["membership_id"].is_unique
    assert package_data["package_id"].is_unique
    assert session_data["session_id"].is_unique

    client_ids = set(client_data["client_id"])
    for data in (membership_data, package_data, session_data):
        assert set(data["client_id"]).issubset(client_ids)

    assert (package_data["sessions_remaining"] >= 0).all()
    assert (
        package_data["sessions_redeemed"] + package_data["sessions_remaining"]
        == package_data["package_sessions"]
    ).all()

    linked_sessions = session_data[session_data["package_id"].notna()]
    assert (linked_sessions["pricing_model"] == "package").all()
    assert set(linked_sessions["package_id"]).issubset(set(package_data["package_id"]))

    #Validate package ownership
    package_owners = package_data.set_index("package_id")["client_id"]
    assert (
        linked_sessions["package_id"].map(package_owners).to_numpy()
        == linked_sessions["client_id"].to_numpy()
    ).all()
    assert (session_data.loc[
        session_data["pricing_model"] != "per_session", "session_fee"
    ] == 0).all()
    assert session_data.loc[
        session_data["pricing_model"] == "per_session", "package_id"
    ].isna().all()

    for package in package_data.itertuples(index=False):
        redeemed = linked_sessions.loc[
            (linked_sessions["package_id"] == package.package_id)
            & (linked_sessions["session_status"] == "completed")
        ]
        assert len(redeemed) == package.sessions_redeemed
        assert (
            (pd.to_datetime(redeemed["session_date"]) >= pd.Timestamp(package.purchase_date))
            & (pd.to_datetime(redeemed["session_date"]) <= pd.Timestamp(package.expiration_date))
        ).all()

    #Confirm no trainer has overlapping appointments on the same day
    for _, group in session_data.groupby(["trainer_id", "session_date"]):
        appointments = sorted(
            (
                dt.datetime.strptime(row.session_time, "%H:%M"),
                dt.datetime.strptime(row.session_time, "%H:%M")
                + dt.timedelta(minutes=row.duration_minutes)
            )
            for row in group.itertuples(index=False)
        )
        assert all(
            appointments[i][1] <= appointments[i + 1][0]
            for i in range(len(appointments) - 1)
        )
    

    #Confirm no client has overlapping appointments on the same day
    for _, group in session_data.groupby(["client_id", "session_date"]):
        appointments = sorted(
            (
                dt.datetime.strptime(row.session_time, "%H:%M"),
                dt.datetime.strptime(row.session_time, "%H:%M")
                + dt.timedelta(minutes=row.duration_minutes)
            )
            for row in group.itertuples(index=False)
        )
        assert all(
            appointments[i][1] <= appointments[i + 1][0]
            for i in range(len(appointments) - 1)
        )

    #Validate active package and membership access is paid in this model
    entitled = payment_data.loc[
        (payment_data["transaction_type"] == "payment")
        & (payment_data["payment_type"].isin(["membership", "package"]))
    ]
    assert entitled["payment_status"].eq("completed").all()
    assert not payment_data.loc[
        payment_data["transaction_type"] == "refund", "payment_type"
    ].isin(["membership", "package"]).any()

    assert payment_data["payment_id"].is_unique
    assert payment_data["payment_id"].notna().all()

    assert payment_data["client_id"].isin(
        client_data["client_id"]
    ).all()


    assert payment_data["amount"].gt(0).all()

    assert payment_data["payment_status"].isin(
        ["completed", "pending", "failed"]
    ).all()

    assert payment_data["transaction_type"].isin(
        ["payment", "refund"]
    ).all()

    assert payment_data["payment_type"].isin(
        ["membership", "session", "package"]
    ).all()

    payment_dates = pd.to_datetime(payment_data["payment_date"])

    assert payment_dates.notna().all()
    assert (payment_dates <= pd.Timestamp(REFERENCE_DATE)).all()

    payment_groups = {
        "membership": ("membership_id", membership_data["membership_id"]),
        "session": ("session_id", session_data["session_id"]),
        "package": ("package_id", package_data["package_id"])
    }

    for payment_type, (id_column, valid_ids) in payment_groups.items():

        records = payment_data[
            payment_data["payment_type"] == payment_type
        ]

        assert records[id_column].notna().all()
        assert records[id_column].isin(valid_ids).all()

        for other_column in ["membership_id", "session_id", "package_id"]:
            if other_column != id_column:
                assert records[other_column].isna().all()

    for payment_type, (id_column, _) in payment_groups.items():

        records = payment_data[
            payment_data["payment_type"] == payment_type
        ]

        source_data = {
            "membership": membership_data,
            "session": session_data,
            "package": package_data
        }[payment_type]

        client_lookup = source_data.set_index(id_column)["client_id"]

        assert (
            records[id_column].map(client_lookup).to_numpy()
            == records["client_id"].to_numpy()
        ).all()

    original_payments = payment_data[
        payment_data["transaction_type"] == "payment"
    ].set_index("payment_id")

    refunds = payment_data[
        payment_data["transaction_type"] == "refund"
    ]

    assert refunds["payment_status"].eq("completed").all()
    assert refunds["original_payment_id"].notna().all()

    assert refunds["original_payment_id"].isin(
        original_payments.index
    ).all()

    assert payment_data.loc[
        payment_data["transaction_type"] == "payment",
        "original_payment_id"
    ].isna().all()

    for _, refund in refunds.iterrows():

        original = original_payments.loc[
            refund["original_payment_id"]
        ]

        assert original["payment_status"] == "completed"
        assert refund["client_id"] == original["client_id"]
        assert refund["payment_type"] == original["payment_type"]
        assert refund["amount"] <= original["amount"]

        assert pd.Timestamp(refund["payment_date"]) >= pd.Timestamp(
            original["payment_date"]
        )

        for column in ["membership_id", "session_id", "package_id"]:
            assert (
                refund[column] == original[column]
                or (pd.isna(refund[column]) and pd.isna(original[column]))
            )

    refund_totals = refunds.groupby("original_payment_id")["amount"].sum()

    for original_id, total in refund_totals.items():
        assert total <= original_payments.loc[original_id, "amount"]

    session_payments = payment_data[
        (payment_data["payment_type"] == "session")
        & (payment_data["transaction_type"] == "payment")
    ]

    eligible_sessions = session_data[
        (session_data["pricing_model"] == "per_session")
        & (session_data["session_status"] == "completed")
    ]

    assert session_payments["session_id"].is_unique

    assert set(session_payments["session_id"]) == set(
        eligible_sessions["session_id"]
    )

    session_lookup = eligible_sessions.set_index("session_id")

    for _, payment in session_payments.iterrows():

        session = session_lookup.loc[payment["session_id"]]

        assert payment["amount"] == session["session_fee"]
        assert payment["payment_date"] == session["session_date"]

    package_payments = payment_data[
        (payment_data["payment_type"] == "package")
        & (payment_data["transaction_type"] == "payment")
    ]

    assert package_payments["package_id"].is_unique
    assert set(package_payments["package_id"]) == set(
        package_data["package_id"]
    )

    package_lookup = package_data.set_index("package_id")

    for _, payment in package_payments.iterrows():

        package = package_lookup.loc[payment["package_id"]]

        assert payment["amount"] == package["package_price"]
        assert payment["payment_date"] == package["purchase_date"]

    membership_payments = payment_data[
        (payment_data["payment_type"] == "membership")
        & (payment_data["transaction_type"] == "payment")
    ]

    membership_lookup = membership_data.set_index("membership_id")

    for _, payment in membership_payments.iterrows():

        membership = membership_lookup.loc[payment["membership_id"]]

        start_date = pd.Timestamp(membership["start_date"])

        end_date = (
            pd.Timestamp(membership["end_date"])
            if pd.notna(membership["end_date"])
            else pd.Timestamp(REFERENCE_DATE)
        )

        payment_date = pd.Timestamp(payment["payment_date"])

        assert start_date <= payment_date <= end_date
        assert payment["amount"] == membership["monthly_fee"]

    print("Payment validation passed.")

    print("All data validation checks passed.")
    
    

if __name__ == "__main__":
    random.seed(42)

    clients = generate_clients(100)
    memberships = generate_memberships(clients)
    packages = generate_packages(clients, memberships)
    sessions = generate_sessions(clients, memberships, packages)
    packages = update_package_balances(packages, sessions)
    clients = update_client_statuses(clients, memberships, packages, sessions)

    print("\nDataset sizes:")
    print(f"Clients: {len(clients)}")
    print(f"Memberships: {len(memberships)}")
    print(f"Packages: {len(packages)}")
    print(f"Sessions: {len(sessions)}")

    print("\nClient statuses:")
    print(clients["status"].value_counts())
    print("\nSession statuses:")
    print(sessions["session_status"].value_counts())
    print("\nPricing models:")
    print(sessions["pricing_model"].value_counts())
    print("\nPackage statuses:")
    print(packages["status"].value_counts())
    
    #Generate payment data
    payments = generate_payments(
        clients, memberships, packages, sessions
    )

    print(payments.head())
    print(payments.shape)
    print(payments["payment_id"].nunique())
    print(payments["payment_type"].value_counts())
    print(payments["payment_status"].value_counts())

    #Calculate net collected revenue
    completed = payments[
        payments["payment_status"] == "completed"
    ]

    gross_revenue = completed.loc[
        completed["transaction_type"] == "payment", "amount"
    ].sum()

    refunds = completed.loc[
        completed["transaction_type"] == "refund", "amount"
    ].sum()

    net_revenue = gross_revenue - refunds

    print(f"Gross collected revenue: ${gross_revenue:,.2f}")
    print(f"Total refunds: ${refunds:,.2f}")
    print(f"Net collected revenue: ${net_revenue:,.2f}")
    
    #Validate generated datasets
    validate_data(
        clients,
        memberships,
        packages,
        sessions,
        payments
        )

