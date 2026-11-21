#Author: etrykk
#Date: 11/20/2026
#Personal Training Analytics Platform
#Imported Data Validation

#Columns required for the existing analytics pipeline
REQUIRED_COLUMNS = {
    'clients': ['client_id', 'join_date', 'status'],
    'memberships': ['membership_id', 'client_id', 'start_date', 'monthly_fee', 'status'],
    'packages': ['package_id', 'client_id', 'purchase_date', 'expiration_date',
                 'package_sessions', 'package_price'],
    'sessions': ['session_id', 'client_id', 'session_date', 'session_status',
                 'pricing_model', 'session_fee'],
    'payments': ['payment_id', 'client_id', 'payment_date', 'amount',
                 'payment_status', 'transaction_type', 'payment_type']
}

ID_COLUMNS = {'clients': 'client_id', 'memberships': 'membership_id',
              'packages': 'package_id', 'sessions': 'session_id',
              'payments': 'payment_id'}

DATE_COLUMNS = {
    'clients': ['join_date'],
    'memberships': ['start_date', 'end_date', 'cancellation_date'],
    'packages': ['purchase_date', 'expiration_date'],
    'sessions': ['session_date'],
    'payments': ['payment_date']
}

NUMERIC_COLUMNS = {
    'memberships': ['monthly_fee'],
    'packages': ['package_sessions', 'package_price', 'sessions_redeemed', 'sessions_remaining'],
    'sessions': ['duration_minutes', 'session_fee'],
    'payments': ['amount']
}

ALLOWED_VALUES = {
    'clients': {'status': {'active', 'inactive'}},
    'memberships': {'status': {'active', 'canceled', 'expired'}},
    'packages': {'status': {'active', 'expired', 'depleted'}},
    'sessions': {'session_status': {'completed', 'canceled', 'no_show'},
                 'pricing_model': {'monthly', 'package', 'per_session'}},
    'payments': {'payment_status': {'completed', 'pending', 'failed'},
                 'transaction_type': {'payment', 'refund'},
                 'payment_type': {'membership', 'session', 'package'}}
}


