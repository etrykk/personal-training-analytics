#Author: etrykk
#Personal Training Analytics Platform - V2
#Dataset Analytics Readiness

import pandas as pd

from ..data_schema import REQUIRED_COLUMNS


def analyze_data_readiness(datasets):
    '''
    Check whether each expected dataset is available and contains
    the minimum columns and records needed for basic analytics.

    Parameters
    ----------
    datasets : dict
        Dictionary mapping dataset names to pandas DataFrames.

    Returns
    -------
    pandas.DataFrame
        One row per expected dataset, including presence, row count,
        missing required columns, and basic readiness status.
    '''
    if not isinstance(datasets, dict):
        raise TypeError('datasets must be a dictionary')

    results = []

    for dataset_name, required_columns in REQUIRED_COLUMNS.items():
        data = datasets.get(dataset_name)
        present = isinstance(data, pd.DataFrame)

        if present:
            row_count = len(data)
            missing_columns = [column for column in required_columns
                               if column not in data.columns]
        else:
            row_count = 0
            missing_columns = list(required_columns)

        ready = present and row_count > 0 and not missing_columns

        results.append({
            'dataset': dataset_name,
            'present': present,
            'row_count': row_count,
            'missing_columns': missing_columns,
            'ready': ready
        })

    return pd.DataFrame(results, columns=[
        'dataset', 'present', 'row_count', 'missing_columns', 'ready'
    ])
