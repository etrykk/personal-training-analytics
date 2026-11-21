#Author: etrykk
#Date: 11/20/2026
#Personal Training Analytics Platform
#CSV Data Loader

import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def validate_file(file_path):
    file_path = Path(file_path).expanduser()

    if not file_path.is_absolute():
        file_path = PROJECT_ROOT / file_path

    if not file_path.exists():
        raise FileNotFoundError(f"File does not exist: {file_path}")

    if not file_path.is_file():
        raise ValueError(f"Path is not a file: {file_path}")

    if file_path.suffix.lower() != ".csv":
        raise ValueError(f"File must be a CSV: {file_path}")

    return file_path


def load_csv(file_path):
    file_path = validate_file(file_path)

    try:
        data = pd.read_csv(file_path)

    except pd.errors.EmptyDataError as error:
        raise ValueError(
            f"CSV file is empty: {file_path}"
        ) from error

    except pd.errors.ParserError as error:
        raise ValueError(
            f"CSV file has invalid formatting: {file_path}"
        ) from error

    except (UnicodeError, OSError) as error:
        raise ValueError(
            f"Unable to read CSV file: {file_path}"
        ) from error

    if data.empty:
        raise ValueError(
            f"CSV file contains no data rows: {file_path}"
        )

    return data


def load_multiple_csv(file_paths):
    if not isinstance(file_paths, dict):
        raise TypeError("file_paths must be a dictionary")

    datasets = {}

    for dataset_name, file_path in file_paths.items():
        try:
            datasets[dataset_name] = load_csv(file_path)

        except (ValueError, FileNotFoundError, TypeError) as error:
            raise ValueError(
                f"Unable to load dataset '{dataset_name}': {error}"
            ) from error

    return datasets


if __name__ == "__main__":
    file_paths = {
        "clients": "data/sample/clients.csv",
        "memberships": "data/sample/memberships.csv",
        "packages": "data/sample/packages.csv",
        "sessions": "data/sample/sessions.csv",
        "payments": "data/sample/payments.csv"
    }

    datasets = load_multiple_csv(file_paths)

    for name, data in datasets.items():
        print(f"{name.title()}: {data.shape[0]} rows, "
              f"{data.shape[1]} columns")
