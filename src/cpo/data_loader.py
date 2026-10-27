from pathlib import Path

import numpy as np
import pandas as pd


MISSING_STRINGS = {
    "",
    "na",
    "n/a",
    "nan",
    "nil",
    "none",
    "nd",
    "-",
}


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Remove leading/trailing whitespace from Excel column names."""

    df = df.copy()

    df.columns = [
        str(col).strip()
        for col in df.columns
    ]

    return df


def normalize_missing(value):
    """Convert common Excel missing-value representations into NaN."""

    if pd.isna(value):
        return np.nan

    if isinstance(value, str):
        cleaned = value.strip()

        if cleaned.lower() in MISSING_STRINGS:
            return np.nan

        return cleaned

    return value


def coerce_date(value):
    """
    Convert either:
      - an ordinary date
      - an Excel serial date
    into pandas.Timestamp.
    """

    value = normalize_missing(value)

    if pd.isna(value):
        return pd.NaT

    # Excel serial date
    if isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        if 20_000 < float(value) < 80_000:
            return (
                pd.Timestamp("1899-12-30")
                + pd.to_timedelta(float(value), unit="D")
            )

    return pd.to_datetime(
        value,
        errors="coerce",
    )


def detect_header_row(
    filepath: Path,
    sheet_name: str,
    required_column: str,
    max_rows: int = 10,
) -> int:
    """
    Some sheets have titles/blank rows before the real header.
    Detect where a particular column name appears.
    """

    preview = pd.read_excel(
        filepath,
        sheet_name=sheet_name,
        header=None,
        nrows=max_rows,
    )

    for row_idx in range(len(preview)):
        row_values = (
            preview.iloc[row_idx]
            .astype(str)
            .str.strip()
            .tolist()
        )

        if required_column in row_values:
            return row_idx

    raise ValueError(
        f"Could not find column '{required_column}' "
        f"in first {max_rows} rows of sheet '{sheet_name}'."
    )


def read_sheet(
    filepath: Path,
    sheet_name: str,
    required_column: str | None = None,
) -> pd.DataFrame:
    """
    Robust Excel sheet reader.

    If required_column is supplied, automatically determines
    the real header row.
    """

    if required_column is None:

        df = pd.read_excel(
            filepath,
            sheet_name=sheet_name,
        )

    else:

        header_row = detect_header_row(
            filepath=filepath,
            sheet_name=sheet_name,
            required_column=required_column,
        )

        df = pd.read_excel(
            filepath,
            sheet_name=sheet_name,
            header=header_row,
        )

    df = clean_column_names(df)

    return df


def unique_non_missing(series: pd.Series) -> str:
    """Return unique, nonmissing values separated by '; '."""

    results = []

    for value in series:

        value = normalize_missing(value)

        if pd.isna(value):
            continue

        value = str(value)

        if value not in results:
            results.append(value)

    return "; ".join(results)