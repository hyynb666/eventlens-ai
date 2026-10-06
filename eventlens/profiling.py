"""Simple, reusable exploratory profiling functions for pandas DataFrames."""

from __future__ import annotations

import pandas as pd
from pandas.api.types import is_numeric_dtype


def profile_dataset(dataframe: pd.DataFrame) -> dict[str, int | float]:
    """Return basic dataset dimensions, duplicate count, and memory use in bytes."""
    memory_bytes = int(dataframe.memory_usage(index=True, deep=True).sum())
    return {
        "row_count": int(dataframe.shape[0]),
        "column_count": int(dataframe.shape[1]),
        "duplicate_row_count": int(dataframe.duplicated().sum()),
        "memory_usage_bytes": memory_bytes,
    }


def profile_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Describe each column's dtype, missing values, and distinct values."""
    row_count = len(dataframe)
    missing_count = dataframe.isna().sum()
    result = pd.DataFrame(
        {
            "column": dataframe.columns,
            "dtype": [str(dtype) for dtype in dataframe.dtypes],
            "missing_count": missing_count.to_numpy(),
            "missing_percentage": (missing_count / row_count * 100).fillna(0).to_numpy(),
            "unique_count": dataframe.nunique(dropna=True).to_numpy(),
        }
    )
    return result


def numeric_summary(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Summarize count, mean, standard deviation, min, median, and max."""
    columns = [name for name in dataframe.columns if is_numeric_dtype(dataframe[name])]
    labels = ["count", "mean", "std", "min", "median", "max"]
    if not columns:
        return pd.DataFrame(columns=labels)
    summary = dataframe[columns].describe().T
    summary = summary.rename(columns={"50%": "median"})
    return summary.reindex(columns=labels)


def categorical_summary(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Summarize distinct values and the most frequent value for text-like columns."""
    columns = [
        name
        for name in dataframe.columns
        if not is_numeric_dtype(dataframe[name]) and str(dataframe[name].dtype) != "datetime64[ns]"
    ]
    rows: list[dict[str, object]] = []
    for name in columns:
        values = dataframe[name].dropna()
        counts = values.value_counts()
        rows.append(
            {
                "column": name,
                "unique_count": int(values.nunique()),
                "most_common_value": counts.index[0] if not counts.empty else None,
                "most_common_frequency": int(counts.iloc[0]) if not counts.empty else 0,
            }
        )
    return pd.DataFrame(rows, columns=["column", "unique_count", "most_common_value", "most_common_frequency"])


def missing_summary(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Return only columns with missing values, ordered by missing percentage."""
    missing_count = dataframe.isna().sum()
    result = pd.DataFrame(
        {
            "column": dataframe.columns,
            "missing_count": missing_count.to_numpy(),
            "missing_percentage": (missing_count / len(dataframe) * 100).fillna(0).to_numpy(),
        }
    )
    return result.loc[result["missing_count"] > 0].sort_values("missing_percentage", ascending=False).reset_index(drop=True)
