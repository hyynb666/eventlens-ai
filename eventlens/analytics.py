"""Reusable interactive-analysis functions for pandas DataFrames."""

from __future__ import annotations

import re
from typing import Literal

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype

Aggregation = Literal["count", "sum", "mean", "median", "min", "max"]
TrendAggregation = Literal["count", "sum", "mean", "median"]
TimeGranularity = Literal["day", "week", "month"]

_DATETIME_NAME_HINT = re.compile(r"date|time|timestamp|datetime|created|updated", re.IGNORECASE)
_DATETIME_VALUE_HINT = re.compile(r"\d{1,4}[-/]\d{1,2}[-/]\d{1,4}|\d{4}-\d{2}-\d{2}[T ]", re.IGNORECASE)
_ID_NAME = re.compile(r"(^|[_\W])(id|code|sku)([_\W]|$)", re.IGNORECASE)


def categorical_columns(dataframe: pd.DataFrame) -> list[str]:
    """Return text, category, and boolean columns suitable for category analysis."""
    return [
        column
        for column in dataframe.columns
        if not is_numeric_dtype(dataframe[column]) and not is_datetime64_any_dtype(dataframe[column])
    ]


def numeric_columns(dataframe: pd.DataFrame) -> list[str]:
    """Return numeric columns, excluding boolean flags."""
    return [
        column
        for column in dataframe.columns
        if is_numeric_dtype(dataframe[column]) and str(dataframe[column].dtype) != "bool"
    ]


def datetime_compatible_columns(dataframe: pd.DataFrame, success_threshold: float = 0.8) -> list[str]:
    """Find datetime columns without trying to parse arbitrary numeric or ID columns.

    Text columns are considered only when their name suggests a date/time or
    their values contain recognizable date separators. At least the requested
    share of non-missing values must parse successfully.
    """
    if not 0 < success_threshold <= 1:
        raise ValueError("success_threshold must be greater than 0 and at most 1")

    result: list[str] = []
    for column in dataframe.columns:
        values = dataframe[column]
        if is_datetime64_any_dtype(values):
            result.append(column)
            continue
        if is_numeric_dtype(values) or not (values.dtype == object or isinstance(values.dtype, pd.StringDtype)):
            continue
        name_suggests_datetime = bool(_DATETIME_NAME_HINT.search(str(column)))
        if _ID_NAME.search(str(column)) and not name_suggests_datetime:
            continue

        text_values = values.dropna().astype(str).str.strip()
        if text_values.empty:
            continue
        has_date_like_values = text_values.str.contains(_DATETIME_VALUE_HINT).mean() >= success_threshold
        if not name_suggests_datetime and not has_date_like_values:
            continue
        parsed = pd.to_datetime(text_values, errors="coerce", format="mixed", utc=True)
        if parsed.notna().mean() >= success_threshold:
            result.append(column)
    return result


def categorical_distribution(
    dataframe: pd.DataFrame, column: str, top_n: int = 10
) -> pd.DataFrame:
    """Return category counts and percentages, including missing values."""
    _require_column(dataframe, column)
    if column not in categorical_columns(dataframe):
        raise ValueError(f"Column {column!r} is not categorical or text data")
    if top_n < 1:
        raise ValueError("top_n must be at least 1")

    display_values = dataframe[column].astype(object).where(dataframe[column].notna(), "(Missing)")
    counts = display_values.value_counts(dropna=False).head(top_n)
    result = counts.rename_axis("value").rename("count").reset_index()
    result["percentage"] = result["count"] / len(dataframe) * 100 if len(dataframe) else 0.0
    return result


def numeric_distribution(dataframe: pd.DataFrame, column: str) -> dict[str, int | float | None]:
    """Return count, mean, median, sample standard deviation, min, and max."""
    _require_numeric_column(dataframe, column)
    values = dataframe[column].dropna()
    if values.empty:
        return {"count": 0, "mean": None, "median": None, "std": None, "min": None, "max": None}
    return {
        "count": int(values.count()),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "std": float(values.std()) if values.count() > 1 else None,
        "min": float(values.min()),
        "max": float(values.max()),
    }


def numeric_histogram(dataframe: pd.DataFrame, column: str, bins: int = 20) -> pd.DataFrame:
    """Return binned counts suitable for a simple Streamlit bar chart."""
    _require_numeric_column(dataframe, column)
    if bins < 1:
        raise ValueError("bins must be at least 1")
    values = dataframe[column].dropna()
    if values.empty:
        return pd.DataFrame(columns=["bin", "count"])
    binned = pd.cut(values, bins=bins, include_lowest=True)
    counts = binned.value_counts(sort=False)
    return pd.DataFrame({"bin": counts.index.astype(str), "count": counts.to_numpy()})


def grouped_aggregation(
    dataframe: pd.DataFrame,
    group_column: str,
    value_column: str,
    aggregation: Aggregation = "mean",
    top_n: int | None = None,
) -> pd.DataFrame:
    """Aggregate a numeric column by category and sort results descending."""
    _require_column(dataframe, group_column)
    _require_numeric_column(dataframe, value_column)
    if group_column == value_column:
        raise ValueError("Choose different columns for grouping and numeric values")
    if aggregation not in {"count", "sum", "mean", "median", "min", "max"}:
        raise ValueError(f"Unsupported aggregation: {aggregation}")
    if top_n is not None and top_n < 1:
        raise ValueError("top_n must be at least 1 or None")

    group_values = dataframe[group_column].astype(object).where(dataframe[group_column].notna(), "(Missing)")
    working = pd.DataFrame({"group": group_values, "value": dataframe[value_column]})
    result = working.groupby("group", dropna=False, sort=False)["value"].agg(aggregation).reset_index(name="result")
    result = result.dropna(subset=["result"]).sort_values("result", ascending=False, kind="stable").reset_index(drop=True)
    if top_n is not None:
        result = result.head(top_n)
    return result


def time_trend(
    dataframe: pd.DataFrame,
    datetime_column: str,
    value_column: str,
    aggregation: TrendAggregation = "sum",
    granularity: TimeGranularity = "day",
) -> pd.DataFrame:
    """Aggregate a numeric metric by day, week, or month on a local copy."""
    _require_column(dataframe, datetime_column)
    _require_numeric_column(dataframe, value_column)
    if aggregation not in {"count", "sum", "mean", "median"}:
        raise ValueError(f"Unsupported time-trend aggregation: {aggregation}")
    if granularity not in {"day", "week", "month"}:
        raise ValueError(f"Unsupported time granularity: {granularity}")

    parsed_dates = pd.to_datetime(dataframe[datetime_column], errors="coerce", format="mixed", utc=True)
    working = pd.DataFrame({"date": parsed_dates, "value": dataframe[value_column]}).dropna(subset=["date"])
    working["date"] = working["date"].dt.tz_convert(None)
    if granularity == "day":
        working["period"] = working["date"].dt.floor("D")
    elif granularity == "week":
        working["period"] = working["date"].dt.to_period("W").dt.start_time
    else:
        working["period"] = working["date"].dt.to_period("M").dt.start_time

    result = working.groupby("period", sort=True)["value"].agg(aggregation).dropna().reset_index(name="result")
    return result


def compare_latest_periods(
    trend: pd.DataFrame, period_column: str = "period", value_column: str = "result"
) -> dict[str, object] | None:
    """Compare the final two observed periods in a trend result, when available."""
    if period_column not in trend.columns or value_column not in trend.columns or len(trend) < 2:
        return None
    ordered = trend.sort_values(period_column, kind="stable").dropna(subset=[value_column])
    if len(ordered) < 2:
        return None
    previous, current = ordered.iloc[-2], ordered.iloc[-1]
    previous_value, current_value = float(previous[value_column]), float(current[value_column])
    absolute_change = current_value - previous_value
    return {
        "previous_period": previous[period_column],
        "current_period": current[period_column],
        "previous_value": previous_value,
        "current_value": current_value,
        "absolute_change": absolute_change,
        "percentage_change": absolute_change / previous_value * 100 if previous_value else None,
    }


def correlation_matrix(dataframe: pd.DataFrame, method: Literal["pearson", "spearman"] = "pearson") -> pd.DataFrame:
    """Return pairwise correlations for numeric columns."""
    if method not in {"pearson", "spearman"}:
        raise ValueError("method must be 'pearson' or 'spearman'")
    columns = numeric_columns(dataframe)
    if len(columns) < 2:
        return pd.DataFrame(index=columns, columns=columns, dtype=float)
    return dataframe[columns].corr(method=method)


def _require_column(dataframe: pd.DataFrame, column: str) -> None:
    if column not in dataframe.columns:
        raise ValueError(f"Column {column!r} was not found")


def _require_numeric_column(dataframe: pd.DataFrame, column: str) -> None:
    _require_column(dataframe, column)
    if column not in numeric_columns(dataframe):
        raise ValueError(f"Column {column!r} must be numeric")
