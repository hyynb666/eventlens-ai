"""Reusable product and business analytics for user-event datasets."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import pandas as pd

from eventlens.analytics import numeric_columns

ActivePeriod = Literal["day", "week", "month"]
RetentionDays = (1, 7, 30)


def active_users(
    dataframe: pd.DataFrame,
    user_column: str,
    datetime_column: str,
    period: ActivePeriod = "day",
) -> pd.DataFrame:
    """Count distinct non-missing users with a valid event timestamp per period."""
    _require_columns(dataframe, [user_column, datetime_column])
    _require_distinct_columns(user_column, datetime_column)
    if period not in {"day", "week", "month"}:
        raise ValueError("period must be 'day', 'week', or 'month'")
    working = _valid_user_events(dataframe, user_column, datetime_column)
    working["period"] = _period_starts(working["timestamp"], period)
    if working.empty:
        return pd.DataFrame(columns=["period", "active_users"])
    return (
        working.groupby("period", sort=True)[user_column]
        .nunique()
        .rename("active_users")
        .reset_index()
    )


def revenue_metrics(
    dataframe: pd.DataFrame,
    user_column: str,
    datetime_column: str,
    revenue_column: str,
    event_column: str | None = None,
    purchase_event: object | None = None,
) -> dict[str, int | float | None]:
    """Calculate revenue, ARPU, and ARPPU with optional purchase-event filtering.

    ARPU divides filtered revenue by distinct users in the full dataset.
    ARPPU divides it by users with positive revenue in the filtered rows.
    """
    _require_columns(dataframe, [user_column, datetime_column, revenue_column])
    _require_distinct_columns(user_column, datetime_column, revenue_column, *([event_column] if event_column else []))
    _require_numeric_column(dataframe, revenue_column)
    revenue_rows = _event_filter(dataframe, event_column, purchase_event)
    revenue_rows = _with_valid_timestamps(revenue_rows, datetime_column)
    valid_revenue = revenue_rows[revenue_column].dropna()
    total_revenue = float(valid_revenue.sum())
    active_user_count = int(_valid_user_events(dataframe, user_column, datetime_column)[user_column].nunique())
    paying_users = int(
        revenue_rows.loc[
            revenue_rows[user_column].notna() & (revenue_rows[revenue_column] > 0), user_column
        ].nunique()
    )
    return {
        "total_revenue": total_revenue,
        "active_users": active_user_count,
        "paying_users": paying_users,
        "arpu": total_revenue / active_user_count if active_user_count else None,
        "arppu": total_revenue / paying_users if paying_users else None,
    }


def revenue_over_time(
    dataframe: pd.DataFrame,
    datetime_column: str,
    revenue_column: str,
    period: ActivePeriod = "day",
    event_column: str | None = None,
    purchase_event: object | None = None,
) -> pd.DataFrame:
    """Sum non-missing revenue by day, week, or month, skipping invalid dates."""
    _require_columns(dataframe, [datetime_column, revenue_column])
    _require_distinct_columns(datetime_column, revenue_column, *([event_column] if event_column else []))
    _require_numeric_column(dataframe, revenue_column)
    if period not in {"day", "week", "month"}:
        raise ValueError("period must be 'day', 'week', or 'month'")
    revenue_rows = _event_filter(dataframe, event_column, purchase_event)
    dates = pd.to_datetime(revenue_rows[datetime_column], errors="coerce", format="mixed", utc=True).dt.tz_convert(None)
    working = pd.DataFrame({"timestamp": dates, "revenue": revenue_rows[revenue_column]}).dropna(subset=["timestamp", "revenue"])
    if working.empty:
        return pd.DataFrame(columns=["period", "revenue"])
    working["period"] = _period_starts(working["timestamp"], period)
    return working.groupby("period", sort=True)["revenue"].sum().reset_index()


def conversion_rate(
    dataframe: pd.DataFrame,
    user_column: str,
    datetime_column: str,
    event_column: str,
    starting_event: object,
    conversion_event: object,
) -> dict[str, int | float | None]:
    """Measure distinct users converting after their first starting event.

    Event order is strict: a conversion at the same timestamp as the start
    does not count as occurring after it.
    """
    _require_columns(dataframe, [user_column, datetime_column, event_column])
    _require_distinct_columns(user_column, datetime_column, event_column)
    working = _valid_user_events(dataframe, user_column, datetime_column, [event_column])
    working = working.loc[working[event_column].notna()]
    starts = working.loc[working[event_column] == starting_event].groupby(user_column, sort=False)["timestamp"].min()
    conversions = working.loc[working[event_column] == conversion_event].groupby(user_column, sort=False)["timestamp"].max()
    starting_users = int(starts.size)
    converted_users = int((conversions.reindex(starts.index) > starts).fillna(False).sum())
    return {
        "starting_users": starting_users,
        "converted_users": converted_users,
        "conversion_rate": converted_users / starting_users if starting_users else None,
    }


def funnel_analysis(
    dataframe: pd.DataFrame,
    user_column: str,
    datetime_column: str,
    event_column: str,
    steps: Sequence[object],
) -> pd.DataFrame:
    """Count unique users reaching each ordered event step after prior steps."""
    _require_columns(dataframe, [user_column, datetime_column, event_column])
    _require_distinct_columns(user_column, datetime_column, event_column)
    if not 2 <= len(steps) <= 5:
        raise ValueError("A funnel must contain between 2 and 5 event steps")
    working = _valid_user_events(dataframe, user_column, datetime_column, [event_column])
    working = working.loc[working[event_column].notna()].sort_values("timestamp", kind="stable")

    reached_by_step = [0] * len(steps)
    for _, user_events in working.groupby(user_column, sort=False):
        event_times = list(zip(user_events[event_column].tolist(), user_events["timestamp"].tolist()))
        next_position = 0
        last_timestamp: pd.Timestamp | None = None
        for step_index, event_name in enumerate(steps):
            match = next(
                (
                    position
                    for position in range(next_position, len(event_times))
                    if event_times[position][0] == event_name
                    and (last_timestamp is None or event_times[position][1] > last_timestamp)
                ),
                None,
            )
            if match is None:
                break
            reached_by_step[step_index] += 1
            last_timestamp = event_times[match][1]
            next_position = match + 1

    first_count = reached_by_step[0]
    rows: list[dict[str, object]] = []
    for index, event_name in enumerate(steps):
        previous_count = reached_by_step[index - 1] if index else None
        current_count = reached_by_step[index]
        rows.append(
            {
                "step": index + 1,
                "event": event_name,
                "users": current_count,
                "conversion_from_previous_pct": (
                    current_count / previous_count * 100 if previous_count else (None if index else 100.0)
                ),
                "conversion_from_first_pct": current_count / first_count * 100 if first_count else None,
            }
        )
    return pd.DataFrame(
        rows,
        columns=["step", "event", "users", "conversion_from_previous_pct", "conversion_from_first_pct"],
    )


def retention_analysis(
    dataframe: pd.DataFrame,
    user_column: str,
    datetime_column: str,
) -> pd.DataFrame:
    """Calculate exact-day D1, D7, and D30 retention for first-activity cohorts.

    A retention value is unavailable (None) if the dataset's latest valid
    activity date is earlier than that cohort's retention target date.
    """
    _require_columns(dataframe, [user_column, datetime_column])
    _require_distinct_columns(user_column, datetime_column)
    working = _valid_user_events(dataframe, user_column, datetime_column)
    columns = ["cohort_date", "cohort_size", "D1", "D7", "D30"]
    if working.empty:
        return pd.DataFrame(columns=columns)

    working["activity_date"] = working["timestamp"].dt.floor("D")
    first_dates = working.groupby(user_column, sort=False)["activity_date"].min()
    activity_dates = working.groupby(user_column, sort=False)["activity_date"].agg(set)
    cohort_members: dict[pd.Timestamp, list[object]] = {}
    for user, first_date in first_dates.items():
        cohort_members.setdefault(first_date, []).append(user)
    latest_activity = working["activity_date"].max()

    rows: list[dict[str, object]] = []
    for cohort_date, users in sorted(cohort_members.items()):
        row: dict[str, object] = {"cohort_date": cohort_date, "cohort_size": len(users)}
        for days in RetentionDays:
            target_date = cohort_date + pd.Timedelta(days=days)
            if target_date > latest_activity:
                row[f"D{days}"] = None
                continue
            retained_users = sum(target_date in activity_dates[user] for user in users)
            row[f"D{days}"] = retained_users / len(users) * 100
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def _valid_user_events(
    dataframe: pd.DataFrame,
    user_column: str,
    datetime_column: str,
    additional_columns: Sequence[str] = (),
) -> pd.DataFrame:
    _require_columns(dataframe, [user_column, datetime_column, *additional_columns])
    _require_distinct_columns(user_column, datetime_column, *additional_columns)
    source = dataframe.reset_index(drop=True)
    parsed_dates = pd.to_datetime(source[datetime_column], errors="coerce", format="mixed", utc=True)
    working = source[[user_column, *additional_columns]].copy()
    working.insert(1, "timestamp", parsed_dates)
    working = working.loc[working[user_column].notna() & working["timestamp"].notna()].copy()
    working["timestamp"] = working["timestamp"].dt.tz_convert(None)
    return working


def _with_valid_timestamps(dataframe: pd.DataFrame, datetime_column: str) -> pd.DataFrame:
    _require_columns(dataframe, [datetime_column])
    parsed_dates = pd.to_datetime(dataframe[datetime_column], errors="coerce", format="mixed", utc=True)
    valid_rows = dataframe.loc[parsed_dates.notna()].copy()
    return valid_rows


def _period_starts(timestamps: pd.Series, period: ActivePeriod) -> pd.Series:
    if period == "day":
        return timestamps.dt.floor("D")
    if period == "week":
        return timestamps.dt.to_period("W").dt.start_time
    if period == "month":
        return timestamps.dt.to_period("M").dt.start_time
    raise ValueError("period must be 'day', 'week', or 'month'")


def _event_filter(
    dataframe: pd.DataFrame,
    event_column: str | None,
    event_value: object | None,
) -> pd.DataFrame:
    if event_column is None:
        if event_value is not None:
            raise ValueError("event_column is required when filtering to an event value")
        return dataframe
    _require_columns(dataframe, [event_column])
    if event_value is None:
        raise ValueError("Choose an event value or disable event filtering")
    return dataframe.loc[dataframe[event_column] == event_value]


def _require_numeric_column(dataframe: pd.DataFrame, column: str) -> None:
    _require_columns(dataframe, [column])
    if column not in numeric_columns(dataframe):
        raise ValueError(f"Column {column!r} must be numeric")


def _require_columns(dataframe: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in dataframe.columns]
    if missing:
        raise ValueError(f"Required column(s) not found: {', '.join(map(str, missing))}")


def _require_distinct_columns(*columns: str) -> None:
    if len(set(columns)) != len(columns):
        raise ValueError("Choose a different column for each selected role")
