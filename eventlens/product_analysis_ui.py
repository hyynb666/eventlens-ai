"""Streamlit interface for reusable product analytics."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from eventlens.analytics import categorical_columns, datetime_compatible_columns, numeric_columns
from eventlens.product_analytics import (
    active_users,
    conversion_rate,
    funnel_analysis,
    revenue_metrics,
    revenue_over_time,
    retention_analysis,
)


def render_product_analytics(dataframe: pd.DataFrame) -> None:
    """Render active user, revenue, conversion, funnel, and retention tools."""
    st.header("Product Analytics")
    date_columns = datetime_compatible_columns(dataframe)
    user_columns = [column for column in dataframe.columns if column not in date_columns]
    event_columns = [column for column in categorical_columns(dataframe) if column not in date_columns]

    active_tab, revenue_tab, conversion_tab, funnel_tab, retention_tab = st.tabs(
        ["Active Users", "Revenue", "Conversion", "Funnel", "Retention"]
    )
    with active_tab:
        _render_active_users(dataframe, user_columns, date_columns)
    with revenue_tab:
        _render_revenue(dataframe, user_columns, date_columns, event_columns)
    with conversion_tab:
        _render_conversion(dataframe, user_columns, date_columns, event_columns)
    with funnel_tab:
        _render_funnel(dataframe, user_columns, date_columns, event_columns)
    with retention_tab:
        _render_retention(dataframe, user_columns, date_columns)


def _render_active_users(dataframe: pd.DataFrame, user_columns: list[str], date_columns: list[str]) -> None:
    if not _has_user_and_date_columns(user_columns, date_columns):
        st.info("Active user metrics need a user identifier and a datetime-compatible column.")
        return
    controls = st.columns(3)
    user_column = controls[0].selectbox("User identifier", user_columns, key="active_user_id")
    datetime_column = controls[1].selectbox("Datetime column", date_columns, key="active_datetime")
    period = controls[2].selectbox("Active user period", ["day", "week", "month"], key="active_period")
    metric_name = {"day": "DAU", "week": "WAU", "month": "MAU"}[period]
    result = active_users(dataframe, user_column, datetime_column, period)
    if result.empty:
        st.info("No rows have both a user identifier and a valid timestamp.")
        return
    st.caption(f"{metric_name} counts distinct users with at least one valid event in each period.")
    st.dataframe(result, width="stretch", hide_index=True)
    st.line_chart(result.set_index("period")[["active_users"]])


def _render_revenue(
    dataframe: pd.DataFrame,
    user_columns: list[str],
    date_columns: list[str],
    event_columns: list[str],
) -> None:
    if not _has_user_and_date_columns(user_columns, date_columns):
        st.info("Revenue metrics need a user identifier and a datetime-compatible column.")
        return
    controls = st.columns(3)
    user_column = controls[0].selectbox("User identifier", user_columns, key="revenue_user_id")
    datetime_column = controls[1].selectbox("Datetime column", date_columns, key="revenue_datetime")
    numeric = [column for column in numeric_columns(dataframe) if column != user_column]
    if not numeric:
        st.info("No numeric value column other than the selected user identifier is available.")
        return
    revenue_column = controls[2].selectbox("Revenue or value column", numeric, key="revenue_value")
    filter_to_event = st.checkbox("Filter revenue to a specific event", key="revenue_filter_enabled")
    event_column = None
    event_value = None
    if filter_to_event:
        available_events = [column for column in event_columns if column != user_column]
        if not available_events:
            st.info("Choose an event column to filter revenue.")
            return
        event_column = st.selectbox("Event column", available_events, key="revenue_event_column")
        values = _event_values(dataframe, event_column)
        if not values:
            st.info("The selected event column has no non-missing values.")
            return
        event_value = st.selectbox("Revenue event", values, format_func=str, key="revenue_event_value")

    metrics = revenue_metrics(dataframe, user_column, datetime_column, revenue_column, event_column, event_value)
    metric_columns = st.columns(5)
    metric_columns[0].metric("Total revenue", _format_number(metrics["total_revenue"]))
    metric_columns[1].metric("Active users", f"{metrics['active_users']:,}")
    metric_columns[2].metric("Paying users", f"{metrics['paying_users']:,}")
    metric_columns[3].metric("ARPU", _format_number(metrics["arpu"]))
    metric_columns[4].metric("ARPPU", _format_number(metrics["arppu"]))
    st.caption("ARPU divides revenue by all users with valid dated activity. ARPPU divides by users with positive filtered revenue.")

    period = st.selectbox("Revenue trend granularity", ["day", "week", "month"], key="revenue_period")
    trend = revenue_over_time(
        dataframe, datetime_column, revenue_column, period, event_column, event_value
    )
    if trend.empty:
        st.info("No revenue rows have both a valid timestamp and numeric value.")
        return
    st.dataframe(trend, width="stretch", hide_index=True)
    st.line_chart(trend.set_index("period")[["revenue"]])


def _render_conversion(
    dataframe: pd.DataFrame,
    user_columns: list[str],
    date_columns: list[str],
    event_columns: list[str],
) -> None:
    if not _has_user_and_date_columns(user_columns, date_columns):
        st.info("Conversion analysis needs a user identifier and a datetime-compatible column.")
        return
    if not event_columns:
        st.info("Conversion analysis needs a categorical event column.")
        return
    controls = st.columns(4)
    user_column = controls[0].selectbox("User identifier", user_columns, key="conversion_user_id")
    datetime_column = controls[1].selectbox("Datetime column", date_columns, key="conversion_datetime")
    available_events = [column for column in event_columns if column != user_column]
    if not available_events:
        st.info("Conversion analysis needs an event column separate from the user identifier.")
        return
    event_column = controls[2].selectbox("Event column", available_events, key="conversion_event_column")
    values = _event_values(dataframe, event_column)
    if len(values) < 2:
        st.info("The selected event column needs at least two different non-missing event values.")
        return
    starting_event = controls[3].selectbox("Starting event", values, format_func=str, key="conversion_start")
    conversion_event = st.selectbox("Conversion event", values, index=len(values) - 1, format_func=str, key="conversion_finish")
    result = conversion_rate(
        dataframe, user_column, datetime_column, event_column, starting_event, conversion_event
    )
    metrics = st.columns(3)
    metrics[0].metric("Users who started", f"{result['starting_users']:,}")
    metrics[1].metric("Users who converted later", f"{result['converted_users']:,}")
    rate = result["conversion_rate"]
    metrics[2].metric("Conversion rate", "—" if rate is None else f"{rate:.1%}")
    st.caption("Each user is counted once. A conversion only counts when its timestamp is later than that user's first starting event.")
    summary = pd.DataFrame(
        [
            {"event": str(starting_event), "unique_users": result["starting_users"]},
            {"event": str(conversion_event), "unique_users": result["converted_users"]},
        ]
    )
    st.dataframe(summary, width="stretch", hide_index=True)


def _render_funnel(
    dataframe: pd.DataFrame,
    user_columns: list[str],
    date_columns: list[str],
    event_columns: list[str],
) -> None:
    if not _has_user_and_date_columns(user_columns, date_columns):
        st.info("Funnel analysis needs a user identifier and a datetime-compatible column.")
        return
    if not event_columns:
        st.info("Funnel analysis needs a categorical event column.")
        return
    controls = st.columns(3)
    user_column = controls[0].selectbox("User identifier", user_columns, key="funnel_user_id")
    datetime_column = controls[1].selectbox("Datetime column", date_columns, key="funnel_datetime")
    available_events = [column for column in event_columns if column != user_column]
    if not available_events:
        st.info("Funnel analysis needs an event column separate from the user identifier.")
        return
    event_column = controls[2].selectbox("Event column", available_events, key="funnel_event_column")
    values = _event_values(dataframe, event_column)
    if not values:
        st.info("The selected event column has no non-missing values.")
        return
    default_step_count = max(2, min(3, len(values)))
    step_count = st.select_slider(
        "Number of funnel steps", options=[2, 3, 4, 5], value=default_step_count, key="funnel_step_count"
    )
    step_columns = st.columns(step_count)
    steps = [
        step_columns[index].selectbox(
            f"Step {index + 1}",
            values,
            index=min(index, len(values) - 1),
            format_func=str,
            key=f"funnel_step_{index + 1}",
        )
        for index in range(step_count)
    ]
    result = funnel_analysis(dataframe, user_column, datetime_column, event_column, steps)
    if result.empty or result["users"].sum() == 0:
        st.info("No users reached the first funnel step in a valid event sequence.")
        return
    st.caption("A user advances only when each selected event occurs later than the previous step.")
    st.dataframe(result, width="stretch", hide_index=True)
    st.bar_chart(result.set_index("event")[["users"]])


def _render_retention(dataframe: pd.DataFrame, user_columns: list[str], date_columns: list[str]) -> None:
    if not _has_user_and_date_columns(user_columns, date_columns):
        st.info("Retention analysis needs a user identifier and a datetime-compatible column.")
        return
    controls = st.columns(2)
    user_column = controls[0].selectbox("User identifier", user_columns, key="retention_user_id")
    datetime_column = controls[1].selectbox("Datetime column", date_columns, key="retention_datetime")
    result = retention_analysis(dataframe, user_column, datetime_column)
    if result.empty:
        st.info("No rows have both a user identifier and a valid timestamp.")
        return
    st.caption("Retention means activity on exactly D+1, D+7, or D+30. Recent cohorts without a complete observation window show as unavailable.")
    display = result.copy()
    retention_columns = ["D1", "D7", "D30"]
    for column in retention_columns:
        display[column] = display[column].map(
            lambda value: "Unavailable" if pd.isna(value) else f"{value:.1f}%"
        )
    display = display.rename(
        columns={"D1": "D1 retention", "D7": "D7 retention", "D30": "D30 retention"}
    )
    st.dataframe(display, width="stretch", hide_index=True)


def _has_user_and_date_columns(user_columns: list[str], date_columns: list[str]) -> bool:
    return bool(user_columns and date_columns)


def _event_values(dataframe: pd.DataFrame, event_column: str) -> list[object]:
    return dataframe[event_column].dropna().unique().tolist()


def _format_number(value: int | float | None) -> str:
    return "—" if value is None else f"{value:,.2f}"
