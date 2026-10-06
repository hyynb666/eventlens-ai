"""Streamlit rendering for EventLens AI's interactive analytics."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from eventlens.analytics import (
    categorical_columns,
    categorical_distribution,
    correlation_matrix,
    datetime_compatible_columns,
    grouped_aggregation,
    numeric_columns,
    numeric_distribution,
    numeric_histogram,
    time_trend,
)

_AGGREGATIONS = ["count", "sum", "mean", "median", "min", "max"]
_TOP_N_OPTIONS = [5, 10, 20, 50]


def render_interactive_analysis(dataframe: pd.DataFrame) -> None:
    """Show interactive analysis tabs for columns compatible with each task."""
    st.header("Interactive Analysis")
    categorical_tab, numeric_tab, grouped_tab, trend_tab, correlation_tab = st.tabs(
        ["Categorical Distribution", "Numeric Distribution", "Grouped Aggregation", "Time Trend", "Correlation"]
    )
    with categorical_tab:
        _render_categorical(dataframe)
    with numeric_tab:
        _render_numeric(dataframe)
    with grouped_tab:
        _render_grouped(dataframe)
    with trend_tab:
        _render_time_trend(dataframe)
    with correlation_tab:
        _render_correlation(dataframe)


def _render_categorical(dataframe: pd.DataFrame) -> None:
    columns = categorical_columns(dataframe)
    if not columns:
        st.info("This dataset has no categorical or text columns to analyze.")
        return
    column = st.selectbox("Categorical or text column", columns, key="category_column")
    top_n = st.selectbox("Show top values", _TOP_N_OPTIONS, index=1, key="category_top_n")
    distribution = categorical_distribution(dataframe, column, top_n=top_n)
    if distribution.empty:
        st.info("There are no values to display for this column.")
        return
    st.caption("Missing values are grouped as (Missing). Percentages use all dataset rows.")
    st.dataframe(distribution, width="stretch", hide_index=True)
    st.bar_chart(distribution.set_index("value")[["count"]])


def _render_numeric(dataframe: pd.DataFrame) -> None:
    columns = numeric_columns(dataframe)
    if not columns:
        st.info("This dataset has no numeric columns to analyze.")
        return
    column = st.selectbox("Numeric column", columns, key="numeric_column")
    statistics = numeric_distribution(dataframe, column)
    metric_columns = st.columns(6)
    for container, label in zip(metric_columns, statistics):
        value = statistics[label]
        shown_value = "—" if value is None else (f"{value:,.3g}" if isinstance(value, float) else f"{value:,}")
        container.metric(label.capitalize(), shown_value)
    histogram = numeric_histogram(dataframe, column)
    if histogram.empty:
        st.info("This column has no non-missing numeric values for a histogram.")
    else:
        st.bar_chart(histogram.set_index("bin")[["count"]])


def _render_grouped(dataframe: pd.DataFrame) -> None:
    group_columns = categorical_columns(dataframe)
    value_columns = numeric_columns(dataframe)
    if not group_columns:
        st.info("Grouped aggregation needs at least one categorical or text column.")
        return
    if not value_columns:
        st.info("Grouped aggregation needs at least one numeric value column.")
        return
    controls = st.columns(3)
    group_column = controls[0].selectbox("Group by", group_columns, key="group_column")
    value_column = controls[1].selectbox("Numeric value", value_columns, key="group_value")
    aggregation = controls[2].selectbox("Aggregation", _AGGREGATIONS, index=2, key="group_aggregation")
    top_choice = st.selectbox("Number of groups", [5, 10, 20, 50, "All"], index=2, key="group_top_n")
    top_n = None if top_choice == "All" else int(top_choice)
    result = grouped_aggregation(dataframe, group_column, value_column, aggregation, top_n=top_n)
    if result.empty:
        st.info("No groups have usable numeric values for this aggregation.")
        return
    st.caption("Groups are sorted by aggregated value, highest first. Missing group values are shown as (Missing).")
    st.dataframe(result, width="stretch", hide_index=True)
    st.bar_chart(result.set_index("group")[["result"]])


def _render_time_trend(dataframe: pd.DataFrame) -> None:
    date_columns = datetime_compatible_columns(dataframe)
    value_columns = numeric_columns(dataframe)
    if not date_columns:
        st.info("No datetime-compatible columns were found. A date/time name or recognizable date values are needed.")
        return
    if not value_columns:
        st.info("Time trends need at least one numeric metric column.")
        return
    controls = st.columns(4)
    date_column = controls[0].selectbox("Datetime column", date_columns, key="trend_date")
    value_column = controls[1].selectbox("Numeric metric", value_columns, key="trend_value")
    aggregation = controls[2].selectbox("Aggregation", ["count", "sum", "mean", "median"], index=1, key="trend_aggregation")
    granularity = controls[3].selectbox("Time granularity", ["day", "week", "month"], key="trend_granularity")
    trend = time_trend(dataframe, date_column, value_column, aggregation, granularity)
    if trend.empty:
        st.info("No valid dates and numeric values were available for this trend.")
        return
    st.dataframe(trend, width="stretch", hide_index=True)
    st.line_chart(trend.set_index("period")[["result"]])


def _render_correlation(dataframe: pd.DataFrame) -> None:
    columns = numeric_columns(dataframe)
    if len(columns) < 2:
        st.info("Correlation analysis needs at least two numeric columns.")
        return
    method = st.selectbox("Correlation method", ["pearson", "spearman"], key="correlation_method")
    matrix = correlation_matrix(dataframe, method=method)
    if matrix.empty or matrix.isna().all().all():
        st.info("A correlation matrix is unavailable. Check for constant or entirely missing numeric columns.")
        return
    st.caption("Correlation values range from -1 to 1. Constant columns may show missing correlations.")
    styled_matrix = matrix.style.apply(_correlation_heatmap_styles, axis=None).format("{:.2f}")
    st.dataframe(styled_matrix, width="stretch")


def _correlation_heatmap_styles(matrix: pd.DataFrame) -> pd.DataFrame:
    """Build a small red-white-green CSS scale without a plotting dependency."""
    styles = pd.DataFrame("", index=matrix.index, columns=matrix.columns)
    for row_number in range(matrix.shape[0]):
        for column_number in range(matrix.shape[1]):
            value = matrix.iat[row_number, column_number]
            if pd.isna(value):
                styles.iat[row_number, column_number] = "background-color: #eeeeee; color: #555555"
                continue
            strength = min(abs(float(value)), 1.0)
            endpoint = (215, 48, 39) if value < 0 else (26, 152, 80)
            color = tuple(round(255 + strength * (channel - 255)) for channel in endpoint)
            styles.iat[row_number, column_number] = f"background-color: rgb{color}"
    return styles
