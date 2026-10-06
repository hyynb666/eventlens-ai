"""Validated dispatch from structured requests to approved EventLens functions."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd

from eventlens.analytics import (
    categorical_columns,
    categorical_distribution,
    correlation_matrix,
    datetime_compatible_columns,
    grouped_aggregation,
    numeric_columns,
    numeric_distribution,
    time_trend,
)
from eventlens.product_analytics import (
    active_users,
    conversion_rate,
    funnel_analysis,
    revenue_metrics,
    retention_analysis,
)
from eventlens.profiling import missing_summary, profile_columns, profile_dataset


@dataclass
class AnalysisResult:
    """Streamlit-neutral result returned by a controlled analytics tool."""

    title: str
    summary: str
    request: dict[str, Any] | None = None
    table: pd.DataFrame | None = None
    metrics: dict[str, int | float | str | None] = field(default_factory=dict)
    chart_type: Literal["bar", "line"] | None = None
    chart_data: pd.DataFrame | None = None
    warnings: list[str] = field(default_factory=list)
    error: str | None = None


_REQUEST_FIELDS: dict[str, tuple[set[str], set[str]]] = {
    "dataset_overview": (set(), set()),
    "missing_summary": (set(), set()),
    "numeric_summary": ({"column"}, set()),
    "categorical_distribution": ({"column"}, {"top_n"}),
    "grouped_aggregation": ({"group_by", "value_column", "aggregation"}, set()),
    "time_trend": ({"datetime_column", "value_column", "aggregation", "granularity"}, set()),
    "correlation": ({"columns"}, {"method"}),
    "active_users": ({"metric", "user_column", "datetime_column"}, set()),
    "revenue_metrics": (
        {"user_column", "datetime_column", "revenue_column"},
        {"metric", "event_column", "purchase_event"},
    ),
    "conversion": (
        {"user_column", "datetime_column", "event_column", "starting_event", "conversion_event"},
        set(),
    ),
    "funnel": ({"user_column", "datetime_column", "event_column", "steps"}, set()),
    "retention": ({"user_column", "datetime_column"}, {"metric"}),
}

SUPPORTED_ANALYTICS_INTENTS = tuple(_REQUEST_FIELDS)


def validate_analysis_request(
    dataframe: pd.DataFrame,
    request: object,
    question: str = "",
    *,
    planned: bool = False,
) -> str | None:
    """Validate one approved request without running an analytics calculation.

    Guided plans use this to check every step before any step executes.
    """
    if not isinstance(request, dict) or not isinstance(request.get("intent"), str):
        return "The analysis request must be a JSON object with an intent."
    intent = request["intent"]
    if intent not in _REQUEST_FIELDS:
        return "This analysis type is not supported by the current EventLens tools."
    required, optional = _REQUEST_FIELDS[intent]
    fields = set(request) - {"intent"}
    if fields - required - optional:
        return "The analysis request contained unsupported fields."
    missing = required - fields
    if missing:
        return f"The request is missing required information: {', '.join(sorted(missing))}."
    try:
        _validate_parameters(dataframe, request, question, planned=planned)
    except (ValueError, KeyError, TypeError, IndexError) as error:
        return str(error)
    return None


def _validate_parameters(
    dataframe: pd.DataFrame, request: dict[str, Any], question: str, *, planned: bool = False
) -> None:
    """Pure request validation shared by the plan preflight and dispatcher."""
    intent = request["intent"]
    if intent in {"dataset_overview", "missing_summary"}:
        return
    if intent == "numeric_summary":
        column = _require_numeric(dataframe, request["column"])
        _check_metric_ambiguity(dataframe, column, question)
    elif intent == "categorical_distribution":
        column = _require_categorical(dataframe, request["column"])
        if not planned:
            _check_column_ambiguity(column, categorical_columns(dataframe), question, "categorical column")
        top_n = request.get("top_n", 10)
        if isinstance(top_n, bool) or not isinstance(top_n, int) or top_n not in {5, 10, 20, 50}:
            raise ValueError("Top N must be 5, 10, 20, or 50.")
    elif intent == "grouped_aggregation":
        group = _require_categorical(dataframe, request["group_by"])
        value = _require_numeric(dataframe, request["value_column"])
        if group == value:
            raise ValueError("Choose different columns for grouping and numeric values.")
        if not planned:
            _check_column_ambiguity(group, categorical_columns(dataframe), question, "grouping column")
        _check_metric_ambiguity(dataframe, value, question)
        _one_of(request["aggregation"], {"count", "sum", "mean", "median", "min", "max"}, "aggregation")
    elif intent == "time_trend":
        date = _require_datetime(dataframe, request["datetime_column"])
        value = _require_numeric(dataframe, request["value_column"])
        _check_column_ambiguity(date, datetime_compatible_columns(dataframe), question, "datetime column")
        _check_metric_ambiguity(dataframe, value, question)
        _one_of(request["aggregation"], {"count", "sum", "mean", "median"}, "aggregation")
        _one_of(request["granularity"], {"day", "week", "month"}, "time granularity")
    elif intent == "correlation":
        columns = request["columns"]
        if not isinstance(columns, list) or len(columns) < 2 or len(set(columns)) != len(columns):
            raise ValueError("Correlation needs at least two different numeric columns.")
        invalid = [column for column in columns if column not in numeric_columns(dataframe)]
        if invalid:
            raise ValueError(f"Correlation columns must be numeric: {', '.join(map(str, invalid))}.")
        if len(numeric_columns(dataframe)) > 2 and any(not _question_mentions(question, str(column)) for column in columns):
            raise ValueError("Please specify at least two numeric column names for correlation.")
        _one_of(request.get("method", "pearson"), {"pearson", "spearman"}, "correlation method")
    elif intent in {"active_users", "revenue_metrics", "conversion", "funnel", "retention"}:
        user = _require_user(dataframe, request["user_column"])
        date = _require_datetime(dataframe, request["datetime_column"])
        _check_user_ambiguity(dataframe, user, question)
        _check_datetime_ambiguity(dataframe, date, question)
        if intent == "active_users":
            _one_of(str(request["metric"]).lower(), {"dau", "wau", "mau"}, "active-user metric")
        elif intent == "revenue_metrics":
            revenue = _require_numeric(dataframe, request["revenue_column"])
            _check_metric_ambiguity(dataframe, revenue, question, metric_context=True)
            _one_of(request.get("metric", "total_revenue"), {"total_revenue", "arpu", "arppu"}, "revenue metric")
            _optional_event_filter(dataframe, request)
        elif intent in {"conversion", "funnel"}:
            event = _require_event_column(dataframe, request["event_column"])
            _check_event_column_ambiguity(dataframe, event, question)
            if intent == "conversion":
                _resolve_event_value(dataframe, event, request["starting_event"])
                _resolve_event_value(dataframe, event, request["conversion_event"])
            else:
                steps = request["steps"]
                if not isinstance(steps, list) or not 2 <= len(steps) <= 5:
                    raise ValueError("A funnel must contain between 2 and 5 event steps.")
                for step in steps:
                    _resolve_event_value(dataframe, event, step)
        elif request.get("metric") is not None:
            _one_of(str(request["metric"]).lower(), {"d1", "d7", "d30"}, "retention metric")


def run_analysis_request(
    dataframe: pd.DataFrame,
    request: object,
    question: str = "",
) -> AnalysisResult:
    """Validate a JSON-like request, then call only an explicitly supported tool."""
    if not isinstance(request, dict) or not isinstance(request.get("intent"), str):
        return _error_result("The analysis request must be a JSON object with an intent.")

    intent = request["intent"]
    if intent == "clarification":
        message = request.get("message")
        if not isinstance(message, str) or not message.strip():
            return _error_result("I need one more detail before I can analyze this question.")
        return AnalysisResult("Clarification needed", message.strip(), request=dict(request), error=message.strip())
    if intent == "unsupported":
        return AnalysisResult(
            "Unsupported question",
            "This question is not supported by the current EventLens analytics tools.",
            request=dict(request),
            error="unsupported",
        )
    if intent not in _REQUEST_FIELDS:
        return _error_result("This analysis type is not supported by the current EventLens tools.", dict(request))

    validation_error = validate_analysis_request(dataframe, request, question)
    if validation_error:
        return _error_result(validation_error, dict(request))

    required_fields, optional_fields = _REQUEST_FIELDS[intent]
    provided_fields = set(request) - {"intent"}
    unknown_fields = provided_fields - required_fields - optional_fields
    missing_fields = required_fields - provided_fields
    if unknown_fields:
        return _error_result("The analysis request contained unsupported fields.", dict(request))
    if missing_fields:
        return _error_result(f"The request is missing required information: {', '.join(sorted(missing_fields))}.", dict(request))

    try:
        return _dispatch(dataframe, request, question)
    except ValueError as error:
        return _error_result(str(error), dict(request))
    except (KeyError, TypeError, IndexError) as error:
        return _error_result(f"The request could not be applied to this dataset: {error}.", dict(request))


def _dispatch(dataframe: pd.DataFrame, request: dict[str, Any], question: str) -> AnalysisResult:
    intent = request["intent"]
    if intent == "dataset_overview":
        overview = profile_dataset(dataframe)
        missing = missing_summary(dataframe)
        columns = profile_columns(dataframe)
        warnings = ["No missing values were found."] if missing.empty else []
        return AnalysisResult(
            "Dataset overview",
            f"The dataset has {overview['row_count']:,} rows, {overview['column_count']:,} columns, and {overview['duplicate_row_count']:,} duplicate rows.",
            request,
            table=missing if not missing.empty else columns,
            metrics={
                "rows": overview["row_count"],
                "columns": overview["column_count"],
                "duplicate_rows": overview["duplicate_row_count"],
                "memory_bytes": overview["memory_usage_bytes"],
            },
            warnings=warnings,
        )
    if intent == "missing_summary":
        table = missing_summary(dataframe)
        summary = "Columns with missing values are listed below." if not table.empty else "No missing values were found."
        return AnalysisResult("Missing values", summary, request, table=table)
    if intent == "numeric_summary":
        column = _require_numeric(dataframe, request["column"])
        _check_metric_ambiguity(dataframe, column, question)
        statistics = numeric_distribution(dataframe, column)
        table = pd.DataFrame([statistics])
        return AnalysisResult(
            f"Numeric summary: {column}",
            f"Calculated statistics for {column!r} using {statistics['count']:,} non-missing values.",
            request,
            table=table,
        )
    if intent == "categorical_distribution":
        column = _require_categorical(dataframe, request["column"])
        _check_column_ambiguity(column, categorical_columns(dataframe), question, "categorical column")
        top_n = request.get("top_n", 10)
        if isinstance(top_n, bool) or not isinstance(top_n, int) or top_n not in {5, 10, 20, 50}:
            raise ValueError("Top N must be 5, 10, 20, or 50.")
        table = categorical_distribution(dataframe, column, top_n)
        return AnalysisResult(
            f"Categorical distribution: {column}",
            f"Listed the most common values in {column!r}; percentages are based on all dataset rows.",
            request,
            table=table,
            chart_type="bar",
            chart_data=table.set_index("value")[["count"]] if not table.empty else None,
            warnings=["This column has no values to summarize."] if table.empty else [],
        )
    if intent == "grouped_aggregation":
        group_by = _require_categorical(dataframe, request["group_by"])
        value_column = _require_numeric(dataframe, request["value_column"])
        _check_column_ambiguity(group_by, categorical_columns(dataframe), question, "grouping column")
        _check_metric_ambiguity(dataframe, value_column, question)
        aggregation = _one_of(request["aggregation"], {"count", "sum", "mean", "median", "min", "max"}, "aggregation")
        table = grouped_aggregation(dataframe, group_by, value_column, aggregation, top_n=50)
        summary = f"Computed {aggregation} {value_column!r} for {len(table):,} groups by {group_by!r}."
        if not table.empty:
            first = table.iloc[0]
            summary += f" The highest result is {first['result']:,.2f} for {first['group']}."
        return AnalysisResult(
            f"{aggregation.capitalize()} {value_column} by {group_by}",
            summary,
            request,
            table=table,
            chart_type="bar",
            chart_data=table.set_index("group")[["result"]] if not table.empty else None,
        )
    if intent == "time_trend":
        datetime_column = _require_datetime(dataframe, request["datetime_column"])
        value_column = _require_numeric(dataframe, request["value_column"])
        _check_column_ambiguity(datetime_column, datetime_compatible_columns(dataframe), question, "datetime column")
        _check_metric_ambiguity(dataframe, value_column, question)
        aggregation = _one_of(request["aggregation"], {"count", "sum", "mean", "median"}, "aggregation")
        granularity = _one_of(request["granularity"], {"day", "week", "month"}, "time granularity")
        table = time_trend(dataframe, datetime_column, value_column, aggregation, granularity)
        return AnalysisResult(
            f"{granularity.capitalize()} {aggregation} trend for {value_column}",
            f"Computed {aggregation} {value_column!r} by {granularity} from valid timestamps.",
            request,
            table=table,
            chart_type="line",
            chart_data=table.set_index("period")[["result"]] if not table.empty else None,
            warnings=["No valid dates and values were available for this trend."] if table.empty else [],
        )
    if intent == "correlation":
        columns = request["columns"]
        if not isinstance(columns, list) or len(columns) < 2 or len(set(columns)) != len(columns):
            raise ValueError("Correlation needs at least two different numeric columns.")
        invalid = [column for column in columns if column not in numeric_columns(dataframe)]
        if invalid:
            raise ValueError(f"Correlation columns must be numeric: {', '.join(map(str, invalid))}.")
        if len(numeric_columns(dataframe)) > 2 and any(not _question_mentions(question, column) for column in columns):
            raise ValueError("Please specify at least two numeric column names for correlation.")
        method = _one_of(request.get("method", "pearson"), {"pearson", "spearman"}, "correlation method")
        table = correlation_matrix(dataframe[columns], method)
        return AnalysisResult(
            f"{method.capitalize()} correlation",
            f"Calculated {method} correlations for {len(columns)} numeric columns.",
            request,
            table=table,
            warnings=["Some correlations are unavailable because columns are constant or missing."] if table.isna().any().any() else [],
        )
    if intent == "active_users":
        user_column = _require_user(dataframe, request["user_column"])
        datetime_column = _require_datetime(dataframe, request["datetime_column"])
        _check_user_ambiguity(dataframe, user_column, question)
        _check_datetime_ambiguity(dataframe, datetime_column, question)
        metric = _one_of(str(request["metric"]).lower(), {"dau", "wau", "mau"}, "active-user metric")
        period = {"dau": "day", "wau": "week", "mau": "month"}[metric]
        table = active_users(dataframe, user_column, datetime_column, period)
        return AnalysisResult(
            metric.upper(),
            f"Calculated {metric.upper()} for {len(table):,} {period} periods.",
            request,
            table=table,
            chart_type="line",
            chart_data=table.set_index("period")[["active_users"]] if not table.empty else None,
            warnings=["No rows have both a user ID and a valid timestamp."] if table.empty else [],
        )
    if intent == "revenue_metrics":
        user_column = _require_user(dataframe, request["user_column"])
        datetime_column = _require_datetime(dataframe, request["datetime_column"])
        revenue_column = _require_numeric(dataframe, request["revenue_column"])
        _check_user_ambiguity(dataframe, user_column, question)
        _check_datetime_ambiguity(dataframe, datetime_column, question)
        _check_metric_ambiguity(dataframe, revenue_column, question, metric_context=True)
        event_column, event_value = _optional_event_filter(dataframe, request)
        metrics = revenue_metrics(
            dataframe,
            user_column,
            datetime_column,
            revenue_column,
            event_column,
            event_value,
        )
        metric = request.get("metric", "total_revenue")
        metric = _one_of(metric, {"total_revenue", "arpu", "arppu"}, "revenue metric")
        value = metrics[metric]
        summary = f"{metric.replace('_', ' ').capitalize()} is unavailable because its denominator has no eligible users." if value is None else f"{metric.replace('_', ' ').capitalize()} is {value:,.2f}."
        return AnalysisResult(
            metric.replace("_", " ").title(),
            summary,
            request,
            metrics=metrics,
            warnings=["Revenue rows with missing values or invalid timestamps are excluded from revenue totals."]
            if metrics["total_revenue"] == 0
            else [],
        )
    if intent == "conversion":
        user_column = _require_user(dataframe, request["user_column"])
        datetime_column = _require_datetime(dataframe, request["datetime_column"])
        event_column = _require_event_column(dataframe, request["event_column"])
        _check_user_ambiguity(dataframe, user_column, question)
        _check_datetime_ambiguity(dataframe, datetime_column, question)
        _check_event_column_ambiguity(dataframe, event_column, question)
        starting_event = _resolve_event_value(dataframe, event_column, request["starting_event"])
        conversion_event = _resolve_event_value(dataframe, event_column, request["conversion_event"])
        metrics = conversion_rate(
            dataframe, user_column, datetime_column, event_column, starting_event, conversion_event
        )
        rate = metrics["conversion_rate"]
        summary = (
            "No users performed the starting event, so conversion rate is unavailable."
            if rate is None
            else f"{metrics['converted_users']:,} of {metrics['starting_users']:,} users converted ({rate:.1%})."
        )
        return AnalysisResult("Conversion", summary, request, metrics=metrics)
    if intent == "funnel":
        user_column = _require_user(dataframe, request["user_column"])
        datetime_column = _require_datetime(dataframe, request["datetime_column"])
        event_column = _require_event_column(dataframe, request["event_column"])
        _check_user_ambiguity(dataframe, user_column, question)
        _check_datetime_ambiguity(dataframe, datetime_column, question)
        _check_event_column_ambiguity(dataframe, event_column, question)
        steps = request["steps"]
        if not isinstance(steps, list) or not 2 <= len(steps) <= 5:
            raise ValueError("A funnel must contain between 2 and 5 event steps.")
        resolved_steps = [_resolve_event_value(dataframe, event_column, step) for step in steps]
        table = funnel_analysis(dataframe, user_column, datetime_column, event_column, resolved_steps)
        reached = table["users"].tolist()
        return AnalysisResult(
            "Event funnel",
            "Users reaching each ordered step: " + " → ".join(str(count) for count in reached) + ".",
            request,
            table=table,
            chart_type="bar",
            chart_data=table.set_index("event")[["users"]] if not table.empty else None,
            warnings=["No users reached the first funnel step."] if not table.empty and reached[0] == 0 else [],
        )
    if intent == "retention":
        user_column = _require_user(dataframe, request["user_column"])
        datetime_column = _require_datetime(dataframe, request["datetime_column"])
        _check_user_ambiguity(dataframe, user_column, question)
        _check_datetime_ambiguity(dataframe, datetime_column, question)
        metric = request.get("metric")
        if metric is not None:
            metric = _one_of(str(metric).lower(), {"d1", "d7", "d30"}, "retention metric").upper()
        table = retention_analysis(dataframe, user_column, datetime_column)
        if metric:
            table = table[["cohort_date", "cohort_size", metric]]
            summary = f"Calculated {metric} retention by first-activity cohort. Unobservable recent cohorts remain unavailable."
        else:
            summary = "Calculated exact-day D1, D7, and D30 retention by first-activity cohort."
        if table.empty:
            summary = "No users have valid activity timestamps for retention analysis."
        return AnalysisResult("Cohort retention", summary, request, table=table)
    return _error_result("This question is not supported by the current EventLens analytics tools.", request)


def _optional_event_filter(dataframe: pd.DataFrame, request: dict[str, Any]) -> tuple[str | None, object | None]:
    event_column = request.get("event_column")
    event_value = request.get("purchase_event")
    if (event_column is None) != (event_value is None):
        raise ValueError("Choose both an event column and event value, or omit both to include all rows.")
    if event_column is None:
        return None, None
    event_column = _require_event_column(dataframe, event_column)
    return event_column, _resolve_event_value(dataframe, event_column, event_value)


def _require_numeric(dataframe: pd.DataFrame, column: object) -> str:
    if not isinstance(column, str) or column not in dataframe.columns:
        raise ValueError(f"Numeric column {column!r} was not found.")
    if column not in numeric_columns(dataframe):
        raise ValueError(f"Column {column!r} is not numeric.")
    return column


def _require_categorical(dataframe: pd.DataFrame, column: object) -> str:
    if not isinstance(column, str) or column not in dataframe.columns:
        raise ValueError(f"Categorical column {column!r} was not found.")
    if column not in categorical_columns(dataframe):
        raise ValueError(f"Column {column!r} is not categorical or text data.")
    return column


def _require_datetime(dataframe: pd.DataFrame, column: object) -> str:
    if not isinstance(column, str) or column not in dataframe.columns:
        raise ValueError(f"Datetime column {column!r} was not found.")
    if column not in datetime_compatible_columns(dataframe):
        raise ValueError(f"Column {column!r} is not datetime-compatible.")
    return column


def _require_user(dataframe: pd.DataFrame, column: object) -> str:
    if not isinstance(column, str) or column not in dataframe.columns:
        raise ValueError(f"User identifier column {column!r} was not found.")
    if column in datetime_compatible_columns(dataframe):
        raise ValueError(f"Column {column!r} cannot be used as a user identifier because it is datetime-like.")
    return column


def _require_event_column(dataframe: pd.DataFrame, column: object) -> str:
    return _require_categorical(dataframe, column)


def _resolve_event_value(dataframe: pd.DataFrame, column: str, value: object) -> object:
    actual_values = dataframe[column].dropna().unique().tolist()
    matches = [actual for actual in actual_values if actual == value or str(actual).casefold() == str(value).casefold()]
    if not matches:
        raise ValueError(f"Event value {value!r} was not found in column {column!r}.")
    if len(matches) > 1:
        raise ValueError(f"Event value {value!r} is ambiguous in column {column!r}.")
    return matches[0]


def _one_of(value: object, choices: set[str], label: str) -> str:
    if not isinstance(value, str) or value.lower() not in choices:
        allowed = ", ".join(sorted(choices))
        raise ValueError(f"Unsupported {label}. Choose one of: {allowed}.")
    return value.lower()


def _check_metric_ambiguity(
    dataframe: pd.DataFrame,
    selected_column: str,
    question: str,
    metric_context: bool = False,
) -> None:
    if _looks_like_identifier(selected_column) and not _question_mentions(question, selected_column):
        raise ValueError("The selected numeric column looks like an identifier. Please specify a numeric measure.")
    candidates = [column for column in numeric_columns(dataframe) if not _looks_like_identifier(column)]
    if len(candidates) <= 1 or _question_mentions(question, selected_column):
        return
    if metric_context:
        strong_candidates = [column for column in candidates if _looks_like_revenue_name(column)]
        if len(strong_candidates) == 1 and selected_column == strong_candidates[0]:
            return
    elif re.search(r"revenue|sales", question, re.IGNORECASE):
        strong_candidates = [column for column in candidates if _looks_like_revenue_name(column)]
        if len(strong_candidates) == 1 and selected_column == strong_candidates[0]:
            return
    raise ValueError("I found multiple numeric columns. Please specify which one to analyze.")


def _check_column_ambiguity(selected: str, candidates: list[str], question: str, label: str) -> None:
    if len(candidates) > 1 and not _question_mentions(question, selected):
        raise ValueError(f"I found multiple possible {label}s. Please specify one by name.")


def _check_user_ambiguity(dataframe: pd.DataFrame, selected: str, question: str) -> None:
    candidates = [column for column in dataframe.columns if _looks_like_user_identifier(str(column))]
    if candidates and selected not in candidates and not _question_mentions(question, selected):
        raise ValueError("Please specify which column contains the user identifier.")
    _check_column_ambiguity(selected, [str(column) for column in candidates], question, "user identifier")


def _check_datetime_ambiguity(dataframe: pd.DataFrame, selected: str, question: str) -> None:
    _check_column_ambiguity(
        selected, [str(column) for column in datetime_compatible_columns(dataframe)], question, "datetime column"
    )


def _check_event_column_ambiguity(dataframe: pd.DataFrame, selected: str, question: str) -> None:
    candidates = [
        column
        for column in categorical_columns(dataframe)
        if re.search(r"event|action|activity|type", str(column), re.IGNORECASE)
    ]
    if not candidates:
        candidates = categorical_columns(dataframe)
    _check_column_ambiguity(selected, [str(column) for column in candidates], question, "event column")


def _question_mentions(question: str, column: str) -> bool:
    question_words = _singularized_words(question)
    column_words = _singularized_words(column)
    while column_words and column_words[-1] in {"id", "identifier", "code", "sku"}:
        column_words.pop()
    if not column_words:
        return False
    column_phrase = " ".join(column_words)
    question_phrase = " ".join(question_words)
    return f" {column_phrase} " in f" {question_phrase} "


def _singularized_words(value: str) -> list[str]:
    words = re.sub(r"[^a-z0-9]+", " ", value.casefold()).split()
    result: list[str] = []
    for word in words:
        if len(word) > 4 and word.endswith("ies"):
            word = word[:-3] + "y"
        elif len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
            word = word[:-1]
        result.append(word)
    return result


def _looks_like_identifier(column: str) -> bool:
    return bool(
        re.search(r"(^|[_\W])(id|identifier|code|sku|user|customer|account|member)([_\W]|$)", column, re.IGNORECASE)
    )


def _looks_like_user_identifier(column: str) -> bool:
    return bool(re.search(r"user|customer|account|member|(^|[_\W])id([_\W]|$)", column, re.IGNORECASE))


def _looks_like_revenue_name(column: str) -> bool:
    return bool(re.search(r"revenue|sales|amount|price|spend|value", column, re.IGNORECASE))


def _error_result(message: str, request: dict[str, Any] | None = None) -> AnalysisResult:
    return AnalysisResult("Could not analyze this question", message, request=request, error=message)
