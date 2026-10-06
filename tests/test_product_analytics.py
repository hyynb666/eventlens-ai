import pandas as pd
import pytest

from eventlens.product_analytics import (
    active_users,
    conversion_rate,
    funnel_analysis,
    revenue_metrics,
    revenue_over_time,
    retention_analysis,
)


@pytest.fixture
def events():
    return pd.DataFrame(
        [
            ("u1", "2026-01-01T09:00:00Z", "view", None),
            ("u1", "2026-01-01T10:00:00Z", "add_to_cart", None),
            ("u1", "2026-01-02T11:00:00Z", "purchase", 100.0),
            ("u1", "2026-01-02T12:00:00Z", "view", None),
            ("u1", "2026-01-08T09:00:00Z", "view", None),
            ("u1", "2026-01-31T09:00:00Z", "view", None),
            ("u2", "2026-01-01T09:00:00Z", "view", None),
            ("u2", "2026-01-01T09:05:00Z", "add_to_cart", None),
            ("u2", "2026-01-01T09:10:00Z", "purchase", 50.0),
            ("u3", "2026-01-01T09:00:00Z", "purchase", 20.0),
            ("u3", "2026-01-02T09:00:00Z", "view", None),
            ("u4", "2026-02-20T09:00:00Z", "session_start", None),
            (None, "2026-01-01T12:00:00Z", "purchase", 10.0),
            ("u5", "not a date", "purchase", 1000.0),
            ("u1", "2026-01-01T09:00:00Z", "view", None),  # duplicate event
        ],
        columns=["user_id", "timestamp", "event_type", "amount"],
    )


def test_dau_counts_distinct_users_and_skips_invalid_rows(events):
    result = active_users(events, "user_id", "timestamp", "day").set_index("period")
    assert result.loc[pd.Timestamp("2026-01-01"), "active_users"] == 3
    assert result.loc[pd.Timestamp("2026-01-02"), "active_users"] == 2
    assert result.loc[pd.Timestamp("2026-02-20"), "active_users"] == 1


def test_wau_counts_users_once_per_monday_week(events):
    result = active_users(events, "user_id", "timestamp", "week").set_index("period")
    assert result.loc[pd.Timestamp("2025-12-29"), "active_users"] == 3
    assert result.loc[pd.Timestamp("2026-01-05"), "active_users"] == 1
    assert result.loc[pd.Timestamp("2026-02-16"), "active_users"] == 1


def test_mau_counts_distinct_users_per_calendar_month(events):
    result = active_users(events, "user_id", "timestamp", "month").set_index("period")
    assert result.loc[pd.Timestamp("2026-01-01"), "active_users"] == 3
    assert result.loc[pd.Timestamp("2026-02-01"), "active_users"] == 1


def test_revenue_arpu_and_arppu(events):
    result = revenue_metrics(events, "user_id", "timestamp", "amount", "event_type", "purchase")
    assert result["total_revenue"] == 180
    assert result["active_users"] == 4
    assert result["paying_users"] == 3
    assert result["arpu"] == 45
    assert result["arppu"] == 60


def test_revenue_event_filter_and_over_time(events):
    result = revenue_over_time(events, "timestamp", "amount", "month", "event_type", "purchase")
    assert result.to_dict(orient="records") == [
        {"period": pd.Timestamp("2026-01-01"), "revenue": 180.0},
    ]


def test_conversion_requires_later_event_for_same_user(events):
    result = conversion_rate(events, "user_id", "timestamp", "event_type", "view", "purchase")
    assert result["starting_users"] == 3
    assert result["converted_users"] == 2
    assert result["conversion_rate"] == pytest.approx(2 / 3)


def test_conversion_before_start_does_not_convert(events):
    # u3's purchase is before their first view; only u1 and u2 convert.
    result = conversion_rate(events, "user_id", "timestamp", "event_type", "view", "purchase")
    assert result["converted_users"] == 2


def test_two_step_funnel_counts_users_not_events(events):
    result = funnel_analysis(events, "user_id", "timestamp", "event_type", ["view", "purchase"])
    assert result["users"].tolist() == [3, 2]
    assert result.loc[1, "conversion_from_previous_pct"] == pytest.approx(200 / 3)
    assert result.loc[1, "conversion_from_first_pct"] == pytest.approx(200 / 3)


def test_multi_step_funnel_requires_every_step_in_order(events):
    result = funnel_analysis(
        events,
        "user_id",
        "timestamp",
        "event_type",
        ["view", "add_to_cart", "purchase"],
    )
    assert result["users"].tolist() == [3, 2, 2]


def test_funnel_does_not_count_out_of_order_events(events):
    result = funnel_analysis(events, "user_id", "timestamp", "event_type", ["view", "purchase"])
    assert result["users"].tolist() == [3, 2]


def test_funnel_rejects_too_few_or_too_many_steps(events):
    with pytest.raises(ValueError, match="between 2 and 5"):
        funnel_analysis(events, "user_id", "timestamp", "event_type", ["view"])
    with pytest.raises(ValueError, match="between 2 and 5"):
        funnel_analysis(events, "user_id", "timestamp", "event_type", list("abcdef"))


def test_d1_retention_uses_exact_next_calendar_day(events):
    result = retention_analysis(events, "user_id", "timestamp").set_index("cohort_date")
    jan_1 = result.loc[pd.Timestamp("2026-01-01")]
    assert jan_1["cohort_size"] == 3
    assert jan_1["D1"] == pytest.approx(200 / 3)


def test_d7_retention_uses_exact_seventh_day(events):
    result = retention_analysis(events, "user_id", "timestamp").set_index("cohort_date")
    assert result.loc[pd.Timestamp("2026-01-01"), "D7"] == pytest.approx(100 / 3)


def test_d30_retention_uses_exact_thirtieth_day(events):
    result = retention_analysis(events, "user_id", "timestamp").set_index("cohort_date")
    assert result.loc[pd.Timestamp("2026-01-01"), "D30"] == pytest.approx(100 / 3)


def test_recent_cohort_retention_is_unavailable_not_zero(events):
    result = retention_analysis(events, "user_id", "timestamp").set_index("cohort_date")
    recent = result.loc[pd.Timestamp("2026-02-20")]
    assert pd.isna(recent["D1"])
    assert pd.isna(recent["D7"])
    assert pd.isna(recent["D30"])


def test_invalid_dates_and_missing_users_are_excluded_from_user_metrics(events):
    result = active_users(events, "user_id", "timestamp", "day")
    assert result["active_users"].sum() == 8
    retention = retention_analysis(events, "user_id", "timestamp")
    assert retention["cohort_size"].sum() == 4


def test_no_valid_activity_returns_empty_tables(events):
    invalid = events.assign(timestamp="bad date")
    assert active_users(invalid, "user_id", "timestamp").empty
    assert retention_analysis(invalid, "user_id", "timestamp").empty


def test_missing_required_columns_raise_friendly_error(events):
    with pytest.raises(ValueError, match="not found"):
        active_users(events, "missing_user", "timestamp")


def test_revenue_without_payers_has_unavailable_arppu():
    dataframe = pd.DataFrame({"user": ["a", "b"], "value": [0.0, None]})
    dataframe["timestamp"] = ["2026-01-01", "2026-01-02"]
    result = revenue_metrics(dataframe, "user", "timestamp", "value")
    assert result["total_revenue"] == 0
    assert result["arpu"] == 0
    assert result["arppu"] is None
