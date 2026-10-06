import pandas as pd
import pytest

from eventlens.analytics import (
    categorical_distribution,
    correlation_matrix,
    datetime_compatible_columns,
    grouped_aggregation,
    numeric_distribution,
    numeric_histogram,
    time_trend,
)


@pytest.fixture
def events():
    return pd.DataFrame(
        {
            "country": ["US", "CA", "US", None, "CA", "US"],
            "amount": [10.0, 20.0, 30.0, 40.0, None, 60.0],
            "timestamp": [
                "2026-01-01T08:00:00Z",
                "2026-01-01T09:00:00Z",
                "2026-01-02T10:00:00Z",
                "2026-02-03T11:00:00Z",
                "not a date",
                "2026-02-28T12:00:00Z",
            ],
            "trend": [1.0, 4.0, 9.0, 16.0, 25.0, 36.0],
            "user_id": ["1001", "1002", "1003", "1004", "1005", "1006"],
        }
    )


def test_categorical_distribution_includes_missing_and_percentages(events):
    result = categorical_distribution(events, "country", top_n=10).set_index("value")
    assert result.loc["US", "count"] == 3
    assert result.loc["US", "percentage"] == 50
    assert result.loc["(Missing)", "count"] == 1
    assert result["percentage"].sum() == pytest.approx(100)


def test_categorical_distribution_respects_top_n(events):
    result = categorical_distribution(events, "country", top_n=2)
    assert len(result) == 2


def test_numeric_summary_and_histogram(events):
    result = numeric_distribution(events, "amount")
    assert result["count"] == 5
    assert result["mean"] == 32
    assert result["median"] == 30
    assert result["std"] == pytest.approx(pd.Series([10, 20, 30, 40, 60]).std())
    assert result["min"] == 10
    assert result["max"] == 60
    histogram = numeric_histogram(events, "amount", bins=4)
    assert histogram["count"].sum() == 5


def test_numeric_summary_handles_all_missing_values():
    dataframe = pd.DataFrame({"amount": pd.Series([None, None], dtype="float64")})
    result = numeric_distribution(dataframe, "amount")
    assert result == {"count": 0, "mean": None, "median": None, "std": None, "min": None, "max": None}
    assert numeric_histogram(dataframe, "amount").empty


@pytest.mark.parametrize(
    ("method", "expected_us_value"),
    [("count", 3), ("sum", 100), ("mean", 100 / 3), ("median", 30), ("min", 10), ("max", 60)],
)
def test_grouped_aggregation_methods(events, method, expected_us_value):
    result = grouped_aggregation(events, "country", "amount", method).set_index("group")
    assert result.loc["US", "result"] == pytest.approx(expected_us_value)
    assert "(Missing)" in result.index


def test_grouped_aggregation_sorts_and_limits_results(events):
    result = grouped_aggregation(events, "country", "amount", "sum", top_n=1)
    assert result.iloc[0]["group"] == "US"
    assert len(result) == 1


def test_daily_time_trend_and_invalid_dates(events):
    original_dates = events["timestamp"].copy()
    result = time_trend(events, "timestamp", "amount", "sum", "day")
    assert result.to_dict(orient="records") == [
        {"period": pd.Timestamp("2026-01-01"), "result": 30.0},
        {"period": pd.Timestamp("2026-01-02"), "result": 30.0},
        {"period": pd.Timestamp("2026-02-03"), "result": 40.0},
        {"period": pd.Timestamp("2026-02-28"), "result": 60.0},
    ]
    pd.testing.assert_series_equal(events["timestamp"], original_dates)


def test_monthly_time_trend(events):
    result = time_trend(events, "timestamp", "amount", "mean", "month")
    assert result["period"].tolist() == [pd.Timestamp("2026-01-01"), pd.Timestamp("2026-02-01")]
    assert result["result"].tolist() == pytest.approx([20, 50])


def test_weekly_time_trend(events):
    result = time_trend(events, "timestamp", "amount", "count", "week")
    assert result["period"].tolist() == [pd.Timestamp("2025-12-29"), pd.Timestamp("2026-02-02"), pd.Timestamp("2026-02-23")]
    assert result["result"].tolist() == [3, 1, 1]


def test_datetime_detection_avoids_identifier_columns(events):
    assert datetime_compatible_columns(events) == ["timestamp"]


def test_pearson_correlation(events):
    result = correlation_matrix(events, "pearson")
    assert result.loc["amount", "amount"] == pytest.approx(1)
    assert 0.9 < result.loc["amount", "trend"] < 1


def test_spearman_correlation(events):
    result = correlation_matrix(events, "spearman")
    assert result.loc["amount", "trend"] == pytest.approx(1)
    pearson = correlation_matrix(events, "pearson")
    assert result.loc["amount", "trend"] != pytest.approx(pearson.loc["amount", "trend"])


def test_correlation_handles_constant_and_missing_values():
    result = correlation_matrix(pd.DataFrame({"a": [1, 1, 1], "b": [1, None, 3]}))
    assert result.loc["a"].isna().all()
    assert result.loc["b", "b"] == pytest.approx(1)


def test_invalid_column_combinations_are_rejected(events):
    with pytest.raises(ValueError, match="must be numeric"):
        numeric_distribution(events, "country")
    with pytest.raises(ValueError, match="categorical"):
        categorical_distribution(events, "amount")
