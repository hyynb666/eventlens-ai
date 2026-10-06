import pandas as pd

from eventlens.profiling import categorical_summary, missing_summary, numeric_summary, profile_columns, profile_dataset


def sample_dataframe():
    return pd.DataFrame(
        {
            "amount": [10.0, 20.0, 20.0, None],
            "country": ["US", "CA", "US", None],
        }
    )


def test_dataset_dimensions_and_duplicate_count():
    dataframe = pd.DataFrame({"x": [1, 1, 2], "y": ["a", "a", "b"]})
    result = profile_dataset(dataframe)
    assert result["row_count"] == 3
    assert result["column_count"] == 2
    assert result["duplicate_row_count"] == 1
    assert result["memory_usage_bytes"] > 0


def test_numeric_summary_values():
    summary = numeric_summary(sample_dataframe())
    assert list(summary.columns) == ["count", "mean", "std", "min", "median", "max"]
    assert summary.loc["amount", "count"] == 3
    assert summary.loc["amount", "mean"] == 50 / 3
    assert summary.loc["amount", "median"] == 20


def test_categorical_summary_values():
    summary = categorical_summary(sample_dataframe()).set_index("column")
    assert summary.loc["country", "unique_count"] == 2
    assert summary.loc["country", "most_common_value"] == "US"
    assert summary.loc["country", "most_common_frequency"] == 2


def test_missing_summary_counts_and_percentages():
    summary = missing_summary(sample_dataframe())
    amount = summary.set_index("column").loc["amount"]
    assert amount["missing_count"] == 1
    assert amount["missing_percentage"] == 25
    assert list(summary["column"]) == ["amount", "country"]


def test_column_profile_includes_dtype_and_unique_count():
    summary = profile_columns(sample_dataframe()).set_index("column")
    assert summary.loc["amount", "dtype"] == "float64"
    assert summary.loc["country", "missing_count"] == 1
    assert summary.loc["country", "unique_count"] == 2
