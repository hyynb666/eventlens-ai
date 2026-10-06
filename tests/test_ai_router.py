import ast
import json
from pathlib import Path

import pandas as pd

from eventlens.ai_router import NO_API_KEY_MESSAGE, analyze_question
from eventlens.ai_tools import run_analysis_request
from eventlens.llm.prompts import metadata_for_dataframe, user_prompt_for_question
from eventlens.llm.provider import client_from_environment


class FakeLLM:
    def __init__(self, response):
        self.response = response
        self.system_prompt = None
        self.user_prompt = None

    def complete(self, system_prompt, user_prompt):
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def request(intent, **fields):
    return json.dumps({"intent": intent, **fields})


def events():
    return pd.DataFrame(
        [
            ("u1", "2026-01-01T09:00:00Z", "view", "US", "mobile", None, 1),
            ("u1", "2026-01-01T09:10:00Z", "add_to_cart", "US", "mobile", None, 1),
            ("u1", "2026-01-02T09:00:00Z", "purchase", "US", "mobile", 20.0, 1),
            ("u1", "2026-01-08T09:00:00Z", "view", "US", "desktop", None, 2),
            ("u1", "2026-01-31T09:00:00Z", "view", "US", "desktop", None, 1),
            ("u2", "2026-01-01T10:00:00Z", "view", "CA", "desktop", None, 2),
            ("u2", "2026-01-01T10:10:00Z", "purchase", "CA", "desktop", 10.0, 1),
            ("u2", "2026-01-02T10:00:00Z", "view", "CA", "mobile", None, 2),
            ("u3", "2026-01-01T11:00:00Z", "purchase", "US", "mobile", 5.0, 3),
            ("u3", "2026-01-02T11:00:00Z", "view", "US", "mobile", None, 1),
            ("u4", "2026-02-20T12:00:00Z", "view", "CA", "tablet", None, 2),
        ],
        columns=["user_id", "timestamp", "event_type", "country", "device", "amount", "order_count"],
    )


def test_dataset_overview_intent():
    result = analyze_question(events(), "How many rows and columns?", FakeLLM(request("dataset_overview")))
    assert result.metrics["rows"] == 11
    assert result.metrics["columns"] == 7
    assert result.metrics["duplicate_rows"] == 0


def test_numeric_summary_routes_to_local_calculation():
    result = analyze_question(
        events(), "What is the average amount?", FakeLLM(request("numeric_summary", column="amount"))
    )
    assert result.table.loc[0, "mean"] == 35 / 3
    assert "3 non-missing" in result.summary


def test_categorical_distribution_routing():
    result = analyze_question(
        events(),
        "What are the most common countries?",
        FakeLLM(request("categorical_distribution", column="country", top_n=5)),
    )
    assert result.table.iloc[0]["value"] == "US"
    assert result.chart_type == "bar"


def test_grouped_aggregation_routing():
    result = analyze_question(
        events(),
        "What is average amount by country?",
        FakeLLM(request("grouped_aggregation", group_by="country", value_column="amount", aggregation="mean")),
    )
    assert result.table.iloc[0]["group"] == "US"
    assert result.table.iloc[0]["result"] == 12.5


def test_time_trend_routing():
    result = analyze_question(
        events(),
        "Show monthly revenue trend.",
        FakeLLM(
            request(
                "time_trend",
                datetime_column="timestamp",
                value_column="amount",
                aggregation="sum",
                granularity="month",
            )
        ),
    )
    assert result.chart_type == "line"
    assert result.table.iloc[0]["result"] == 35


def test_correlation_routing():
    result = analyze_question(
        events(),
        "What is the correlation between amount and order_count?",
        FakeLLM(request("correlation", columns=["amount", "order_count"], method="spearman")),
    )
    assert result.table.loc["amount", "amount"] == 1
    assert "Spearman" in result.title


def test_active_user_routing():
    result = analyze_question(
        events(), "What is DAU?", FakeLLM(request("active_users", metric="dau", user_column="user_id", datetime_column="timestamp"))
    )
    assert result.table.iloc[0]["active_users"] == 3
    assert "DAU" in result.title


def test_revenue_metrics_routing():
    result = analyze_question(
        events(),
        "What is total revenue?",
        FakeLLM(
            request(
                "revenue_metrics",
                user_column="user_id",
                datetime_column="timestamp",
                revenue_column="amount",
                metric="total_revenue",
                event_column="event_type",
                purchase_event="purchase",
            )
        ),
    )
    assert result.metrics["total_revenue"] == 35
    assert result.metrics["arpu"] == 8.75


def test_conversion_routing():
    result = analyze_question(
        events(),
        "Show conversion from view to purchase.",
        FakeLLM(
            request(
                "conversion",
                user_column="user_id",
                datetime_column="timestamp",
                event_column="event_type",
                starting_event="view",
                conversion_event="purchase",
            )
        ),
    )
    assert result.metrics["starting_users"] == 4
    assert result.metrics["converted_users"] == 2


def test_funnel_routing():
    result = analyze_question(
        events(),
        "Show the funnel from view to add_to_cart to purchase.",
        FakeLLM(
            request(
                "funnel",
                user_column="user_id",
                datetime_column="timestamp",
                event_column="event_type",
                steps=["view", "add_to_cart", "purchase"],
            )
        ),
    )
    assert result.table["users"].tolist() == [4, 1, 1]
    assert result.chart_type == "bar"


def test_retention_routing():
    result = analyze_question(
        events(),
        "What is D7 retention?",
        FakeLLM(request("retention", user_column="user_id", datetime_column="timestamp", metric="d7")),
    )
    assert list(result.table.columns) == ["cohort_date", "cohort_size", "D7"]
    assert "D7" in result.summary


def test_invalid_column_is_rejected_before_tool_call():
    result = run_analysis_request(events(), {"intent": "numeric_summary", "column": "secret"}, "average amount")
    assert result.error
    assert "not found" in result.summary


def test_invalid_event_value_is_rejected():
    result = run_analysis_request(
        events(),
        {
            "intent": "conversion",
            "user_column": "user_id",
            "datetime_column": "timestamp",
            "event_column": "event_type",
            "starting_event": "not_an_event",
            "conversion_event": "purchase",
        },
        "Convert from not_an_event to purchase",
    )
    assert result.error
    assert "was not found" in result.summary


def test_unsupported_intent_never_runs_a_tool():
    result = run_analysis_request(events(), {"intent": "execute_python", "code": "1 + 1"})
    assert result.error
    assert "not supported" in result.summary


def test_out_of_scope_prediction_is_rejected_before_llm_call():
    fake = FakeLLM(RuntimeError("should not be called"))
    result = analyze_question(events(), "Predict next year's revenue.", fake)
    assert result.error == "unsupported"
    assert fake.user_prompt is None


def test_ambiguous_numeric_column_is_not_guessed():
    result = run_analysis_request(
        events(), {"intent": "numeric_summary", "column": "amount"}, "Show the average."
    )
    assert result.error
    assert "multiple numeric columns" in result.summary


def test_model_can_return_a_clarification():
    result = analyze_question(
        events(),
        "Show revenue trend.",
        FakeLLM(request("clarification", message="Which numeric column should represent revenue?")),
    )
    assert result.error
    assert "Which numeric column" in result.summary


def test_malformed_structured_output_is_friendly():
    result = analyze_question(events(), "What is DAU?", FakeLLM("not valid json"))
    assert result.error == "malformed structured output"
    assert "rephrase" in result.summary


def test_no_api_key_behavior(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert client_from_environment() is None
    result = analyze_question(events(), "What is DAU?", None)
    assert result.summary == NO_API_KEY_MESSAGE


def test_llm_failure_is_friendly():
    result = analyze_question(events(), "What is DAU?", FakeLLM(RuntimeError("network unavailable")))
    assert result.error == "language model request failed"
    assert "try again" in result.summary


def test_dataset_metadata_contains_no_rows():
    dataframe = events()
    metadata = metadata_for_dataframe(dataframe)
    prompt = user_prompt_for_question(dataframe, "What is DAU?")
    assert "user_id" in metadata["columns"][0]["name"]
    assert "u1" not in prompt
    assert "numeric_columns" in prompt


def test_ai_modules_do_not_call_eval_or_exec():
    ai_files = [
        Path("eventlens/ai_tools.py"),
        Path("eventlens/ai_router.py"),
        Path("eventlens/ai_analyst_ui.py"),
        *Path("eventlens/llm").glob("*.py"),
    ]
    for file_path in ai_files:
        tree = ast.parse(file_path.read_text(encoding="utf-8"))
        forbidden_calls = [
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {"eval", "exec"}
        ]
        assert forbidden_calls == [], f"Unexpected dynamic execution call in {file_path}"
