import ast
import json
from pathlib import Path

import pandas as pd

from eventlens.ai_router import NO_API_KEY_MESSAGE, analyze_user_question
from eventlens.ai_tools import AnalysisResult
from eventlens.analytics import compare_latest_periods
from eventlens.multi_step_analyst import (
    HARD_MAX_STEPS,
    TOOL_REGISTRY,
    run_guided_analysis,
    configured_max_steps,
    validate_plan,
)


class FakeLLM:
    def __init__(self, response):
        self.response = response
        self.user_prompt = None

    def complete(self, system_prompt, user_prompt):
        self.user_prompt = user_prompt
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def events():
    return pd.DataFrame(
        [
            ("u1", "2026-01-01", "view", "US", "mobile", 10.0),
            ("u1", "2026-01-02", "purchase", "US", "mobile", 20.0),
            ("u1", "2026-02-01", "view", "US", "desktop", 0.0),
            ("u2", "2026-01-01", "view", "CA", "desktop", 5.0),
            ("u2", "2026-01-03", "purchase", "CA", "desktop", 10.0),
            ("u2", "2026-02-03", "view", "CA", "mobile", 0.0),
            ("u3", "2026-01-05", "view", "US", "mobile", 0.0),
            ("u3", "2026-02-05", "purchase", "US", "mobile", 8.0),
        ],
        columns=["user_id", "timestamp", "event_type", "country", "device", "amount"],
    )


def step(intent, parameters, reason="Inspect this evidence"):
    return {"step_id": 1, "intent": intent, "parameters": parameters, "reason": reason}


def plan(*steps, goal="Investigate the question"):
    return json.dumps({"goal": goal, "steps": list(steps)})


def run(question, response, **kwargs):
    return run_guided_analysis(events(), question, FakeLLM(response), **kwargs)


def trend_step():
    return step(
        "time_trend",
        {"datetime_column": "timestamp", "value_column": "amount", "aggregation": "sum", "granularity": "month"},
    )


def group_step(group="country"):
    return step("grouped_aggregation", {"group_by": group, "value_column": "amount", "aggregation": "sum"})


def test_valid_two_step_plan_executes_and_collects_evidence():
    second = group_step()
    second["step_id"] = 2
    result = run("Why did revenue drop?", plan(trend_step(), second))
    assert len(result.plan) == len(result.evidence) == 2
    assert result.evidence[0].table is not None
    assert result.evidence[0].intent == "time_trend"
    assert "Latest period" in result.final_answer
    assert "do not establish causation" in result.final_answer


def test_valid_five_step_plan_is_accepted():
    steps = [step("dataset_overview", {}, f"Evidence {i}") for i in range(1, 6)]
    for index, item in enumerate(steps, start=1):
        item["step_id"] = index
    result = run("Investigate the data", plan(*steps), max_steps=5)
    assert len(result.evidence) == 5


def test_plan_exceeding_configured_step_count_is_rejected_before_execution(monkeypatch):
    called = []
    monkeypatch.setitem(TOOL_REGISTRY, "dataset_overview", lambda *args: called.append(True))
    steps = [step("dataset_overview", {}) for _ in range(3)]
    for index, item in enumerate(steps, start=1):
        item["step_id"] = index
    result = run("Inspect", plan(*steps), max_steps=2)
    assert result.error and "between 1 and 2" in result.error
    assert called == []


def test_hard_step_limit_never_exceeds_five():
    payload = {"goal": "test", "steps": [step("dataset_overview", {}) for _ in range(6)]}
    error, _, _ = validate_plan(events(), payload, "Inspect", HARD_MAX_STEPS)
    assert error and "between 1 and 5" in error


def test_unsupported_intent_rejected_before_execution():
    result = run("Analyze", plan(step("execute_python", {"code": "print(1)"})))
    assert "unsupported" in result.error
    assert result.evidence == []


def test_invalid_column_in_second_step_rejects_entire_plan(monkeypatch):
    called = []
    monkeypatch.setitem(TOOL_REGISTRY, "dataset_overview", lambda *args: called.append(True))
    later = step("numeric_summary", {"column": "private_column"})
    later["step_id"] = 2
    result = run("Inspect amount", plan(step("dataset_overview", {}), later))
    assert "not found" in result.error
    assert not called


def test_invalid_numeric_type_rejected_before_execution():
    result = run("Inspect country", plan(step("numeric_summary", {"column": "country"})))
    assert "not numeric" in result.error
    assert result.evidence == []


def test_malformed_planner_json_is_friendly():
    result = run("Why did revenue change?", "not json")
    assert result.error and "malformed" in result.error


def test_execution_order_is_sequential(monkeypatch):
    calls = []
    def tool(name):
        def execute(dataframe, request, question):
            calls.append(name)
            return AnalysisResult(name, f"Verified {name}")
        return execute
    monkeypatch.setitem(TOOL_REGISTRY, "dataset_overview", tool("overview"))
    monkeypatch.setitem(TOOL_REGISTRY, "missing_summary", tool("missing"))
    result = run("Inspect", plan(step("dataset_overview", {}), {**step("missing_summary", {}), "step_id": 2}))
    assert calls == ["overview", "missing"]
    assert [item.step_id for item in result.evidence] == [1, 2]


def test_evidence_is_compact_and_contains_only_tool_output():
    result = run("Inspect", plan(trend_step()))
    evidence = result.evidence[0]
    assert evidence.title.startswith("Month")
    assert evidence.table is not None
    assert len(evidence.table) <= 50
    assert "Verified" not in evidence.summary


def test_critical_first_step_failure_stops_workflow(monkeypatch):
    later_called = []
    def fail(*args):
        return AnalysisResult("failed", "no result", error="tool failed")
    monkeypatch.setitem(TOOL_REGISTRY, "dataset_overview", fail)
    monkeypatch.setitem(TOOL_REGISTRY, "missing_summary", lambda *args: later_called.append(True))
    result = run("Inspect", plan(step("dataset_overview", {}), {**step("missing_summary", {}), "step_id": 2}))
    assert result.error == "tool failed"
    assert len(result.evidence) == 1
    assert later_called == []


def test_noncritical_failure_is_recorded_and_later_steps_continue(monkeypatch):
    calls = []
    def fail(*args):
        return AnalysisResult("failed", "no result", error="later tool failed")
    def succeed(*args):
        calls.append("overview")
        return AnalysisResult("overview", "Rows were counted")
    monkeypatch.setitem(TOOL_REGISTRY, "missing_summary", fail)
    monkeypatch.setitem(TOOL_REGISTRY, "dataset_overview", succeed)
    p = plan(step("dataset_overview", {}), {**step("missing_summary", {}), "step_id": 2},
             {**step("dataset_overview", {}), "step_id": 3})
    result = run("Inspect", p)
    assert len(result.evidence) == 3
    assert result.evidence[1].error == "later tool failed"
    assert calls == ["overview", "overview"]
    assert "conclusion is limited" in result.final_answer


def test_synthesis_uses_verified_evidence_and_marks_uncertainty():
    result = run("Why did revenue drop?", plan(trend_step()))
    assert "Evidence 1" in result.final_answer
    assert "Latest period" in result.final_answer
    assert "do not establish causation" in result.final_answer


def test_revenue_decline_diagnosis_flow():
    result = run("Why did revenue drop in the latest month?", plan(trend_step()))
    assert result.evidence[0].intent == "time_trend"
    assert "change" in result.final_answer


def test_segment_comparison_flow():
    result = run("Which user segments have the highest revenue?", plan(group_step("country")))
    assert result.evidence[0].table.iloc[0]["group"] == "US"
    assert "highest result" in result.final_answer


def test_conversion_diagnosis_flow():
    conv = step("conversion", {
        "user_column": "user_id", "datetime_column": "timestamp", "event_column": "event_type",
        "starting_event": "view", "conversion_event": "purchase",
    })
    result = run("Why did conversion decrease?", plan(conv))
    assert result.evidence[0].metrics["starting_users"] == 3
    assert "converted" in result.final_answer


def test_activity_change_flow():
    active = step("active_users", {"metric": "dau", "user_column": "user_id", "datetime_column": "timestamp"})
    result = run("What changed in user activity recently?", plan(active))
    assert result.evidence[0].intent == "active_users"
    assert result.evidence[0].table is not None


def test_retention_investigation_flow():
    retention = step("retention", {"user_column": "user_id", "datetime_column": "timestamp", "metric": "d7"})
    result = run("Why is retention lower for recent cohorts?", plan(retention))
    assert result.evidence[0].table is not None
    assert list(result.evidence[0].table.columns) == ["cohort_date", "cohort_size", "D7"]


def test_single_step_issue_four_compatibility():
    response = json.dumps({"intent": "active_users", "metric": "dau", "user_column": "user_id", "datetime_column": "timestamp"})
    result = analyze_user_question(events(), "What is DAU?", FakeLLM(response))
    assert isinstance(result, AnalysisResult)
    assert result.title == "DAU"


def test_guided_no_api_key_behavior():
    result = run_guided_analysis(events(), "Why did revenue drop?", None)
    assert result.error == NO_API_KEY_MESSAGE


def test_planner_receives_schema_metadata_without_dataset_rows():
    fake = FakeLLM(plan(trend_step()))
    run_guided_analysis(events(), "Why did revenue drop?", fake)
    assert "user_id" in fake.user_prompt
    assert "u1" not in fake.user_prompt
    assert '"US"' not in fake.user_prompt


def test_environment_step_limit_is_bounded(monkeypatch):
    monkeypatch.setenv("EVENTLENS_MAX_ANALYSIS_STEPS", "20")
    assert configured_max_steps() == 5
    monkeypatch.setenv("EVENTLENS_MAX_ANALYSIS_STEPS", "2")
    assert configured_max_steps() == 2
    monkeypatch.setenv("EVENTLENS_MAX_ANALYSIS_STEPS", "bad")
    assert configured_max_steps() == 5


def test_router_selects_planner_for_diagnosis_questions():
    fake = FakeLLM(plan(trend_step()))
    result = analyze_user_question(events(), "Why did revenue drop in the latest month?", fake)
    assert len(result.plan) == 1
    assert "Maximum steps: 5" in fake.user_prompt


def test_registry_is_closed_and_has_no_arbitrary_intent():
    assert "execute_python" not in TOOL_REGISTRY
    assert set(TOOL_REGISTRY) == {"dataset_overview", "missing_summary", "numeric_summary", "categorical_distribution",
        "grouped_aggregation", "time_trend", "correlation", "active_users", "revenue_metrics", "conversion", "funnel", "retention"}


def test_workflow_source_has_no_dynamic_execution():
    for relative in ["eventlens/multi_step_analyst.py", "eventlens/ai_tools.py", "eventlens/ai_router.py"]:
        tree = ast.parse(Path(relative).read_text(encoding="utf-8"))
        calls = [node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Name) and node.func.id in {"eval", "exec"}]
        assert calls == []


def test_period_comparison_helper_reports_change_without_forecast():
    trend = pd.DataFrame({"period": ["2026-01", "2026-02"], "result": [100.0, 80.0]})
    compared = compare_latest_periods(trend)
    assert compared["absolute_change"] == -20
    assert compared["percentage_change"] == -20
