"""Streamlit interface for the controlled natural-language analyst."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from eventlens.ai_router import NO_API_KEY_MESSAGE, analyze_user_question
from eventlens.ai_tools import AnalysisResult
from eventlens.llm.provider import client_from_environment
from eventlens.multi_step_analyst import GuidedAnalysisResult


def render_ai_analyst(dataframe: pd.DataFrame, dataset_key: str) -> None:
    """Show the AI Analyst form and render its local, verified analysis result."""
    st.header("AI Analyst")
    client = client_from_environment()
    result_key = "eventlens_ai_analyst_result"
    if client is None:
        st.info(NO_API_KEY_MESSAGE)
        st.session_state.pop(result_key, None)
        return
    stored_result = st.session_state.get(result_key)
    if stored_result is not None and stored_result[0] != dataset_key:
        st.session_state.pop(result_key, None)

    with st.form("eventlens_ai_analyst_form"):
        question = st.text_input("Ask a question about your data")
        submitted = st.form_submit_button("Analyze")
    if submitted:
        st.session_state[result_key] = (dataset_key, analyze_user_question(dataframe, question, client))

    stored_result = st.session_state.get(result_key)
    result = stored_result[1] if stored_result is not None else None
    if isinstance(result, AnalysisResult):
        _render_result(result)
    elif isinstance(result, GuidedAnalysisResult):
        _render_guided_result(result)


def _render_guided_result(result: GuidedAnalysisResult) -> None:
    st.subheader("Guided analysis")
    st.caption(f"Question: {result.question}")
    if result.error and not result.plan:
        st.info(result.error)
        return
    st.write(f"**Goal:** {result.goal}")
    st.markdown("**Analysis plan**")
    for step in result.plan:
        st.markdown(f"{step.step_id}. **{step.intent.replace('_', ' ').title()}** — {step.reason}")

    st.markdown("**Evidence**")
    for item in result.evidence:
        with st.expander(f"Step {item.step_id}: {item.title}", expanded=item.step_id == 1):
            if item.error:
                st.warning(item.error)
            else:
                st.write(item.summary)
            for warning in item.warnings:
                st.warning(warning)
            if item.metrics:
                metric_items = list(item.metrics.items())
                cols = st.columns(min(4, len(metric_items)))
                for index, (name, value) in enumerate(metric_items):
                    cols[index % len(cols)].metric(name.replace("_", " ").title(), _format_metric(value))
            if item.table is not None:
                st.dataframe(item.table, width="stretch")

    st.markdown("**Final answer**")
    st.write(result.final_answer)


def _render_result(result: AnalysisResult) -> None:
    st.subheader(result.title)
    if result.error:
        st.info(result.summary)
    else:
        st.write(result.summary)

    for warning in result.warnings:
        st.warning(warning)

    if result.metrics:
        metric_items = list(result.metrics.items())
        metric_columns = st.columns(min(4, len(metric_items)))
        for index, (name, value) in enumerate(metric_items):
            metric_columns[index % len(metric_columns)].metric(
                name.replace("_", " ").title(), _format_metric(value)
            )

    if result.table is not None:
        st.dataframe(result.table, width="stretch")
    if result.chart_data is not None:
        if result.chart_type == "bar":
            st.bar_chart(result.chart_data)
        elif result.chart_type == "line":
            st.line_chart(result.chart_data)

    if result.request is not None:
        with st.expander("Analysis plan"):
            st.json(result.request)


def _format_metric(value: int | float | str | None) -> str:
    if value is None:
        return "Unavailable"
    if isinstance(value, float):
        return f"{value:,.2f}"
    if isinstance(value, int):
        return f"{value:,}"
    return value
