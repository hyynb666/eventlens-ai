"""Interpret natural-language questions and route validated requests to tools."""

from __future__ import annotations

import json
import re

import pandas as pd

from eventlens.ai_tools import AnalysisResult, run_analysis_request
from eventlens.llm.base import LLMClient
from eventlens.llm.prompts import SYSTEM_PROMPT, user_prompt_for_question
from eventlens.multi_step_analyst import GuidedAnalysisResult, run_guided_analysis

NO_API_KEY_MESSAGE = (
    "AI Analyst is optional. To enable it, set OPENAI_API_KEY in your terminal and restart EventLens. "
    "CSV upload, EDA, interactive analytics, and product analytics work without an API key."
)


def analyze_question(
    dataframe: pd.DataFrame,
    question: str,
    llm_client: LLMClient | None,
) -> AnalysisResult:
    """Ask an LLM for a structured request, then run only a validated tool."""
    if llm_client is None:
        return AnalysisResult("AI Analyst unavailable", NO_API_KEY_MESSAGE, error=NO_API_KEY_MESSAGE)
    if not question.strip():
        return AnalysisResult("Enter a question", "Enter a question about the uploaded dataset.", error="empty question")
    if _is_out_of_scope_question(question):
        message = "This question is not supported by the current EventLens analytics tools."
        return AnalysisResult("Unsupported question", message, error="unsupported")

    try:
        response_text = llm_client.complete(
            SYSTEM_PROMPT,
            user_prompt_for_question(dataframe, question.strip()),
        )
    except Exception:
        return AnalysisResult(
            "AI Analyst could not connect",
            "The language model request failed. Check your API configuration and connection, then try again.",
            error="language model request failed",
        )

    try:
        request = json.loads(response_text)
    except (json.JSONDecodeError, TypeError):
        return AnalysisResult(
            "Question could not be interpreted",
            "The language model returned malformed structured output. Please rephrase the question and try again.",
            error="malformed structured output",
        )
    return run_analysis_request(dataframe, request, question=question)


def analyze_user_question(
    dataframe: pd.DataFrame,
    question: str,
    llm_client: LLMClient | None,
) -> AnalysisResult | GuidedAnalysisResult:
    """Select the bounded planner for diagnosis questions; keep direct questions one-step."""
    if is_multi_step_question(question):
        return run_guided_analysis(dataframe, question, llm_client)
    return analyze_question(dataframe, question, llm_client)


def is_multi_step_question(question: str) -> bool:
    return bool(
        re.search(
            r"\b(why|diagnos\w*|what changed|change recently|declin\w*|drop(?:ped)?|decreas\w*|lower for recent|explain the decline|which segments explain|which .*segments|which .*contribute|which .*highest)\b",
            question,
            re.IGNORECASE,
        )
    )


def _is_out_of_scope_question(question: str) -> bool:
    """Reject common prediction and causal questions before calling a model."""
    return bool(
        re.search(
            r"\b(forecast|predict|prediction|project|projection|root cause|causality|caused by)\b|\bnext year\b|\bin the future\b",
            question,
            re.IGNORECASE,
        )
    )
