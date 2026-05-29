from __future__ import annotations

from functools import lru_cache
from typing import Any

from dotenv import find_dotenv, load_dotenv
from langchain.chat_models import init_chat_model

from analysis.kpi_analysis import Finding
from analysis.retrieval import context_to_text


MODEL_NAME = "gpt-4o-mini"


@lru_cache(maxsize=1)
def get_chat_model():
    load_dotenv(find_dotenv())
    return init_chat_model(model=MODEL_NAME)


def generate_management_summary(findings: list[Finding]) -> str:
    if not findings:
        return "No material KPI anomalies were detected in the current demo data."

    findings_text = "\n".join(
        f"- {item.severity.upper()} | {item.month} | {item.branch_name} | {item.title} | Evidence: {item.evidence}"
        for item in findings[:18]
    )
    prompt = f"""
You are OpsPilot AI, an operational analytics assistant for logistics branch management.

Write a concise executive management summary based ONLY on the structured findings below.
Prioritize critical developments, mention affected branches, and explain likely operational implications.
Do not invent numbers, branches, or causes. If a cause is uncertain, say it is a likely indicator.

Structured findings:
{findings_text}
"""
    return _invoke(prompt)


def answer_question(question: str, retrieved_context: dict[str, Any]) -> str:
    context = context_to_text(retrieved_context)
    prompt = f"""
You are OpsPilot AI, a grounded conversational analytics assistant.

Answer the user's question using ONLY the retrieved SQLite records and KPI findings below.
Rules:
- Do not hallucinate numbers.
- Every factual claim must be supported by the provided context.
- If the context is insufficient, say what is missing and suggest the exact KPI/detail area to inspect.
- Keep the answer professional and concise.
- For root-cause questions, separate observed evidence from likely interpretation.
- For customer satisfaction questions, analyze the retrieved customer_feedback comments, categories, ratings, and sentiment counts.
- When using feedback comments, describe repeated themes; do not claim a theme is dominant unless the summary counts support it.

User question:
{question}

Retrieved context:
{context}
"""
    return _invoke(prompt)


def _invoke(prompt: str) -> str:
    try:
        response = get_chat_model().invoke(prompt)
        return response.content
    except Exception as exc:
        return (
            "LLM response unavailable. The app still retrieved structured context from SQLite, "
            f"but the model call failed with: {exc}"
        )
