from __future__ import annotations

from functools import lru_cache
from typing import Any

from dotenv import find_dotenv, load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from analysis.kpi_analysis import Finding
from analysis.retrieval import context_to_text


MODEL_NAME = "gpt-4o-mini"


MANAGEMENT_SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are OpsPilot AI, an operational analytics assistant for logistics branch management.",
        ),
        (
            "human",
            """Write a concise executive management summary based ONLY on the structured findings below.
Prioritize critical developments, mention affected branches, and explain likely operational implications.
Do not invent numbers, branches, or causes. If a cause is uncertain, say it is a likely indicator.

Structured findings:
{findings_text}""",
        ),
    ]
)


QUESTION_ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are OpsPilot AI, a grounded conversational analytics assistant.",
        ),
        (
            "human",
            """Answer the user's question using ONLY the retrieved SQLite records and KPI findings below.
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
{context}""",
        ),
    ]
)


management_summary_formatter = RunnableLambda(
    lambda findings: {"findings_text": _format_findings(findings)}
).with_config(run_name="format_management_findings", tags=["opspilot", "formatting"])

question_answer_formatter = RunnableLambda(
    lambda inputs: {
        "question": inputs["question"],
        "context": context_to_text(inputs["retrieved_context"]),
    }
).with_config(run_name="format_retrieved_context", tags=["opspilot", "formatting"])


@lru_cache(maxsize=1)
def get_chat_model():
    load_dotenv(find_dotenv())
    return init_chat_model(model=MODEL_NAME)


@lru_cache(maxsize=1)
def get_management_summary_chain():
    return (
        management_summary_formatter
        | MANAGEMENT_SUMMARY_PROMPT
        | get_chat_model()
        | StrOutputParser()
    ).with_config(run_name="management_summary_chain", tags=["opspilot", "summary"])


@lru_cache(maxsize=1)
def get_question_answer_chain():
    return (
        question_answer_formatter
        | QUESTION_ANSWER_PROMPT
        | get_chat_model()
        | StrOutputParser()
    ).with_config(run_name="question_answer_chain", tags=["opspilot", "rag", "chat"])


def generate_management_summary(findings: list[Finding]) -> str:
    if not findings:
        return "No material KPI anomalies were detected in the current demo data."

    return _invoke_chain(get_management_summary_chain(), findings)


def answer_question(question: str, retrieved_context: dict[str, Any]) -> str:
    return _invoke_chain(
        get_question_answer_chain(),
        {"question": question, "retrieved_context": retrieved_context},
    )


def _format_findings(findings: list[Finding]) -> str:
    return "\n".join(
        f"- {item.severity.upper()} | {item.month} | {item.branch_name} | {item.title} | Evidence: {item.evidence}"
        for item in findings[:18]
    )


def _invoke_chain(chain, inputs: Any) -> str:
    try:
        return chain.invoke(inputs)
    except Exception as exc:
        return (
            "LLM response unavailable. The app still retrieved structured context from SQLite, "
            f"but the model call failed with: {exc}"
        )
