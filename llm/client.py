from __future__ import annotations

from functools import lru_cache
from collections.abc import Iterable
from typing import Any

from dotenv import find_dotenv, load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
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
        MessagesPlaceholder(variable_name="chat_history"),
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


CONVERSATION_TITLE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Create a concise, specific title of 3 to 7 words for this analytics conversation. "
            "Capture its main subject and branch when relevant. Return only the title, without quotes or a period.",
        ),
        ("human", "Conversation:\n{conversation_text}"),
    ]
)


def _format_findings_input(findings: list[Finding]) -> dict[str, str]:
    return {"findings_text": _format_findings(findings)}

management_summary_formatter = RunnableLambda(
    _format_findings_input
).with_config(run_name="format_management_findings", tags=["opspilot", "formatting"])

question_answer_formatter = RunnableLambda(
    lambda inputs: {
        "question": inputs.get("question") if isinstance(inputs, dict) else None,
        "context": context_to_text(inputs.get("retrieved_context", {}) if isinstance(inputs, dict) else {}),
        "chat_history": format_chat_history(inputs.get("chat_history", [])) if isinstance(inputs, dict) else [],
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


@lru_cache(maxsize=1)
def get_conversation_title_chain():
    return (
        CONVERSATION_TITLE_PROMPT
        | get_chat_model()
        | StrOutputParser()
    ).with_config(run_name="conversation_title_chain", tags=["opspilot", "chat", "title"])


def generate_management_summary(findings: list[Finding]) -> str:
    if not findings:
        return "No material KPI anomalies were detected in the current demo data."

    return _invoke_chain(get_management_summary_chain(), findings)


def generate_conversation_title(messages: Iterable[dict[str, str]]) -> str | None:
    visible_messages = [
        f"{message.get('role', 'unknown').title()}: {str(message.get('content', '')).strip()[:600]}"
        for message in messages
        if message.get("role") in {"user", "assistant"} and str(message.get("content", "")).strip()
    ]
    if not any(line.startswith("User:") for line in visible_messages):
        return None
    try:
        return get_conversation_title_chain().invoke(
            {"conversation_text": "\n".join(visible_messages[-10:])}
        )
    except Exception:
        return None


def answer_question(question: str, retrieved_context: dict[str, Any]) -> str:
    return _invoke_chain(
        get_question_answer_chain(),
        {"question": question, "retrieved_context": retrieved_context, "chat_history": []},
    )


def stream_answer_question(
    question: str,
    retrieved_context: dict[str, Any],
    chat_history: Iterable[dict[str, str]] | None = None,
):
    try:
        yield from get_question_answer_chain().stream(
            {
                "question": question,
                "retrieved_context": retrieved_context,
                "chat_history": chat_history or [],
            }
        )
    except Exception as exc:
        yield _llm_error_message(exc)


def format_chat_history(messages: Iterable[dict[str, str]]) -> list[BaseMessage]:
    chat_history: list[BaseMessage] = []
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if not content:
            continue
        if role == "user":
            chat_history.append(HumanMessage(content=content))
        elif role == "assistant":
            chat_history.append(AIMessage(content=content))
    return chat_history


def _format_findings(findings: list[Finding]) -> str:
    return "\n".join(
        f"- {item.severity.upper()} | {item.month} | {item.branch_name} | {item.title} | Evidence: {item.evidence}"
        for item in findings[:18]
    )


def _invoke_chain(chain, inputs: Any) -> str:
    try:
        return chain.invoke(inputs)
    except Exception as exc:
        return _llm_error_message(exc)


def _llm_error_message(exc: Exception) -> str:
    return (
        "LLM response unavailable. The app still retrieved structured context from SQLite, "
        f"but the model call failed with: {exc}"
    )
