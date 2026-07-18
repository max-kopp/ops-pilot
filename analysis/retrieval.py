from __future__ import annotations

from functools import lru_cache
import re
from typing import Any, Literal

import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, field_validator

from analysis.kpi_analysis import Finding


BRANCH_NAMES = ["Hamburg", "Munich", "Berlin", "Cologne", "Frankfurt", "Stuttgart", "Dortmund", "Leipzig"]
KPI_NAMES = [
    "service_level",
    "transportation_costs",
    "staffing_level",
    "complaints",
    "customer_satisfaction",
]

Intent = Literal[
    "general_kpi_question",
    "root_cause",
    "critical_branches",
    "similar_developments",
    "cost_drivers",
    "customer_satisfaction_drivers",
] | str
ComparisonMode = Literal["latest", "trend", "root_cause", "similar_branches", "drivers", "critical", "general"]
KpiName = Literal[
    "service_level",
    "transportation_costs",
    "staffing_level",
    "complaints",
    "customer_satisfaction",
]


class TimeRange(BaseModel):
    """Optional month range extracted from a user question."""

    start_month: str | None = Field(default=None, description="Inclusive start month in YYYY-MM format.")
    end_month: str | None = Field(default=None, description="Inclusive end month in YYYY-MM format.")

    @field_validator("start_month", "end_month")
    @classmethod
    def validate_month(cls, value: str | None) -> str | None:
        if value is not None and not _is_month(value):
            raise ValueError("Month must use YYYY-MM format.")
        return value


class RetrievalQuery(BaseModel):
    """Structured representation of what retrieval should look up."""

    original_question: str = Field(description="The user's original question, unchanged.")
    intent: Intent = Field(description="The user's retrieval intent.")
    branches: list[str] = Field(default_factory=list, description="Branch names explicitly mentioned by the user.")
    kpis: list[KpiName] = Field(default_factory=list, description="Canonical KPI names requested by the user.")
    months: list[str] = Field(default_factory=list, description="Specific months explicitly requested, in YYYY-MM format.")
    time_range: TimeRange | None = Field(default=None, description="Optional inclusive month range.")
    comparison_mode: ComparisonMode = Field(default="general", description="How the user wants the data compared.")

    @field_validator("months")
    @classmethod
    def validate_months(cls, values: list[str]) -> list[str]:
        invalid = [value for value in values if not _is_month(value)]
        if invalid:
            raise ValueError("Months must use YYYY-MM format.")
        return values


QUERY_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You extract structured retrieval inputs for OpsPilot AI.
Use only these branch names: Hamburg, Munich, Berlin, Cologne, Frankfurt, Stuttgart, Dortmund, Leipzig.
Use only these KPI names: service_level, transportation_costs, staffing_level, complaints, customer_satisfaction.
Use only these intents: general_kpi_question, root_cause, critical_branches, similar_developments, cost_drivers, customer_satisfaction_drivers.
Use months only when the user explicitly gives a month or range. Month format must be YYYY-MM.
If a field is not explicit, leave lists empty and choose the closest intent/comparison mode.""",
        ),
        ("human", "{question}"),
    ]
)


def retrieve_context(conn, question: str, findings: list[Finding]) -> dict[str, Any]:
    """Structured RAG retrieval based on an extracted query object."""
    retrieval_query, parse_warning = build_retrieval_query(conn, question)
    intent = retrieval_query.intent
    branches = retrieval_query.branches

    if not branches and intent == "critical_branches":
        branches = _critical_branches(findings)

    if not branches and intent == "customer_satisfaction_drivers":
        branches = _low_satisfaction_branches(conn)

    context: dict[str, Any] = {
        "intent": intent,
        "branches": branches,
        "retrieval_query": retrieval_query.model_dump(),
        "findings": _matching_findings(findings, branches, retrieval_query),
    }
    if parse_warning:
        context["query_parse_warning"] = parse_warning

    if branches:
        context["monthly_kpis"] = _query_for_branches(conn, "monthly_kpis", branches, retrieval_query)
        context["delay_reasons"] = _query_for_branches(conn, "delay_reasons", branches, retrieval_query)
        context["staffing_events"] = _query_for_branches(conn, "staffing_events", branches, retrieval_query)
        context["customer_feedback"] = _query_for_branches(conn, "customer_feedback", branches, retrieval_query)
    else:
        context["latest_kpis"] = _latest_or_monthly_kpis(conn, retrieval_query)

    if intent == "cost_drivers":
        context["shipment_cost_sample"] = _shipment_cost_context(conn, branches, retrieval_query)

    if intent == "customer_satisfaction_drivers":
        context["customer_satisfaction_trend"] = _customer_satisfaction_trend(conn, branches, retrieval_query)
        context["feedback_summary"] = _customer_feedback_summary(conn, branches, retrieval_query)
        context["low_rating_feedback"] = _low_rating_feedback(conn, branches, retrieval_query)

    if intent == "similar_developments":
        context["similar_findings"] = _similar_findings(findings, retrieval_query)

    return context


def build_retrieval_query(conn, question: str) -> tuple[RetrievalQuery, str | None]:
    """Extract, normalize, and validate a retrieval query with manual fallback."""
    try:
        extracted = extract_retrieval_query(question)
        return _normalize_query(extracted, conn, question), None
    except Exception as exc:
        return _manual_retrieval_query(question), f"LLM query extraction failed; used manual fallback. Reason: {exc}"


def extract_retrieval_query(question: str) -> RetrievalQuery:
    """Use an LLM structured-output call to parse the user's retrieval request."""
    extracted = get_retrieval_query_chain().invoke({"question": question})
    return RetrievalQuery.model_validate(extracted)


@lru_cache(maxsize=1)
def get_retrieval_query_chain():
    from llm.client import get_chat_model

    return (
        QUERY_EXTRACTION_PROMPT
        | get_chat_model().with_structured_output(RetrievalQuery)
    ).with_config(run_name="retrieval_query_extraction", tags=["opspilot", "retrieval", "extraction"])


def detect_intent(question: str) -> str:
    q = question.lower()
    if any(word in q for word in ["satisfaction", "feedback", "rating", "comments", "customers unhappy", "bad customer"]):
        return "customer_satisfaction_drivers"
    if any(word in q for word in ["why", "root cause", "reason", "driver", "drives"]):
        if any(word in q for word in ["cost", "transport"]):
            return "cost_drivers"
        if any(word in q for word in ["customer", "satisfaction", "feedback", "rating", "comments"]):
            return "customer_satisfaction_drivers"
        return "root_cause"
    if any(word in q for word in ["critical", "risk", "worst", "attention"]):
        return "critical_branches"
    if any(word in q for word in ["similar", "other branches", "same pattern"]):
        return "similar_developments"
    if any(word in q for word in ["cost", "transport"]):
        return "cost_drivers"
    return "general_kpi_question"


def detect_branches(question: str) -> list[str]:
    q = question.lower()
    return [branch for branch in BRANCH_NAMES if branch.lower() in q]


def detect_kpis(question: str) -> list[KpiName]:
    q = question.lower()
    kpi_terms: dict[KpiName, list[str]] = {
        "service_level": ["service", "quality", "delivery"],
        "transportation_costs": ["cost", "transport"],
        "staffing_level": ["staff", "headcount", "overtime"],
        "complaints": ["complaint"],
        "customer_satisfaction": ["satisfaction", "customer", "rating"],
    }
    return [kpi for kpi, terms in kpi_terms.items() if any(term in q for term in terms)]


def context_to_text(context: dict[str, Any]) -> str:
    """Serialize retrieved context compactly for an LLM prompt."""
    sections = [f"Intent: {context['intent']}", f"Branches: {', '.join(context.get('branches') or ['all'])}"]

    for key, value in context.items():
        if key in {"intent", "branches"}:
            continue
        if isinstance(value, pd.DataFrame):
            sections.append(f"\n[{key}]\n{_df_to_text(value)}")
        elif isinstance(value, list):
            sections.append(f"\n[{key}]\n{_list_to_text(value)}")
        else:
            sections.append(f"\n[{key}]\n{value}")

    return "\n".join(sections)


def _query_for_branches(conn, table: str, branches: list[str], query: RetrievalQuery) -> pd.DataFrame:
    placeholders = ",".join("?" for _ in branches)
    month_clause, month_params = _month_filter(query, prefix="AND")
    return pd.read_sql_query(
        f"""
        SELECT *
        FROM {table}
        WHERE branch_name IN ({placeholders})
        {month_clause}
        ORDER BY month DESC
        LIMIT 80
        """,
        conn,
        params=tuple([*branches, *month_params]),
    )


def _latest_or_monthly_kpis(conn, query: RetrievalQuery) -> pd.DataFrame:
    month_clause, month_params = _month_filter(query, prefix="WHERE")
    if month_clause:
        return pd.read_sql_query(
            f"""
            SELECT *
            FROM monthly_kpis
            {month_clause}
            ORDER BY month DESC, service_level ASC
            LIMIT 80
            """,
            conn,
            params=tuple(month_params),
        )

    return pd.read_sql_query(
        """
        SELECT m.*
        FROM monthly_kpis m
        JOIN (
            SELECT branch_id, MAX(month) AS latest_month
            FROM monthly_kpis
            GROUP BY branch_id
        ) latest
        ON m.branch_id = latest.branch_id AND m.month = latest.latest_month
        ORDER BY service_level ASC
        """,
        conn,
    )


def _shipment_cost_context(conn, branches: list[str], query: RetrievalQuery) -> pd.DataFrame:
    if branches:
        placeholders = ",".join("?" for _ in branches)
        where = f"WHERE branch_name IN ({placeholders})"
        params = tuple(branches)
        month_prefix = "AND"
    else:
        where = ""
        params = []
        month_prefix = "WHERE"

    month_clause, month_params = _month_filter(query, prefix=month_prefix)

    return pd.read_sql_query(
        f"""
        SELECT branch_name, month,
               ROUND(AVG(transportation_cost), 2) AS avg_sample_cost,
               SUM(CASE WHEN status = 'Delayed' THEN 1 ELSE 0 END) AS sampled_delays,
               ROUND(AVG(delay_hours), 1) AS avg_delay_hours,
               SUM(damage_flag) AS sampled_damages
        FROM shipment_details
        {where}
        {month_clause}
        GROUP BY branch_name, month
        ORDER BY month DESC, avg_sample_cost DESC
        LIMIT 40
        """,
        conn,
        params=[*params, *month_params],
    )


def _low_satisfaction_branches(conn, limit: int = 4) -> list[str]:
    df = pd.read_sql_query(
        """
        SELECT branch_name, customer_satisfaction
        FROM monthly_kpis
        WHERE month = (SELECT MAX(month) FROM monthly_kpis)
        ORDER BY customer_satisfaction ASC
        LIMIT ?
        """,
        conn,
        params=[limit],
    )
    return df["branch_name"].tolist()


def _customer_satisfaction_trend(conn, branches: list[str], query: RetrievalQuery) -> pd.DataFrame:
    if branches:
        placeholders = ",".join("?" for _ in branches)
        where = f"WHERE branch_name IN ({placeholders})"
        params = branches
        month_prefix = "AND"
    else:
        where = ""
        params = []
        month_prefix = "WHERE"

    month_clause, month_params = _month_filter(query, prefix=month_prefix)

    return pd.read_sql_query(
        f"""
        SELECT branch_name, month, customer_satisfaction, complaints, service_level, damage_rate
        FROM monthly_kpis
        {where}
        {month_clause}
        ORDER BY branch_name, month
        """,
        conn,
        params=[*params, *month_params],
    )


def _customer_feedback_summary(conn, branches: list[str], query: RetrievalQuery) -> pd.DataFrame:
    if branches:
        placeholders = ",".join("?" for _ in branches)
        where = f"WHERE branch_name IN ({placeholders})"
        params = branches
        month_prefix = "AND"
    else:
        where = ""
        params = []
        month_prefix = "WHERE"

    month_clause, month_params = _month_filter(query, prefix=month_prefix)

    return pd.read_sql_query(
        f"""
        SELECT branch_name,
               month,
               category,
               sentiment,
               COUNT(*) AS feedback_count,
               ROUND(AVG(rating), 2) AS avg_rating
        FROM customer_feedback
        {where}
        {month_clause}
        GROUP BY branch_name, month, category, sentiment
        ORDER BY month DESC, branch_name, avg_rating ASC, feedback_count DESC
        LIMIT 80
        """,
        conn,
        params=[*params, *month_params],
    )


def _low_rating_feedback(conn, branches: list[str], query: RetrievalQuery) -> pd.DataFrame:
    if branches:
        placeholders = ",".join("?" for _ in branches)
        where = f"AND branch_name IN ({placeholders})"
        params = branches
    else:
        where = ""
        params = []

    month_clause, month_params = _month_filter(query, prefix="AND")

    return pd.read_sql_query(
        f"""
        SELECT branch_name, month, sentiment, category, rating, comment
        FROM customer_feedback
        WHERE rating <= 3
        {where}
        {month_clause}
        ORDER BY month DESC, rating ASC
        LIMIT 40
        """,
        conn,
        params=[*params, *month_params],
    )


def _matching_findings(findings: list[Finding], branches: list[str], query: RetrievalQuery) -> list[dict[str, Any]]:
    matched = []
    for finding in findings:
        branch_match = not branches or finding.branch_name in branches
        kpi_match = not query.kpis or finding.kpi in query.kpis
        if branch_match and kpi_match and _finding_month_match(finding, query):
            matched.append(finding.to_dict())
    return matched[:12]


def _critical_branches(findings: list[Finding]) -> list[str]:
    critical = []
    for finding in findings:
        if finding.severity == "critical" and finding.branch_name not in critical:
            critical.append(finding.branch_name)
    return critical[:4]


def _similar_findings(findings: list[Finding], query: RetrievalQuery) -> list[dict[str, Any]]:
    rows = []
    for finding in findings:
        if (not query.kpis or finding.kpi in query.kpis) and _finding_month_match(finding, query):
            rows.append(finding.to_dict())
    return rows[:10]


def _manual_retrieval_query(question: str) -> RetrievalQuery:
    months = _extract_months(question)
    return RetrievalQuery(
        original_question=question,
        intent=detect_intent(question),
        branches=detect_branches(question),
        kpis=detect_kpis(question),
        months=months,
        comparison_mode=_detect_comparison_mode(question),
    )


def _normalize_query(query: RetrievalQuery, conn, question: str) -> RetrievalQuery:
    branch_lookup = {branch.lower(): branch for branch in BRANCH_NAMES}
    normalized_branches = []
    unknown_branches = []
    for branch in query.branches:
        match = branch_lookup.get(branch.lower())
        if match:
            normalized_branches.append(match)
        else:
            unknown_branches.append(branch)

    unknown_kpis = [kpi for kpi in query.kpis if kpi not in KPI_NAMES]
    available_months = _available_months(conn)
    requested_months = _query_months(query)
    unknown_months = [month for month in requested_months if month not in available_months]
    if unknown_branches or unknown_kpis or unknown_months:
        raise ValueError(
            "Unsupported extracted fields: "
            f"branches={unknown_branches}, kpis={unknown_kpis}, months={unknown_months}"
        )

    return query.model_copy(
        update={
            "original_question": question,
            "branches": _dedupe(normalized_branches),
            "kpis": _dedupe(query.kpis),
            "months": _dedupe(query.months),
        }
    )


def _month_filter(query: RetrievalQuery, prefix: str) -> tuple[str, list[str]]:
    if query.months:
        placeholders = ",".join("?" for _ in query.months)
        return f"{prefix} month IN ({placeholders})", query.months

    if query.time_range and query.time_range.start_month and query.time_range.end_month:
        return f"{prefix} month BETWEEN ? AND ?", [query.time_range.start_month, query.time_range.end_month]

    if query.time_range and query.time_range.start_month:
        return f"{prefix} month >= ?", [query.time_range.start_month]

    if query.time_range and query.time_range.end_month:
        return f"{prefix} month <= ?", [query.time_range.end_month]

    return "", []


def _finding_month_match(finding: Finding, query: RetrievalQuery) -> bool:
    if query.months:
        return finding.month in query.months

    if query.time_range and query.time_range.start_month and finding.month < query.time_range.start_month:
        return False

    if query.time_range and query.time_range.end_month and finding.month > query.time_range.end_month:
        return False

    return True


def _query_months(query: RetrievalQuery) -> list[str]:
    months = list(query.months)
    if query.time_range:
        if query.time_range.start_month:
            months.append(query.time_range.start_month)
        if query.time_range.end_month:
            months.append(query.time_range.end_month)
    return months


def _available_months(conn) -> set[str]:
    df = pd.read_sql_query("SELECT DISTINCT month FROM monthly_kpis", conn)
    return set(df["month"].tolist())


def _extract_months(question: str) -> list[str]:
    return re.findall(r"\b20\d{2}-(?:0[1-9]|1[0-2])\b", question)


def _detect_comparison_mode(question: str) -> ComparisonMode:
    intent = detect_intent(question)
    if intent == "root_cause":
        return "root_cause"
    if intent == "similar_developments":
        return "similar_branches"
    if intent == "critical_branches":
        return "critical"
    if intent in {"cost_drivers", "customer_satisfaction_drivers"}:
        return "drivers"
    if any(word in question.lower() for word in ["trend", "over time", "develop"]):
        return "trend"
    if any(word in question.lower() for word in ["latest", "current", "currently"]):
        return "latest"
    return "general"


def _dedupe(values: list[Any]) -> list[Any]:
    return list(dict.fromkeys(values))


def _is_month(value: str) -> bool:
    return bool(re.fullmatch(r"20\d{2}-(0[1-9]|1[0-2])", value))


def _df_to_text(df: pd.DataFrame) -> str:
    if df.empty:
        return "No rows retrieved."
    return df.to_markdown(index=False)


def _list_to_text(items: list[Any]) -> str:
    if not items:
        return "No matching records."
    return "\n".join(str(item) for item in items)
