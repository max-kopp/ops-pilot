from __future__ import annotations

import re
from typing import Any

import pandas as pd

from analysis.kpi_analysis import Finding


BRANCH_NAMES = ["Hamburg", "Munich", "Berlin", "Cologne", "Frankfurt", "Stuttgart", "Dortmund", "Leipzig"]


def retrieve_context(conn, question: str, findings: list[Finding]) -> dict[str, Any]:
    """Small manual orchestrator for RAG-style structured retrieval."""
    intent = detect_intent(question)
    branches = detect_branches(question)

    if not branches and intent == "critical_branches":
        branches = _critical_branches(findings)

    if not branches and intent == "customer_satisfaction_drivers":
        branches = _low_satisfaction_branches(conn)

    context: dict[str, Any] = {
        "intent": intent,
        "branches": branches,
        "findings": _matching_findings(findings, branches, question),
    }

    if branches:
        context["monthly_kpis"] = _query_for_branches(conn, "monthly_kpis", branches)
        context["delay_reasons"] = _query_for_branches(conn, "delay_reasons", branches)
        context["staffing_events"] = _query_for_branches(conn, "staffing_events", branches)
        context["customer_feedback"] = _query_for_branches(conn, "customer_feedback", branches)
    else:
        context["latest_kpis"] = pd.read_sql_query(
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

    if intent == "cost_drivers":
        context["shipment_cost_sample"] = _shipment_cost_context(conn, branches)

    if intent == "customer_satisfaction_drivers":
        context["customer_satisfaction_trend"] = _customer_satisfaction_trend(conn, branches)
        context["feedback_summary"] = _customer_feedback_summary(conn, branches)
        context["low_rating_feedback"] = _low_rating_feedback(conn, branches)

    if intent == "similar_developments":
        context["similar_findings"] = _similar_findings(findings, question)

    return context


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


def _query_for_branches(conn, table: str, branches: list[str]) -> pd.DataFrame:
    placeholders = ",".join("?" for _ in branches)
    return pd.read_sql_query(
        f"SELECT * FROM {table} WHERE branch_name IN ({placeholders}) ORDER BY month DESC LIMIT 80",
        conn,
        params=branches,
    )


def _shipment_cost_context(conn, branches: list[str]) -> pd.DataFrame:
    if branches:
        placeholders = ",".join("?" for _ in branches)
        where = f"WHERE branch_name IN ({placeholders})"
        params = branches
    else:
        where = ""
        params = []

    return pd.read_sql_query(
        f"""
        SELECT branch_name, month,
               ROUND(AVG(transportation_cost), 2) AS avg_sample_cost,
               SUM(CASE WHEN status = 'Delayed' THEN 1 ELSE 0 END) AS sampled_delays,
               ROUND(AVG(delay_hours), 1) AS avg_delay_hours,
               SUM(damage_flag) AS sampled_damages
        FROM shipment_details
        {where}
        GROUP BY branch_name, month
        ORDER BY month DESC, avg_sample_cost DESC
        LIMIT 40
        """,
        conn,
        params=params,
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


def _customer_satisfaction_trend(conn, branches: list[str]) -> pd.DataFrame:
    if branches:
        placeholders = ",".join("?" for _ in branches)
        where = f"WHERE branch_name IN ({placeholders})"
        params = branches
    else:
        where = ""
        params = []

    return pd.read_sql_query(
        f"""
        SELECT branch_name, month, customer_satisfaction, complaints, service_level, damage_rate
        FROM monthly_kpis
        {where}
        ORDER BY branch_name, month
        """,
        conn,
        params=params,
    )


def _customer_feedback_summary(conn, branches: list[str]) -> pd.DataFrame:
    if branches:
        placeholders = ",".join("?" for _ in branches)
        where = f"WHERE branch_name IN ({placeholders})"
        params = branches
    else:
        where = ""
        params = []

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
        GROUP BY branch_name, month, category, sentiment
        ORDER BY month DESC, branch_name, avg_rating ASC, feedback_count DESC
        LIMIT 80
        """,
        conn,
        params=params,
    )


def _low_rating_feedback(conn, branches: list[str]) -> pd.DataFrame:
    if branches:
        placeholders = ",".join("?" for _ in branches)
        where = f"AND branch_name IN ({placeholders})"
        params = branches
    else:
        where = ""
        params = []

    return pd.read_sql_query(
        f"""
        SELECT branch_name, month, sentiment, category, rating, comment
        FROM customer_feedback
        WHERE rating <= 3
        {where}
        ORDER BY month DESC, rating ASC
        LIMIT 40
        """,
        conn,
        params=params,
    )


def _matching_findings(findings: list[Finding], branches: list[str], question: str) -> list[dict[str, Any]]:
    q = question.lower()
    kpi_terms = {
        "service_level": ["service", "quality", "delivery"],
        "transportation_costs": ["cost", "transport"],
        "staffing_level": ["staff", "headcount", "overtime"],
        "complaints": ["complaint"],
        "customer_satisfaction": ["satisfaction", "customer"],
    }
    matched = []
    for finding in findings:
        branch_match = not branches or finding.branch_name in branches
        kpi_match = any(term in q for term in kpi_terms.get(finding.kpi, [])) or not any(
            term in q for terms in kpi_terms.values() for term in terms
        )
        if branch_match and kpi_match:
            matched.append(finding.to_dict())
    return matched[:12]


def _critical_branches(findings: list[Finding]) -> list[str]:
    critical = []
    for finding in findings:
        if finding.severity == "critical" and finding.branch_name not in critical:
            critical.append(finding.branch_name)
    return critical[:4]


def _similar_findings(findings: list[Finding], question: str) -> list[dict[str, Any]]:
    q = question.lower()
    requested_kpis = [kpi for kpi in ["service", "cost", "staff", "complaint", "satisfaction"] if kpi in q]
    rows = []
    for finding in findings:
        if not requested_kpis or any(re.search(term, finding.kpi.replace("_", " ")) for term in requested_kpis):
            rows.append(finding.to_dict())
    return rows[:10]


def _df_to_text(df: pd.DataFrame) -> str:
    if df.empty:
        return "No rows retrieved."
    return df.to_markdown(index=False)


def _list_to_text(items: list[Any]) -> str:
    if not items:
        return "No matching records."
    return "\n".join(str(item) for item in items)
