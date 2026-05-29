from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.kpi_analysis import analyze_kpis, findings_to_frame, latest_branch_snapshot, load_monthly_kpis
from analysis.retrieval import retrieve_context
from database.connection import DEFAULT_DB_PATH, get_connection
from database.setup_database import create_database
from llm.client import MODEL_NAME, answer_question, generate_management_summary


st.set_page_config(page_title="OpsPilot AI", page_icon=":bar_chart:", layout="wide")


@st.cache_resource
def ensure_database():
    return create_database(DEFAULT_DB_PATH)


@st.cache_data
def load_data():
    ensure_database()
    with get_connection(DEFAULT_DB_PATH) as conn:
        monthly = load_monthly_kpis(conn)
    findings = analyze_kpis(monthly)
    return monthly, findings


def main() -> None:
    monthly, findings = load_data()
    findings_df = findings_to_frame(findings)

    st.title("OpsPilot AI")
    st.caption("AI-powered operational analytics assistant for logistics branch management")

    with st.sidebar:
        st.header("Controls")
        branches = ["All branches"] + sorted(monthly["branch_name"].unique().tolist())
        selected_branch = st.selectbox("Branch", branches)
        st.caption(f"LLM model: `{MODEL_NAME}`")
        if st.button("Rebuild demo database"):
            create_database(DEFAULT_DB_PATH, force=True)
            st.cache_data.clear()
            st.rerun()

    filtered = monthly if selected_branch == "All branches" else monthly[monthly["branch_name"] == selected_branch]
    latest = latest_branch_snapshot(filtered)

    render_kpi_overview(latest)
    render_dashboard(filtered, findings_df, selected_branch)
    render_management_summary(findings)
    render_chatbot(findings)


def render_kpi_overview(latest: pd.DataFrame) -> None:
    st.subheader("Current KPI Overview")
    cols = st.columns(5)
    cols[0].metric("Avg. service level", f"{latest['service_level'].mean():.1f}%")
    cols[1].metric("Delayed shipments", f"{int(latest['delayed_shipments'].sum()):,}")
    cols[2].metric("Complaints", f"{int(latest['complaints'].sum()):,}")
    cols[3].metric("Avg. cost / shipment", f"EUR {latest['cost_per_shipment'].mean():.2f}")
    cols[4].metric("Avg. satisfaction", f"{latest['customer_satisfaction'].mean():.2f}/5")


def render_dashboard(filtered: pd.DataFrame, findings_df: pd.DataFrame, selected_branch: str) -> None:
    left, right = st.columns([2, 1])

    with left:
        st.subheader("Trend Analysis")
        chart_df = filtered.copy()
        chart_df["cost_per_shipment"] = chart_df["transportation_costs"] / chart_df["shipment_volume"]
        service_view = chart_df.pivot_table(index="month", columns="branch_name", values="service_level")
        st.caption("Service level by branch")
        st.line_chart(service_view)

        satisfaction_view = chart_df.pivot_table(index="month", columns="branch_name", values="customer_satisfaction")
        st.caption("Customer satisfaction by branch")
        st.line_chart(satisfaction_view)

        cost_view = chart_df.pivot_table(index="month", columns="branch_name", values="cost_per_shipment")
        st.caption("Transportation cost per shipment")
        st.line_chart(cost_view)

    with right:
        st.subheader("Anomaly Highlights")
        display_findings = findings_df
        if selected_branch != "All branches" and not findings_df.empty:
            display_findings = findings_df[findings_df["branch_name"] == selected_branch]

        if display_findings.empty:
            st.info("No anomalies detected for the current filter.")
        else:
            for _, row in display_findings.head(8).iterrows():
                if row["severity"] == "critical":
                    st.error(row["title"])
                elif row["severity"] == "warning":
                    st.warning(row["title"])
                else:
                    st.info(row["title"])
                st.caption(row["evidence"])

    st.subheader("Latest Branch Snapshot")
    snapshot = latest_branch_snapshot(filtered)
    st.dataframe(
        snapshot[
            [
                "branch_name",
                "month",
                "service_level",
                "delayed_shipments",
                "complaints",
                "staffing_level",
                "cost_per_shipment",
                "damage_rate",
                "overtime_hours",
                "customer_satisfaction",
            ]
        ],
        width="stretch",
        hide_index=True,
    )


def render_management_summary(findings) -> None:
    st.subheader("AI Management Summary")
    if "management_summary" not in st.session_state:
        with st.spinner("Generating executive summary..."):
            st.session_state.management_summary = generate_management_summary(findings)
    st.write(st.session_state.management_summary)


def render_chatbot(findings) -> None:
    st.subheader("Conversational Analytics")
    st.caption(
        "Example prompts: Why did service quality decrease in Hamburg? "
        "Why was customer satisfaction bad in Hamburg? "
        "Which branches are currently critical? What drives transportation costs in Munich?"
    )

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Ask me about branch risks, KPI trends, cost drivers, or likely root causes. I will ground answers in SQLite records and analysis findings.",
            }
        ]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    question = st.chat_input("Ask a follow-up question")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving structured context and asking the model..."):
                with get_connection(DEFAULT_DB_PATH) as conn:
                    context = retrieve_context(conn, question, findings)
                answer = answer_question(question, context)
                st.write(answer)
                with st.expander("Retrieved context"):
                    from analysis.retrieval import context_to_text

                    st.code(context_to_text(context), language="text")
        st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
