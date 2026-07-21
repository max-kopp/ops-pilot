from __future__ import annotations

from html import escape
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
from llm.client import MODEL_NAME, generate_management_summary, stream_answer_question


st.set_page_config(page_title="OpsPilot AI", page_icon=":bar_chart:", layout="wide")


APP_STYLES = """
<style>
    :root {
        --op-ink: #17223b;
        --op-muted: #667085;
        --op-border: #e5e9f0;
        --op-surface: #ffffff;
        --op-soft: #f5f7fb;
        --op-blue: #3659e3;
        --op-cyan: #18a6b8;
        --op-red: #d04444;
        --op-amber: #c98212;
    }

    .stApp { background: #f5f7fb; }
    [data-testid="stHeader"] { background: transparent; }
    .block-container { max-width: 1480px; padding-top: 2rem; padding-bottom: 4rem; }

    [data-testid="stSidebar"] {
        background: #111b32;
        border-right: 1px solid rgba(255,255,255,.08);
    }
    [data-testid="stSidebar"] * { color: #eef2ff; }
    [data-testid="stSidebar"] [data-baseweb="select"] > div {
        background: rgba(255,255,255,.08);
        border-color: rgba(255,255,255,.16);
    }
    [data-testid="stSidebar"] .stButton button {
        background: rgba(255,255,255,.07);
        border-color: rgba(255,255,255,.16);
    }

    .op-brand { display: flex; align-items: center; gap: .7rem; margin: .2rem 0 1.6rem; }
    .op-brand-mark {
        display: grid; place-items: center; width: 2.25rem; height: 2.25rem;
        border-radius: .7rem; color: white; font-weight: 800;
        background: linear-gradient(135deg, #5b7cff, #20b8bd);
        box-shadow: 0 8px 24px rgba(54,89,227,.35);
    }
    .op-brand-name { font-size: 1.08rem; font-weight: 750; letter-spacing: -.02em; }
    .op-sidebar-label { color: #93a4c7 !important; font-size: .72rem; font-weight: 700; letter-spacing: .09em; text-transform: uppercase; }

    .op-header {
        display: flex; align-items: center; justify-content: space-between; gap: 1.5rem;
        padding: 1.25rem 1.4rem; margin-bottom: 1.55rem; border: 1px solid var(--op-border);
        border-top: 4px solid var(--op-blue); border-radius: 1rem; background: var(--op-surface);
        box-shadow: 0 5px 18px rgba(21,34,61,.045);
    }
    .op-header-title { color: var(--op-ink); font-size: 1.65rem; font-weight: 780; letter-spacing: -.035em; line-height: 1.15; }
    .op-header-subtitle { color: var(--op-muted); font-size: .84rem; margin-top: .35rem; }
    .op-header-facts { display: flex; flex-wrap: wrap; justify-content: flex-end; gap: .65rem; }
    .op-header-fact { min-width: 118px; padding: .55rem .7rem; border-radius: .7rem; background: var(--op-soft); }
    .op-header-fact span { display: block; color: var(--op-muted); font-size: .64rem; font-weight: 720; letter-spacing: .07em; text-transform: uppercase; }
    .op-header-fact strong { display: block; color: var(--op-ink); font-size: .86rem; margin-top: .15rem; }
    .op-header-fact.alert strong { color: var(--op-red); }

    h2, h3 { color: var(--op-ink); letter-spacing: -.025em; }
    [data-testid="stMetric"] {
        min-height: 128px; padding: 1.05rem 1.1rem; border: 1px solid var(--op-border);
        border-radius: 1rem; background: var(--op-surface); box-shadow: 0 5px 18px rgba(21,34,61,.045);
    }
    [data-testid="stMetricLabel"] { color: var(--op-muted); font-size: .82rem; font-weight: 650; }
    [data-testid="stMetricValue"] { color: var(--op-ink); font-weight: 750; letter-spacing: -.035em; }
    [data-testid="stMetricDelta"] { font-size: .75rem; }

    [data-testid="stVerticalBlockBorderWrapper"] {
        border-color: var(--op-border) !important; border-radius: 1rem !important;
        background: var(--op-surface); box-shadow: 0 5px 18px rgba(21,34,61,.04);
    }
    .op-card-title { color: var(--op-ink); font-size: 1.05rem; font-weight: 750; letter-spacing: -.02em; margin-bottom: .15rem; }
    .op-card-subtitle { color: var(--op-muted); font-size: .78rem; margin-bottom: .8rem; }
    .op-alert {
        padding: .85rem .9rem; margin-bottom: .7rem; border: 1px solid var(--op-border);
        border-left-width: 4px; border-radius: .8rem; background: #fff;
    }
    .op-alert.critical { border-left-color: var(--op-red); background: #fffafa; }
    .op-alert.warning { border-left-color: var(--op-amber); background: #fffcf6; }
    .op-alert.info { border-left-color: var(--op-blue); background: #f8faff; }
    .op-alert-meta { color: var(--op-muted); font-size: .69rem; font-weight: 750; letter-spacing: .07em; text-transform: uppercase; }
    .op-alert-title { color: var(--op-ink); font-size: .85rem; font-weight: 680; line-height: 1.4; margin: .22rem 0 .25rem; }
    .op-alert-evidence { color: var(--op-muted); font-size: .75rem; line-height: 1.4; }

    [data-testid="stDataFrame"] { border: 1px solid var(--op-border); border-radius: .9rem; overflow: hidden; }
    [data-testid="stDataFrame"], [data-testid="stVegaLiteChart"] { background: #fff; }
    .stTabs [data-baseweb="tab-list"] { gap: .35rem; border-bottom: 1px solid var(--op-border); }
    .stTabs [data-baseweb="tab"] { height: 2.5rem; padding: 0 .75rem; font-size: .82rem; }
    .stTabs [data-baseweb="tab"] p, [data-testid="stDataFrame"] { color: var(--op-ink); }
    [data-testid="stChatMessage"] { border: 1px solid var(--op-border); border-radius: .9rem; background: #fff; padding: .45rem .7rem; }
    [data-testid="stChatMessage"] p, [data-testid="stCaptionContainer"] p { color: var(--op-muted); }
    [data-testid="stChatInput"] { background: #fff; }
    hr { border-color: var(--op-border); margin: 1.7rem 0; }

    @media (max-width: 800px) {
        .op-header { align-items: flex-start; flex-direction: column; }
        .op-header-facts { justify-content: flex-start; }
    }
</style>
"""


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

    st.markdown(APP_STYLES, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown(
            '<div class="op-brand"><div class="op-brand-mark">O</div>'
            '<div class="op-brand-name">OpsPilot AI</div></div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="op-sidebar-label">Dashboard controls</div>', unsafe_allow_html=True)
        branches = ["All branches"] + sorted(monthly["branch_name"].unique().tolist())
        selected_branch = st.selectbox("Branch view", branches)
        st.caption(f"Analytics copilot · `{MODEL_NAME}`")
        st.divider()
        if st.button("↻  Rebuild demo database", use_container_width=True):
            create_database(DEFAULT_DB_PATH, force=True)
            st.cache_data.clear()
            st.rerun()

    filtered = monthly if selected_branch == "All branches" else monthly[monthly["branch_name"] == selected_branch]
    latest = latest_branch_snapshot(filtered)

    scope_label = selected_branch if selected_branch != "All branches" else f"All {len(latest)} branches"
    latest_month = pd.to_datetime(filtered["month"]).max().strftime("%B %Y")
    scoped_findings = findings_df
    if selected_branch != "All branches" and not findings_df.empty:
        scoped_findings = findings_df[findings_df["branch_name"] == selected_branch]
    critical_count = int((scoped_findings["severity"] == "critical").sum()) if not scoped_findings.empty else 0
    st.markdown(
        f"""
        <section class="op-header">
            <div>
                <div class="op-header-title">Operations dashboard</div>
                <div class="op-header-subtitle">Current logistics network performance and exceptions</div>
            </div>
            <div class="op-header-facts">
                <div class="op-header-fact"><span>Scope</span><strong>{escape(scope_label)}</strong></div>
                <div class="op-header-fact"><span>Reporting period</span><strong>{latest_month}</strong></div>
                <div class="op-header-fact alert"><span>Critical signals</span><strong>{critical_count}</strong></div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    render_kpi_overview(filtered, latest)
    render_dashboard(filtered, findings_df, selected_branch)
    render_management_summary(findings)
    render_chatbot(findings)


def render_kpi_overview(filtered: pd.DataFrame, latest: pd.DataFrame) -> None:
    st.subheader("Performance at a glance")
    st.caption("Latest reporting period with change versus the prior month")
    ordered_months = sorted(filtered["month"].unique())
    previous = filtered[filtered["month"] == ordered_months[-2]] if len(ordered_months) > 1 else latest

    current_service = latest["service_level"].mean()
    current_delays = int(latest["delayed_shipments"].sum())
    current_complaints = int(latest["complaints"].sum())
    current_cost = latest["cost_per_shipment"].mean()
    current_satisfaction = latest["customer_satisfaction"].mean()
    previous_cost = (previous["transportation_costs"] / previous["shipment_volume"]).mean()

    cols = st.columns(5)
    cols[0].metric("Service level", f"{current_service:.1f}%", f"{current_service - previous['service_level'].mean():+.1f} pp")
    cols[1].metric(
        "Delayed shipments",
        f"{current_delays:,}",
        f"{current_delays - int(previous['delayed_shipments'].sum()):+,}",
        delta_color="inverse",
    )
    cols[2].metric(
        "Complaints",
        f"{current_complaints:,}",
        f"{current_complaints - int(previous['complaints'].sum()):+,}",
        delta_color="inverse",
    )
    cols[3].metric("Cost / shipment", f"€{current_cost:.2f}", f"€{current_cost - previous_cost:+.2f}", delta_color="inverse")
    cols[4].metric(
        "Satisfaction",
        f"{current_satisfaction:.2f}/5",
        f"{current_satisfaction - previous['customer_satisfaction'].mean():+.2f}",
    )


def render_dashboard(filtered: pd.DataFrame, findings_df: pd.DataFrame, selected_branch: str) -> None:
    st.divider()
    left, right = st.columns([2, 1])

    with left:
        with st.container(border=True):
            st.markdown('<div class="op-card-title">Trend explorer</div>', unsafe_allow_html=True)
            st.markdown('<div class="op-card-subtitle">Compare monthly performance and spot emerging movement</div>', unsafe_allow_html=True)
            chart_df = filtered.copy()
            chart_df["cost_per_shipment"] = chart_df["transportation_costs"] / chart_df["shipment_volume"]
            service_view = chart_df.pivot_table(index="month", columns="branch_name", values="service_level")
            satisfaction_view = chart_df.pivot_table(index="month", columns="branch_name", values="customer_satisfaction")
            cost_view = chart_df.pivot_table(index="month", columns="branch_name", values="cost_per_shipment")
            service_tab, satisfaction_tab, cost_tab = st.tabs(["Service level", "Satisfaction", "Cost efficiency"])
            with service_tab:
                st.line_chart(service_view, height=360, y_label="Service level (%)")
            with satisfaction_tab:
                st.line_chart(satisfaction_view, height=360, y_label="Satisfaction (1–5)")
            with cost_tab:
                st.line_chart(cost_view, height=360, y_label="Cost per shipment (€)")

    with right:
        with st.container(border=True, height=495):
            st.markdown('<div class="op-card-title">Priority signals</div>', unsafe_allow_html=True)
            st.markdown('<div class="op-card-subtitle">Highest-impact anomalies requiring attention</div>', unsafe_allow_html=True)
            display_findings = findings_df
            if selected_branch != "All branches" and not findings_df.empty:
                display_findings = findings_df[findings_df["branch_name"] == selected_branch]

            if display_findings.empty:
                st.info("No anomalies detected for the current filter.")
            else:
                for _, row in display_findings.head(6).iterrows():
                    severity = row["severity"] if row["severity"] in {"critical", "warning", "info"} else "info"
                    st.markdown(
                        f"""
                        <div class="op-alert {severity}">
                            <div class="op-alert-meta">{escape(severity)} · {escape(str(row['branch_name']))}</div>
                            <div class="op-alert-title">{escape(str(row['title']))}</div>
                            <div class="op-alert-evidence">{escape(str(row['evidence']))}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

    st.subheader("Branch snapshot")
    st.caption("Latest available operational KPIs, ordered by service level")
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
        column_config={
            "branch_name": st.column_config.TextColumn("Branch", width="medium"),
            "month": st.column_config.TextColumn("Period"),
            "service_level": st.column_config.NumberColumn("Service", format="%.1f%%"),
            "delayed_shipments": st.column_config.NumberColumn("Delayed", format="%d"),
            "complaints": st.column_config.NumberColumn("Complaints", format="%d"),
            "staffing_level": st.column_config.NumberColumn("Staff", format="%d FTE"),
            "cost_per_shipment": st.column_config.NumberColumn("Cost / shipment", format="€%.2f"),
            "damage_rate": st.column_config.NumberColumn("Damage rate", format="%.2f%%"),
            "overtime_hours": st.column_config.NumberColumn("Overtime", format="%.1f h"),
            "customer_satisfaction": st.column_config.NumberColumn("Satisfaction", format="%.2f / 5"),
        },
    )


def render_management_summary(findings) -> None:
    st.divider()
    with st.container(border=True):
        st.markdown('<div class="op-card-title">AI management brief</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="op-card-subtitle">A concise executive readout grounded in the latest detected signals</div>',
            unsafe_allow_html=True,
        )
        if "management_summary" not in st.session_state:
            with st.spinner("Generating executive summary..."):
                st.session_state.management_summary = generate_management_summary(findings)
        st.write(st.session_state.management_summary)


def render_chatbot(findings) -> None:
    st.subheader("Ask OpsPilot")
    st.caption(
        "Explore the data in plain language. Try: Why did service quality decrease in Hamburg? "
        "Why was customer satisfaction bad in Hamburg? "
        "Which branches are currently critical? What drives transportation costs in Munich?"
    )

    if "messages" not in st.session_state:
        st.session_state.messages = initial_chat_messages()

    if st.button("Clear conversation"):
        st.session_state.messages = initial_chat_messages()
        st.rerun()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    question = st.chat_input("Ask a follow-up question")
    if question:
        chat_history = __import__("typing").cast(list[dict[str, str]], st.session_state.messages).copy()
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving structured context..."):
                with get_connection(DEFAULT_DB_PATH) as conn:
                    context = retrieve_context(conn, question, findings, chat_history)
            answer = st.write_stream(stream_answer_question(question, context, chat_history))
            answer_text = " ".join(answer) if isinstance(answer, list) else str(answer)
            with st.expander("Retrieved context"):
                from analysis.retrieval import context_to_text

                st.code(context_to_text(context), language="text")
        st.session_state.messages.append({"role": "assistant", "content": answer_text})


def initial_chat_messages() -> list[dict[str, str]]:
    return [
        {
            "role": "assistant",
            "content": "Ask me about branch risks, KPI trends, cost drivers, or likely root causes. I will ground answers in SQLite records and analysis findings.",
        }
    ]


if __name__ == "__main__":
    main()
