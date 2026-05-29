from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import pandas as pd


Severity = Literal["critical", "warning", "info"]


@dataclass
class Finding:
    finding_id: str
    branch_id: str
    branch_name: str
    month: str
    kpi: str
    severity: Severity
    title: str
    description: str
    evidence: str
    value: float
    previous_value: float
    change_abs: float
    change_pct: float

    def to_dict(self) -> dict:
        return asdict(self)


def load_monthly_kpis(conn) -> pd.DataFrame:
    df = pd.read_sql_query("SELECT * FROM monthly_kpis ORDER BY branch_name, month", conn)
    return df


def analyze_kpis(monthly_kpis: pd.DataFrame) -> list[Finding]:
    """Create structured business findings from monthly KPI movements."""
    findings: list[Finding] = []
    df = monthly_kpis.sort_values(["branch_name", "month"]).copy()
    df["cost_per_shipment"] = df["transportation_costs"] / df["shipment_volume"]

    for _, branch_df in df.groupby("branch_id"):
        branch_df = branch_df.reset_index(drop=True)
        branch_name = branch_df.loc[0, "branch_name"]
        branch_id = branch_df.loc[0, "branch_id"]

        for idx in range(3, len(branch_df)):
            current = branch_df.loc[idx]
            previous = branch_df.loc[idx - 3]
            month = current["month"]

            service_delta = current["service_level"] - previous["service_level"]
            if service_delta <= -4.0:
                findings.append(
                    _finding(
                        branch_id,
                        branch_name,
                        month,
                        "service_level",
                        "critical" if service_delta <= -6 else "warning",
                        f"Service level in {branch_name} decreased by {abs(service_delta):.1f} points over 3 months",
                        "The branch shows a sustained deterioration in delivery reliability.",
                        f"{previous['service_level']:.1f}% in {previous['month']} to {current['service_level']:.1f}% in {month}",
                        current["service_level"],
                        previous["service_level"],
                    )
                )

            cost_delta_pct = _pct_change(previous["cost_per_shipment"], current["cost_per_shipment"])
            if cost_delta_pct >= 12:
                findings.append(
                    _finding(
                        branch_id,
                        branch_name,
                        month,
                        "transportation_costs",
                        "critical" if cost_delta_pct >= 20 else "warning",
                        f"Transportation cost per shipment in {branch_name} increased by {cost_delta_pct:.1f}% over 3 months",
                        "The cost trend is materially above normal month-to-month variation.",
                        f"EUR {previous['cost_per_shipment']:.2f} to EUR {current['cost_per_shipment']:.2f} per shipment",
                        current["cost_per_shipment"],
                        previous["cost_per_shipment"],
                    )
                )

            staffing_delta = current["staffing_level"] - previous["staffing_level"]
            overtime_delta_pct = _pct_change(previous["overtime_hours"], current["overtime_hours"])
            if staffing_delta <= -3 and overtime_delta_pct >= 20:
                findings.append(
                    _finding(
                        branch_id,
                        branch_name,
                        month,
                        "staffing_level",
                        "critical",
                        f"Staffing pressure in {branch_name}: headcount down {abs(staffing_delta):.0f} while overtime rose {overtime_delta_pct:.1f}%",
                        "Lower staffing and higher overtime indicate capacity stress and service risk.",
                        f"{previous['staffing_level']:.0f} to {current['staffing_level']:.0f} FTE; overtime {previous['overtime_hours']:.1f} to {current['overtime_hours']:.1f} hours",
                        current["staffing_level"],
                        previous["staffing_level"],
                    )
                )

            complaints_delta_pct = _pct_change(previous["complaints"], current["complaints"])
            if complaints_delta_pct >= 35 and current["complaints"] >= 20:
                findings.append(
                    _finding(
                        branch_id,
                        branch_name,
                        month,
                        "complaints",
                        "warning",
                        f"Customer complaints in {branch_name} increased by {complaints_delta_pct:.1f}% over 3 months",
                        "The branch may be experiencing visible customer-impacting failures.",
                        f"{previous['complaints']:.0f} to {current['complaints']:.0f} complaints",
                        current["complaints"],
                        previous["complaints"],
                    )
                )

            satisfaction_delta = current["customer_satisfaction"] - previous["customer_satisfaction"]
            if satisfaction_delta <= -0.35:
                findings.append(
                    _finding(
                        branch_id,
                        branch_name,
                        month,
                        "customer_satisfaction",
                        "warning",
                        f"Customer satisfaction in {branch_name} dropped by {abs(satisfaction_delta):.2f} points over 3 months",
                        "Customer sentiment is moving in the same direction as operational service KPIs.",
                        f"{previous['customer_satisfaction']:.2f} to {current['customer_satisfaction']:.2f}",
                        current["customer_satisfaction"],
                        previous["customer_satisfaction"],
                    )
                )

    return sorted(findings, key=lambda item: (_severity_rank(item.severity), item.month), reverse=True)


def latest_branch_snapshot(monthly_kpis: pd.DataFrame) -> pd.DataFrame:
    df = monthly_kpis.sort_values("month").copy()
    df["cost_per_shipment"] = df["transportation_costs"] / df["shipment_volume"]
    return df.groupby("branch_name", as_index=False).tail(1).sort_values("service_level")


def findings_to_frame(findings: list[Finding]) -> pd.DataFrame:
    return pd.DataFrame([finding.to_dict() for finding in findings])


def _finding(
    branch_id: str,
    branch_name: str,
    month: str,
    kpi: str,
    severity: Severity,
    title: str,
    description: str,
    evidence: str,
    value: float,
    previous_value: float,
) -> Finding:
    change_abs = float(value - previous_value)
    return Finding(
        finding_id=f"{branch_id}-{month}-{kpi}",
        branch_id=branch_id,
        branch_name=branch_name,
        month=month,
        kpi=kpi,
        severity=severity,
        title=title,
        description=description,
        evidence=evidence,
        value=float(value),
        previous_value=float(previous_value),
        change_abs=change_abs,
        change_pct=_pct_change(previous_value, value),
    )


def _pct_change(previous: float, current: float) -> float:
    if previous == 0:
        return 0.0
    return float((current - previous) / previous * 100)


def _severity_rank(severity: str) -> int:
    return {"critical": 3, "warning": 2, "info": 1}.get(severity, 0)
