from __future__ import annotations

from datetime import datetime
import random
from typing import Any

import pandas as pd


BRANCHES = [
    ("B001", "Hamburg"),
    ("B002", "Munich"),
    ("B003", "Berlin"),
    ("B004", "Cologne"),
    ("B005", "Frankfurt"),
    ("B006", "Stuttgart"),
    ("B007", "Dortmund"),
    ("B008", "Leipzig"),
]


def _months() -> list[str]:
    return pd.date_range("2025-06-01", periods=12, freq="MS").strftime("%Y-%m").tolist()


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def generate_demo_data(seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate deterministic KPI and detail data with realistic branch-specific anomalies."""
    random.seed(seed)
    months = _months()

    monthly_rows: list[dict[str, Any]] = []
    shipment_rows: list[dict[str, Any]] = []
    delay_rows: list[dict[str, Any]] = []
    staffing_rows: list[dict[str, Any]] = []
    feedback_rows: list[dict[str, Any]] = []

    base_profiles = {
        "Hamburg": {"service": 96.5, "cost": 28.5, "staff": 46, "volume": 4200},
        "Munich": {"service": 95.8, "cost": 31.0, "staff": 42, "volume": 3900},
        "Berlin": {"service": 95.0, "cost": 27.5, "staff": 44, "volume": 4400},
        "Cologne": {"service": 96.0, "cost": 26.8, "staff": 38, "volume": 3300},
        "Frankfurt": {"service": 94.8, "cost": 33.0, "staff": 48, "volume": 4700},
        "Stuttgart": {"service": 95.5, "cost": 29.5, "staff": 36, "volume": 3100},
        "Dortmund": {"service": 96.2, "cost": 25.5, "staff": 34, "volume": 2800},
        "Leipzig": {"service": 95.4, "cost": 24.8, "staff": 32, "volume": 2600},
    }

    for branch_id, branch_name in BRANCHES:
        profile = base_profiles[branch_name]
        for idx, month in enumerate(months):
            service = profile["service"] + random.uniform(-0.8, 0.7)
            cost_per_shipment = profile["cost"] + random.uniform(-1.0, 1.1)
            staffing = profile["staff"] + random.choice([-1, 0, 0, 1])
            volume = int(profile["volume"] * (1 + 0.04 * random.uniform(-1, 1)))
            overtime = 70 + random.uniform(-18, 22)
            damage_rate = 0.8 + random.uniform(-0.2, 0.25)

            if branch_name == "Hamburg" and idx >= 5:
                service -= (idx - 4) * 1.35
                overtime += (idx - 4) * 14
                damage_rate += (idx - 4) * 0.08

            if branch_name == "Munich" and idx >= 4:
                cost_per_shipment += (idx - 3) * 1.75

            if branch_name == "Berlin" and idx >= 6:
                staffing -= idx - 5
                overtime += (idx - 5) * 20
                service -= (idx - 5) * 0.85

            if branch_name == "Frankfurt" and idx == 8:
                service -= 4.8
                damage_rate += 1.25
                overtime += 55

            service = round(_clamp(service, 82, 99), 1)
            staffing = int(_clamp(staffing, 24, 55))
            damage_rate = round(_clamp(damage_rate, 0.3, 4.5), 2)
            delayed_shipments = int(volume * (100 - service) / 100)
            complaints = int(delayed_shipments * random.uniform(0.10, 0.16) + damage_rate * 7)
            customer_satisfaction = round(_clamp(4.8 - (100 - service) * 0.08 - complaints / max(volume, 1) * 28, 2.6, 4.9), 2)
            transportation_costs = round(volume * cost_per_shipment, 2)

            monthly_rows.append(
                {
                    "branch_id": branch_id,
                    "branch_name": branch_name,
                    "month": month,
                    "service_level": service,
                    "delayed_shipments": delayed_shipments,
                    "complaints": complaints,
                    "staffing_level": staffing,
                    "transportation_costs": transportation_costs,
                    "shipment_volume": volume,
                    "damage_rate": damage_rate,
                    "overtime_hours": round(overtime, 1),
                    "customer_satisfaction": customer_satisfaction,
                }
            )

            _add_detail_rows(
                shipment_rows,
                delay_rows,
                staffing_rows,
                feedback_rows,
                branch_id,
                branch_name,
                month,
                idx,
                service,
                delayed_shipments,
                cost_per_shipment,
                damage_rate,
                staffing,
                customer_satisfaction,
            )

    return (
        pd.DataFrame(monthly_rows),
        pd.DataFrame(shipment_rows),
        pd.DataFrame(delay_rows),
        pd.DataFrame(staffing_rows),
        pd.DataFrame(feedback_rows),
    )


def _add_detail_rows(
    shipment_rows: list[dict[str, Any]],
    delay_rows: list[dict[str, Any]],
    staffing_rows: list[dict[str, Any]],
    feedback_rows: list[dict[str, Any]],
    branch_id: str,
    branch_name: str,
    month: str,
    month_idx: int,
    service_level: float,
    delayed_shipments: int,
    cost_per_shipment: float,
    damage_rate: float,
    staffing_level: int,
    customer_satisfaction: float,
) -> None:
    delay_mix = {
        "Carrier capacity": 0.28,
        "Warehouse backlog": 0.22,
        "Late inbound freight": 0.20,
        "Route disruption": 0.18,
        "System exception": 0.12,
    }

    if branch_name == "Hamburg" and month_idx >= 6:
        delay_mix["Warehouse backlog"] += 0.18
        delay_mix["Carrier capacity"] += 0.07
    if branch_name == "Berlin" and month_idx >= 7:
        delay_mix["Warehouse backlog"] += 0.25
    if branch_name == "Munich" and month_idx >= 5:
        delay_mix["Route disruption"] += 0.20

    total_weight = sum(delay_mix.values())
    for reason, weight in delay_mix.items():
        count = int(delayed_shipments * weight / total_weight)
        delay_rows.append(
            {
                "branch_id": branch_id,
                "branch_name": branch_name,
                "month": month,
                "reason": reason,
                "delayed_shipments": count,
                "avg_delay_hours": round(random.uniform(4, 18) + max(0, 94 - service_level) * 0.7, 1),
            }
        )

    month_start = datetime.strptime(f"{month}-01", "%Y-%m-%d")
    sampled_shipments = 30
    delay_probability = _clamp((100 - service_level) / 100, 0.02, 0.22)
    for shipment_idx in range(sampled_shipments):
        delayed = random.random() < delay_probability
        damaged = random.random() < damage_rate / 100
        shipment_rows.append(
            {
                "shipment_id": f"{branch_id}-{month}-{shipment_idx:03d}",
                "branch_id": branch_id,
                "branch_name": branch_name,
                "shipment_date": (month_start + pd.Timedelta(days=random.randint(0, 27))).strftime("%Y-%m-%d"),
                "month": month,
                "status": "Delayed" if delayed else "On time",
                "delay_hours": round(random.uniform(2, 36), 1) if delayed else 0.0,
                "transportation_cost": round(random.uniform(0.82, 1.24) * cost_per_shipment, 2),
                "damage_flag": int(damaged),
            }
        )

    if branch_name == "Berlin" and month_idx in {6, 8, 10}:
        staffing_rows.append(
            {
                "branch_id": branch_id,
                "branch_name": branch_name,
                "month": month,
                "event_type": "Staff shortage",
                "headcount_delta": -2,
                "notes": "Temporary warehouse vacancies increased picking and loading pressure.",
            }
        )
    elif branch_name == "Hamburg" and month_idx in {6, 9}:
        staffing_rows.append(
            {
                "branch_id": branch_id,
                "branch_name": branch_name,
                "month": month,
                "event_type": "Process change",
                "headcount_delta": 0,
                "notes": "New cross-dock process required additional training and slowed throughput.",
            }
        )
    else:
        staffing_rows.append(
            {
                "branch_id": branch_id,
                "branch_name": branch_name,
                "month": month,
                "event_type": "Normal staffing",
                "headcount_delta": random.choice([-1, 0, 0, 1]),
                "notes": f"Operational staffing remained near plan at {staffing_level} FTE.",
            }
        )

    feedback_templates = [
        ("Delivery reliability", "Customers mention missed delivery windows."),
        ("Shipment condition", "Some packages arrived with visible handling damage."),
        ("Communication", "Customers requested earlier status updates."),
        ("Speed", "Positive comments on fast regional deliveries."),
    ]
    for feedback_idx in range(4):
        category, comment = random.choice(feedback_templates)
        rating = int(round(_clamp(customer_satisfaction + random.uniform(-0.8, 0.5), 1, 5)))
        sentiment = "positive" if rating >= 4 else "neutral" if rating == 3 else "negative"
        feedback_rows.append(
            {
                "feedback_id": f"FB-{branch_id}-{month}-{feedback_idx}",
                "branch_id": branch_id,
                "branch_name": branch_name,
                "month": month,
                "sentiment": sentiment,
                "category": category,
                "rating": rating,
                "comment": comment,
            }
        )
