from __future__ import annotations

from pathlib import Path
import sqlite3

from data.generate_demo_data import generate_demo_data
from database.connection import DEFAULT_DB_PATH


SCHEMA_SQL = """
DROP TABLE IF EXISTS monthly_kpis;
DROP TABLE IF EXISTS shipment_details;
DROP TABLE IF EXISTS delay_reasons;
DROP TABLE IF EXISTS staffing_events;
DROP TABLE IF EXISTS customer_feedback;

CREATE TABLE monthly_kpis (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    branch_id TEXT NOT NULL,
    branch_name TEXT NOT NULL,
    month TEXT NOT NULL,
    service_level REAL NOT NULL,
    delayed_shipments INTEGER NOT NULL,
    complaints INTEGER NOT NULL,
    staffing_level INTEGER NOT NULL,
    transportation_costs REAL NOT NULL,
    shipment_volume INTEGER NOT NULL,
    damage_rate REAL NOT NULL,
    overtime_hours REAL NOT NULL,
    customer_satisfaction REAL NOT NULL,
    UNIQUE(branch_id, month)
);

CREATE TABLE shipment_details (
    shipment_id TEXT PRIMARY KEY,
    branch_id TEXT NOT NULL,
    branch_name TEXT NOT NULL,
    shipment_date TEXT NOT NULL,
    month TEXT NOT NULL,
    status TEXT NOT NULL,
    delay_hours REAL NOT NULL,
    transportation_cost REAL NOT NULL,
    damage_flag INTEGER NOT NULL
);

CREATE TABLE delay_reasons (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    branch_id TEXT NOT NULL,
    branch_name TEXT NOT NULL,
    month TEXT NOT NULL,
    reason TEXT NOT NULL,
    delayed_shipments INTEGER NOT NULL,
    avg_delay_hours REAL NOT NULL
);

CREATE TABLE staffing_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    branch_id TEXT NOT NULL,
    branch_name TEXT NOT NULL,
    month TEXT NOT NULL,
    event_type TEXT NOT NULL,
    headcount_delta INTEGER NOT NULL,
    notes TEXT NOT NULL
);

CREATE TABLE customer_feedback (
    feedback_id TEXT PRIMARY KEY,
    branch_id TEXT NOT NULL,
    branch_name TEXT NOT NULL,
    month TEXT NOT NULL,
    sentiment TEXT NOT NULL,
    category TEXT NOT NULL,
    rating INTEGER NOT NULL,
    comment TEXT NOT NULL
);
"""


def create_database(db_path: Path | str = DEFAULT_DB_PATH, force: bool = False) -> Path:
    """Create and populate the local SQLite demo database."""
    db_path = Path(db_path)
    if db_path.exists() and not force:
        return db_path

    db_path.parent.mkdir(parents=True, exist_ok=True)
    monthly, shipments, delays, staffing, feedback = generate_demo_data()

    with sqlite3.connect(db_path) as conn:
        conn.executescript(SCHEMA_SQL)
        monthly.to_sql("monthly_kpis", conn, if_exists="append", index=False)
        shipments.to_sql("shipment_details", conn, if_exists="append", index=False)
        delays.to_sql("delay_reasons", conn, if_exists="append", index=False)
        staffing.to_sql("staffing_events", conn, if_exists="append", index=False)
        feedback.to_sql("customer_feedback", conn, if_exists="append", index=False)

    return db_path


if __name__ == "__main__":
    path = create_database(force=True)
    print(f"Created demo database at {path}")

