from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from analysis.kpi_analysis import analyze_kpis, load_monthly_kpis
from analysis.retrieval import RetrievalQuery, retrieve_context
from database.setup_database import create_database


class RetrievalQueryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.db_path = Path(cls.temp_dir.name) / "opspilot.db"
        create_database(cls.db_path, force=True)
        with sqlite3.connect(cls.db_path) as conn:
            cls.findings = analyze_kpis(load_monthly_kpis(conn))

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp_dir.cleanup()

    def context_for(self, question: str, extracted_query: RetrievalQuery):
        with patch("analysis.retrieval.extract_retrieval_query", return_value=extracted_query):
            with sqlite3.connect(self.db_path) as conn:
                return retrieve_context(conn, question, self.findings)

    def test_service_question_uses_extracted_branch_and_kpi(self) -> None:
        context = self.context_for(
            "Why did service quality decrease in Hamburg?",
            RetrievalQuery(
                original_question="model paraphrase",
                intent="root_cause",
                branches=["hamburg"],
                kpis=["service_level"],
                comparison_mode="root_cause",
            ),
        )

        self.assertEqual(context["intent"], "root_cause")
        self.assertEqual(context["branches"], ["Hamburg"])
        self.assertEqual(context["retrieval_query"]["original_question"], "Why did service quality decrease in Hamburg?")
        self.assertEqual(context["retrieval_query"]["kpis"], ["service_level"])
        self.assertFalse(context["monthly_kpis"].empty)
        self.assertEqual(set(context["monthly_kpis"]["branch_name"]), {"Hamburg"})
        self.assertTrue(all(finding["kpi"] == "service_level" for finding in context["findings"]))

    def test_cost_driver_question_retrieves_cost_sample(self) -> None:
        context = self.context_for(
            "What drives transportation costs in Munich?",
            RetrievalQuery(
                original_question="What drives transportation costs in Munich?",
                intent="cost_drivers",
                branches=["Munich"],
                kpis=["transportation_costs"],
                comparison_mode="drivers",
            ),
        )

        self.assertIn("shipment_cost_sample", context)
        self.assertFalse(context["shipment_cost_sample"].empty)
        self.assertEqual(set(context["shipment_cost_sample"]["branch_name"]), {"Munich"})

    def test_critical_branch_question_infers_critical_branches(self) -> None:
        context = self.context_for(
            "Which branches are currently critical?",
            RetrievalQuery(
                original_question="Which branches are currently critical?",
                intent="critical_branches",
                branches=[],
                kpis=[],
                comparison_mode="critical",
            ),
        )

        self.assertEqual(context["intent"], "critical_branches")
        self.assertTrue(context["branches"])
        self.assertIn("monthly_kpis", context)

    def test_customer_satisfaction_question_retrieves_feedback_context(self) -> None:
        context = self.context_for(
            "Why was customer satisfaction bad in Hamburg?",
            RetrievalQuery(
                original_question="Why was customer satisfaction bad in Hamburg?",
                intent="customer_satisfaction_drivers",
                branches=["Hamburg"],
                kpis=["customer_satisfaction"],
                comparison_mode="drivers",
            ),
        )

        self.assertIn("customer_satisfaction_trend", context)
        self.assertIn("feedback_summary", context)
        self.assertIn("low_rating_feedback", context)
        self.assertFalse(context["feedback_summary"].empty)

    def test_explicit_month_filters_branch_queries(self) -> None:
        context = self.context_for(
            "Show complaints in Berlin in 2025-09",
            RetrievalQuery(
                original_question="Show complaints in Berlin in 2025-09",
                intent="general_kpi_question",
                branches=["Berlin"],
                kpis=["complaints"],
                months=["2025-09"],
                comparison_mode="general",
            ),
        )

        self.assertEqual(context["retrieval_query"]["months"], ["2025-09"])
        self.assertFalse(context["monthly_kpis"].empty)
        self.assertEqual(set(context["monthly_kpis"]["month"]), {"2025-09"})
        self.assertTrue(all(finding["month"] == "2025-09" for finding in context["findings"]))

    def test_invalid_extracted_fields_fall_back_to_manual_parsing(self) -> None:
        invalid_query = RetrievalQuery.model_construct(
            original_question="bad extraction",
            intent="root_cause",
            branches=["Vienna"],
            kpis=["profit"],
            months=[],
            time_range=None,
            comparison_mode="general",
        )

        context = self.context_for("Why did service quality decrease in Hamburg?", invalid_query)

        self.assertIn("query_parse_warning", context)
        self.assertEqual(context["branches"], ["Hamburg"])
        self.assertEqual(context["retrieval_query"]["kpis"], ["service_level"])


if __name__ == "__main__":
    unittest.main()
