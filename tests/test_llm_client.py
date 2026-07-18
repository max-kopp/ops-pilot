from __future__ import annotations

import unittest

from langchain_core.messages import AIMessage, HumanMessage

from llm.client import format_chat_history, question_answer_formatter


class LlmClientTests(unittest.TestCase):
    def test_format_chat_history_preserves_visible_user_and_assistant_turns(self) -> None:
        messages = [
            {"role": "assistant", "content": "How can I help?"},
            {"role": "user", "content": "What happened in Hamburg?"},
            {"role": "assistant", "content": "Service level dropped."},
            {"role": "system", "content": "internal note"},
            {"role": "user", "content": ""},
        ]

        chat_history = format_chat_history(messages)

        self.assertEqual(len(chat_history), 3)
        self.assertIsInstance(chat_history[0], AIMessage)
        self.assertEqual(chat_history[0].content, "How can I help?")
        self.assertIsInstance(chat_history[1], HumanMessage)
        self.assertEqual(chat_history[1].content, "What happened in Hamburg?")
        self.assertIsInstance(chat_history[2], AIMessage)
        self.assertEqual(chat_history[2].content, "Service level dropped.")

    def test_question_answer_formatter_includes_question_context_and_chat_history(self) -> None:
        formatted = question_answer_formatter.invoke(
            {
                "question": "Can you compare that to Munich?",
                "retrieved_context": {
                    "intent": "general_kpi_question",
                    "branches": ["Munich"],
                    "findings": [],
                },
                "chat_history": [
                    {"role": "user", "content": "What happened in Hamburg?"},
                    {"role": "assistant", "content": "Service level dropped."},
                ],
            }
        )

        self.assertEqual(formatted["question"], "Can you compare that to Munich?")
        self.assertIn("Intent: general_kpi_question", formatted["context"])
        self.assertIn("Branches: Munich", formatted["context"])
        self.assertEqual(len(formatted["chat_history"]), 2)
        self.assertIsInstance(formatted["chat_history"][0], HumanMessage)
        self.assertIsInstance(formatted["chat_history"][1], AIMessage)


if __name__ == "__main__":
    unittest.main()
