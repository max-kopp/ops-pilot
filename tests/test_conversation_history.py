from __future__ import annotations

from datetime import datetime, timezone
import unittest

from app.conversation_history import (
    conversation_label,
    conversation_title,
    create_conversation,
    set_generated_title,
    sorted_conversations,
    update_conversation,
)


WELCOME = [{"role": "assistant", "content": "How can I help?"}]


class ConversationHistoryTests(unittest.TestCase):
    def test_title_uses_first_user_question_and_normalizes_whitespace(self) -> None:
        messages = WELCOME + [
            {"role": "user", "content": "  Why did service quality\n decrease in Hamburg?  "},
            {"role": "assistant", "content": "I found a decline."},
        ]

        self.assertEqual(conversation_title(messages), "Why did service quality decrease in Hamburg?")

    def test_long_title_is_shortened(self) -> None:
        title = conversation_title(
            WELCOME + [{"role": "user", "content": "Explain every transportation cost driver affecting Munich this quarter in detail"}]
        )

        self.assertLessEqual(len(title), 58)
        self.assertTrue(title.endswith("…"))

    def test_update_refreshes_title_and_timestamp(self) -> None:
        original = create_conversation(
            WELCOME,
            conversation_id="conversation-1",
            modified_at=datetime(2026, 7, 20, 9, 0, tzinfo=timezone.utc),
        )
        updated = update_conversation(
            original,
            WELCOME + [{"role": "user", "content": "Which branches are critical?"}],
            modified_at=datetime(2026, 7, 21, 10, 30, tzinfo=timezone.utc),
        )

        self.assertEqual(updated["id"], original["id"])
        self.assertEqual(updated["title"], "Which branches are critical?")
        self.assertGreater(updated["updated_at"], original["updated_at"])

    def test_generated_title_is_normalized_and_preserved_on_updates(self) -> None:
        conversation = create_conversation(
            WELCOME + [{"role": "user", "content": "Tell me what happened in Hamburg"}]
        )
        titled = set_generated_title(conversation, '"Title: Hamburg Service Decline."')
        updated = update_conversation(
            titled,
            titled["messages"] + [{"role": "assistant", "content": "Service levels fell."}],
        )

        self.assertEqual(updated["title"], "Hamburg Service Decline")
        self.assertTrue(updated["title_generated"])

    def test_conversations_are_sorted_newest_first(self) -> None:
        older = create_conversation(
            WELCOME,
            conversation_id="older",
            modified_at=datetime(2026, 7, 20, tzinfo=timezone.utc),
        )
        newer = create_conversation(
            WELCOME,
            conversation_id="newer",
            modified_at=datetime(2026, 7, 21, tzinfo=timezone.utc),
        )

        self.assertEqual([item["id"] for item in sorted_conversations([older, newer])], ["newer", "older"])

    def test_dropdown_label_contains_modified_date_and_title(self) -> None:
        conversation = create_conversation(
            WELCOME + [{"role": "user", "content": "Review Hamburg"}],
            modified_at=datetime(2026, 7, 21, 14, 5, tzinfo=timezone.utc),
        )

        self.assertEqual(conversation_label(conversation), "21 Jul 2026, 14:05 · Review Hamburg")


if __name__ == "__main__":
    unittest.main()
