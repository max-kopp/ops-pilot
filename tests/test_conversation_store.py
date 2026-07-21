from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest

from app.conversation_history import create_conversation, set_generated_title, update_conversation
from database.conversation_store import (
    delete_conversation,
    initialize_conversation_database,
    load_conversations,
    save_conversation,
)


class ConversationStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_directory = TemporaryDirectory()
        self.db_path = Path(self.temp_directory.name) / "conversations.db"

    def tearDown(self) -> None:
        self.temp_directory.cleanup()

    def test_save_and_load_conversation_with_messages(self) -> None:
        conversation = create_conversation(
            [
                {"role": "user", "content": "What happened in Hamburg?"},
                {
                    "role": "assistant",
                    "content": "Service levels declined.",
                    "context": "Intent: root_cause\nBranches: Hamburg",
                },
            ],
            conversation_id="conversation-1",
            modified_at=datetime(2026, 7, 21, 12, 0, tzinfo=timezone.utc),
        )
        conversation = set_generated_title(conversation, "Hamburg Service Decline")

        save_conversation(conversation, self.db_path)

        self.assertEqual(load_conversations(self.db_path), [conversation])

    def test_save_updates_existing_record_without_duplicating_messages(self) -> None:
        conversation = create_conversation(
            [{"role": "user", "content": "Review Berlin"}],
            conversation_id="conversation-1",
        )
        save_conversation(conversation, self.db_path)
        updated = update_conversation(
            conversation,
            conversation["messages"] + [{"role": "assistant", "content": "Staffing is constrained."}],
        )

        save_conversation(updated, self.db_path)

        loaded = load_conversations(self.db_path)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0]["messages"], updated["messages"])

    def test_delete_removes_conversation_and_its_messages(self) -> None:
        conversation = create_conversation(
            [{"role": "user", "content": "Review Munich"}],
            conversation_id="conversation-1",
        )
        save_conversation(conversation, self.db_path)

        delete_conversation(conversation["id"], self.db_path)

        self.assertEqual(load_conversations(self.db_path), [])

    def test_existing_database_is_migrated_to_store_context(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(
                """
                CREATE TABLE conversations (
                    id TEXT PRIMARY KEY, title TEXT NOT NULL,
                    title_generated INTEGER NOT NULL, updated_at TEXT NOT NULL
                );
                CREATE TABLE conversation_messages (
                    id INTEGER PRIMARY KEY, conversation_id TEXT NOT NULL,
                    position INTEGER NOT NULL, role TEXT NOT NULL, content TEXT NOT NULL
                );
                """
            )

        initialize_conversation_database(self.db_path)

        with sqlite3.connect(self.db_path) as conn:
            columns = {row[1] for row in conn.execute("PRAGMA table_info(conversation_messages)")}
        self.assertIn("context", columns)


if __name__ == "__main__":
    unittest.main()
