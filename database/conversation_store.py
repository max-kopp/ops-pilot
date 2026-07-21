from __future__ import annotations

from pathlib import Path
import sqlite3

from app.conversation_history import Conversation


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONVERSATION_DB_PATH = PROJECT_ROOT / "database" / "conversations.db"


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    title_generated INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS conversation_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL,
    position INTEGER NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    context TEXT,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE,
    UNIQUE (conversation_id, position)
);

CREATE INDEX IF NOT EXISTS idx_conversations_updated_at
ON conversations(updated_at DESC);
"""


def initialize_conversation_database(
    db_path: Path | str = DEFAULT_CONVERSATION_DB_PATH,
) -> Path:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _connect(path) as conn:
        conn.executescript(SCHEMA_SQL)
        message_columns = {
            row["name"] for row in conn.execute("PRAGMA table_info(conversation_messages)")
        }
        if "context" not in message_columns:
            conn.execute("ALTER TABLE conversation_messages ADD COLUMN context TEXT")
    return path


def load_conversations(
    db_path: Path | str = DEFAULT_CONVERSATION_DB_PATH,
) -> list[Conversation]:
    initialize_conversation_database(db_path)
    with _connect(db_path) as conn:
        conversation_rows = conn.execute(
            "SELECT id, title, title_generated, updated_at "
            "FROM conversations ORDER BY julianday(updated_at) DESC"
        ).fetchall()
        conversations: list[Conversation] = []
        for row in conversation_rows:
            message_rows = conn.execute(
                "SELECT role, content, context FROM conversation_messages "
                "WHERE conversation_id = ? ORDER BY position",
                (row["id"],),
            ).fetchall()
            conversations.append(
                {
                    "id": row["id"],
                    "title": row["title"],
                    "title_generated": bool(row["title_generated"]),
                    "updated_at": row["updated_at"],
                    "messages": [_message_from_row(message) for message in message_rows],
                }
            )
    return conversations


def save_conversation(
    conversation: Conversation,
    db_path: Path | str = DEFAULT_CONVERSATION_DB_PATH,
) -> None:
    initialize_conversation_database(db_path)
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO conversations (id, title, title_generated, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                title = excluded.title,
                title_generated = excluded.title_generated,
                updated_at = excluded.updated_at
            """,
            (
                conversation["id"],
                conversation["title"],
                int(conversation.get("title_generated", False)),
                conversation["updated_at"],
            ),
        )
        conn.execute(
            "DELETE FROM conversation_messages WHERE conversation_id = ?",
            (conversation["id"],),
        )
        conn.executemany(
            """
            INSERT INTO conversation_messages (conversation_id, position, role, content, context)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                (
                    conversation["id"],
                    position,
                    message["role"],
                    message["content"],
                    message.get("context"),
                )
                for position, message in enumerate(conversation["messages"])
            ],
        )


def delete_conversation(
    conversation_id: str,
    db_path: Path | str = DEFAULT_CONVERSATION_DB_PATH,
) -> None:
    initialize_conversation_database(db_path)
    with _connect(db_path) as conn:
        conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))


def _connect(db_path: Path | str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _message_from_row(row: sqlite3.Row) -> dict[str, str]:
    message = {"role": row["role"], "content": row["content"]}
    if row["context"]:
        message["context"] = row["context"]
    return message
