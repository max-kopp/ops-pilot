from __future__ import annotations

from datetime import datetime
import re
from typing import TypedDict
from uuid import uuid4


DEFAULT_CONVERSATION_TITLE = "New conversation"
MAX_TITLE_LENGTH = 58


class ChatMessage(TypedDict):
    role: str
    content: str


class Conversation(TypedDict):
    id: str
    title: str
    title_generated: bool
    updated_at: str
    messages: list[ChatMessage]


def create_conversation(
    messages: list[ChatMessage],
    *,
    conversation_id: str | None = None,
    modified_at: datetime | None = None,
) -> Conversation:
    """Create a conversation record suitable for Streamlit session state."""
    return {
        "id": conversation_id or str(uuid4()),
        "title": conversation_title(messages),
        "title_generated": False,
        "updated_at": _timestamp(modified_at),
        "messages": [message.copy() for message in messages],
    }


def update_conversation(
    conversation: Conversation,
    messages: list[ChatMessage],
    *,
    modified_at: datetime | None = None,
) -> Conversation:
    """Return a refreshed record after messages have changed."""
    updated = create_conversation(
        messages,
        conversation_id=conversation["id"],
        modified_at=modified_at,
    )
    updated["title_generated"] = conversation.get("title_generated", False)
    if updated["title_generated"]:
        updated["title"] = conversation["title"]
    return updated


def set_generated_title(conversation: Conversation, title: str) -> Conversation:
    """Apply a model-generated title without changing the modification date."""
    updated = conversation.copy()
    updated["title"] = normalize_title(title) or conversation["title"]
    updated["title_generated"] = True
    return updated


def conversation_title(messages: list[ChatMessage]) -> str:
    """Build a concise title from the first meaningful user message."""
    first_question = next(
        (
            re.sub(r"\s+", " ", message["content"]).strip()
            for message in messages
            if message.get("role") == "user" and message.get("content", "").strip()
        ),
        "",
    )
    if not first_question:
        return DEFAULT_CONVERSATION_TITLE
    return normalize_title(first_question)


def normalize_title(title: str) -> str:
    normalized = re.sub(r"\s+", " ", title).strip().strip('"\'“”‘’')
    if normalized.lower().startswith("title:"):
        normalized = normalized[6:].strip()
    normalized = normalized.rstrip(".")
    if not normalized:
        return ""
    if len(normalized) <= MAX_TITLE_LENGTH:
        return normalized

    shortened = normalized[: MAX_TITLE_LENGTH - 1].rsplit(" ", 1)[0].rstrip(".,;:-")
    if not shortened:
        shortened = normalized[: MAX_TITLE_LENGTH - 1]
    return f"{shortened}…"


def sorted_conversations(conversations: list[Conversation]) -> list[Conversation]:
    """Order conversations by last modification, newest first."""
    return sorted(
        conversations,
        key=lambda item: datetime.fromisoformat(item["updated_at"]).timestamp(),
        reverse=True,
    )


def conversation_label(conversation: Conversation) -> str:
    modified = datetime.fromisoformat(conversation["updated_at"])
    return f"{modified.strftime('%d %b %Y, %H:%M')} · {conversation['title']}"


def has_user_message(conversation: Conversation) -> bool:
    return any(
        message.get("role") == "user" and message.get("content", "").strip()
        for message in conversation["messages"]
    )


def _timestamp(value: datetime | None) -> str:
    current = value or datetime.now().astimezone()
    if current.tzinfo is None:
        current = current.astimezone()
    return current.isoformat(timespec="microseconds")
