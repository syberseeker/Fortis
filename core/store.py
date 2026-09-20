"""
Lightweight persistence for the client/engagement concept. Deliberately
plain sqlite3 (no ORM) â€” this is a single-user-ish local tool and the schema
is small and stable.

Concepts:
- Client: an organization/person you're advising (e.g. "Acme Corp").
- Engagement: a bounded piece of work for a client (e.g. "Q3 2026 Config
  Review"). Documents, chat history context, and reports are scoped to an
  engagement, not to a chat window â€” so switching Open WebUI chats, or
  coming back a week later, doesn't lose your uploaded material.
- active_engagement: which engagement a given Open WebUI user currently has
  selected, so the pipe doesn't need to ask every turn.
"""
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import List, Dict, Optional

from .config import settings

_SCHEMA = """
CREATE TABLE IF NOT EXISTS clients (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS engagements (
    id TEXT PRIMARY KEY,
    client_id TEXT NOT NULL REFERENCES clients(id),
    name TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    notes TEXT DEFAULT '',
    created_at TEXT NOT NULL,
    UNIQUE(client_id, name)
);

CREATE TABLE IF NOT EXISTS active_engagement (
    user_id TEXT PRIMARY KEY,
    engagement_id TEXT NOT NULL REFERENCES engagements(id)
);
"""


@contextmanager
def _conn():
    conn = sqlite3.connect(settings.db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with _conn() as conn:
        conn.executescript(_SCHEMA)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slugify(text: str, max_len: int = 40) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:max_len].strip("-") or "item"


# ---- Clients ----------------------------------------------------------

def get_or_create_client(name: str) -> Dict:
    name = name.strip()
    with _conn() as conn:
        row = conn.execute("SELECT * FROM clients WHERE name = ?", (name,)).fetchone()
        if row:
            return dict(row)

        base_slug = _slugify(name)
        client_id = base_slug
        suffix = 2
        while conn.execute("SELECT 1 FROM clients WHERE id = ?", (client_id,)).fetchone():
            client_id = f"{base_slug}-{suffix}"
            suffix += 1

        conn.execute(
            "INSERT INTO clients (id, name, created_at) VALUES (?, ?, ?)",
            (client_id, name, _now()),
        )
        return {"id": client_id, "name": name, "created_at": _now()}


def list_clients() -> List[Dict]:
    with _conn() as conn:
        rows = conn.execute("SELECT * FROM clients ORDER BY name").fetchall()
        return [dict(r) for r in rows]


# ---- Engagements --------------------------------------------------------

def create_engagement(client_name: str, engagement_name: str, notes: str = "") -> Dict:
    client = get_or_create_client(client_name)
    engagement_name = engagement_name.strip()
    with _conn() as conn:
        existing = conn.execute(
            "SELECT * FROM engagements WHERE client_id = ? AND name = ?",
            (client["id"], engagement_name),
        ).fetchone()
        if existing:
            return dict(existing)

        base_id = f"{client['id']}-{_slugify(engagement_name)}"
        eng_id = base_id
        suffix = 2
        while conn.execute("SELECT 1 FROM engagements WHERE id = ?", (eng_id,)).fetchone():
            eng_id = f"{base_id}-{suffix}"
            suffix += 1

        conn.execute(
            "INSERT INTO engagements (id, client_id, name, status, notes, created_at) "
            "VALUES (?, ?, ?, 'active', ?, ?)",
            (eng_id, client["id"], engagement_name, notes, _now()),
        )
        return {
            "id": eng_id, "client_id": client["id"], "name": engagement_name,
            "status": "active", "notes": notes, "created_at": _now(),
        }


def list_engagements(client_id: Optional[str] = None) -> List[Dict]:
    with _conn() as conn:
        if client_id:
            rows = conn.execute(
                "SELECT e.*, c.name AS client_name FROM engagements e "
                "JOIN clients c ON c.id = e.client_id WHERE e.client_id = ? "
                "ORDER BY e.created_at DESC",
                (client_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT e.*, c.name AS client_name FROM engagements e "
                "JOIN clients c ON c.id = e.client_id ORDER BY e.created_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]


def get_engagement(engagement_id: str) -> Optional[Dict]:
    with _conn() as conn:
        row = conn.execute(
            "SELECT e.*, c.name AS client_name FROM engagements e "
            "JOIN clients c ON c.id = e.client_id WHERE e.id = ?",
            (engagement_id,),
        ).fetchone()
        return dict(row) if row else None


def find_engagement_by_names(client_name: str, engagement_name: str) -> Optional[Dict]:
    with _conn() as conn:
        row = conn.execute(
            "SELECT e.*, c.name AS client_name FROM engagements e "
            "JOIN clients c ON c.id = e.client_id "
            "WHERE c.name = ? AND e.name = ?",
            (client_name.strip(), engagement_name.strip()),
        ).fetchone()
        return dict(row) if row else None


def set_engagement_status(engagement_id: str, status: str) -> None:
    with _conn() as conn:
        conn.execute("UPDATE engagements SET status = ? WHERE id = ?", (status, engagement_id))


def delete_engagement(engagement_id: str) -> None:
    with _conn() as conn:
        conn.execute("DELETE FROM active_engagement WHERE engagement_id = ?", (engagement_id,))
        conn.execute("DELETE FROM engagements WHERE id = ?", (engagement_id,))


# ---- Active engagement per user -----------------------------------------

def set_active_engagement(user_id: str, engagement_id: str) -> None:
    with _conn() as conn:
        conn.execute(
            "INSERT INTO active_engagement (user_id, engagement_id) VALUES (?, ?) "
            "ON CONFLICT(user_id) DO UPDATE SET engagement_id = excluded.engagement_id",
            (user_id, engagement_id),
        )


def get_active_engagement(user_id: str) -> Optional[Dict]:
    with _conn() as conn:
        row = conn.execute(
            "SELECT engagement_id FROM active_engagement WHERE user_id = ?", (user_id,)
        ).fetchone()
        if not row:
            return None
        return get_engagement(row["engagement_id"])


def clear_active_engagement(user_id: str) -> None:
    with _conn() as conn:
        conn.execute("DELETE FROM active_engagement WHERE user_id = ?", (user_id,))
