"""
User accounts and multi-thread persistence.

Two new tables, deliberately separate from LangGraph's own checkpoint
tables (which it manages internally via PostgresSaver.setup() in
app/agents/graph.py — this module never touches those):

  app_users   — username + bcrypt password hash
  app_threads — maps a LangGraph thread_id to the user who owns it, plus
                a title and recency timestamp for the UI's thread list

LangGraph's Postgres checkpointer already persists full conversation state
per thread_id — that part was never the gap. The gap was that nothing
tracked *which threads belong to which user*, and nothing let the UI list
or reload an existing thread's history. This module is exactly that layer,
nothing more — it does not duplicate or wrap the checkpointer itself.
"""

import os
import uuid

import bcrypt
import psycopg
from psycopg_pool import ConnectionPool

from app.config import settings

_pool: ConnectionPool | None = None


def _get_pool() -> ConnectionPool:
    """Lazy-init, matching the lazy-loading convention used throughout
    app/services/ — no connection attempt at import time."""
    global _pool
    if _pool is None:
        _pool = ConnectionPool(
            conninfo=settings.POSTGRES_URI,
            max_size=5,
            kwargs={"autocommit": True},
        )
    return _pool


def init_auth_tables() -> None:
    """One-time DDL — mirrors the exact pattern PostgresSaver.setup() uses
    for LangGraph's own tables in app/agents/graph.py. Safe to call on
    every startup; CREATE TABLE IF NOT EXISTS is a no-op once the tables
    already exist."""
    with _get_pool().connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS app_users (
                id SERIAL PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS app_threads (
                thread_id TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES app_users(id) ON DELETE CASCADE,
                title TEXT,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_app_threads_user_id ON app_threads(user_id)")


class UsernameTakenError(Exception):
    pass


class InvalidCredentialsError(Exception):
    pass


def signup(username: str, password: str) -> int:
    """Creates a new user. Raises UsernameTakenError or ValueError — the
    caller (main.py) maps these to real HTTP status codes."""
    username = (username or "").strip()
    if not username or not password:
        raise ValueError("Username and password are required.")

    password_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    try:
        with _get_pool().connection() as conn:
            row = conn.execute(
                "INSERT INTO app_users (username, password_hash) VALUES (%s, %s) RETURNING id",
                (username, password_hash),
            ).fetchone()
            return row[0]
    except psycopg.errors.UniqueViolation:
        raise UsernameTakenError(f"Username '{username}' is already taken.")


def login(username: str, password: str) -> int:
    """Verifies credentials, returns the user's id. Raises
    InvalidCredentialsError on any mismatch — deliberately the same error
    for 'no such user' and 'wrong password' so the response never leaks
    which username exists."""
    with _get_pool().connection() as conn:
        row = conn.execute(
            "SELECT id, password_hash FROM app_users WHERE username = %s",
            ((username or "").strip(),),
        ).fetchone()

    if not row:
        raise InvalidCredentialsError("Invalid username or password.")

    user_id, password_hash = row
    if not bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8")):
        raise InvalidCredentialsError("Invalid username or password.")
    return user_id


def create_thread(user_id: int, title: str | None = None) -> str:
    """Registers a brand-new thread_id for this user. The thread_id itself
    is just a fresh UUID — LangGraph's checkpointer creates its own state
    for it lazily on the first /query call, exactly as it already does for
    any new thread_id today."""
    thread_id = str(uuid.uuid4())
    with _get_pool().connection() as conn:
        conn.execute(
            "INSERT INTO app_threads (thread_id, user_id, title) VALUES (%s, %s, %s)",
            (thread_id, user_id, title),
        )
    return thread_id


def list_threads(user_id: int) -> list[dict]:
    """All threads for a user, most recently active first."""
    with _get_pool().connection() as conn:
        rows = conn.execute(
            "SELECT thread_id, title, created_at, updated_at FROM app_threads "
            "WHERE user_id = %s ORDER BY updated_at DESC",
            (user_id,),
        ).fetchall()
    return [
        {
            "thread_id": r[0],
            "title": r[1],
            "created_at": r[2].isoformat(),
            "updated_at": r[3].isoformat(),
        }
        for r in rows
    ]


def thread_owner(thread_id: str) -> int | None:
    """The user_id that owns this thread, or None if it doesn't exist.
    Used by main.py's /query to reject a thread_id that wasn't issued to
    the caller — otherwise any authenticated user could read any other
    user's conversation just by guessing/reusing a thread_id."""
    with _get_pool().connection() as conn:
        row = conn.execute(
            "SELECT user_id FROM app_threads WHERE thread_id = %s", (thread_id,)
        ).fetchone()
    return row[0] if row else None


def touch_thread(thread_id: str, title_if_untitled: str | None = None) -> None:
    """Bumps updated_at (recency sort in the thread list) and, if the
    thread has no title yet, sets one from the first message — called
    after every /query so a thread the UI created without a title picks
    one up automatically once the user actually says something."""
    with _get_pool().connection() as conn:
        if title_if_untitled:
            conn.execute(
                "UPDATE app_threads SET updated_at = now(), "
                "title = COALESCE(title, %s) WHERE thread_id = %s",
                (title_if_untitled[:80], thread_id),
            )
        else:
            conn.execute(
                "UPDATE app_threads SET updated_at = now() WHERE thread_id = %s",
                (thread_id,),
            )
