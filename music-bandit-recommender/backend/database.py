"""
SQLite database layer for BanditBeats.
Uses standard sqlite3 (synchronous) wrapped in asyncio.to_thread for async compat.
Tables: users, interactions, bandit_metrics, song_stats
"""
import json
import time
import sqlite3
import asyncio
import threading
from datetime import datetime
from typing import List, Optional, Dict, Any

import numpy as np

from music_catalog import SONGS_LIST, get_song_feature_vector, SONGS_DB

DB_PATH = "/tmp/banditbeats_v2.db"

_db_lock = threading.Lock()


# ─────────────────────────────────────────────────────────────────────────────
# Schema
# ─────────────────────────────────────────────────────────────────────────────

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT    UNIQUE NOT NULL,
    created_at  REAL    NOT NULL,
    user_embedding TEXT,
    taste_profile  TEXT
);

CREATE TABLE IF NOT EXISTS interactions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id         INTEGER NOT NULL,
    song_id         TEXT    NOT NULL,
    algorithm       TEXT    NOT NULL,
    reward          REAL    NOT NULL,
    timestamp       REAL    NOT NULL,
    ucb_score       REAL    DEFAULT 0,
    was_exploration INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS song_stats (
    song_id         TEXT PRIMARY KEY,
    total_plays     INTEGER DEFAULT 0,
    total_likes     INTEGER DEFAULT 0,
    total_skips     INTEGER DEFAULT 0,
    last_updated    REAL
);

CREATE INDEX IF NOT EXISTS idx_interactions_user ON interactions(user_id);
CREATE INDEX IF NOT EXISTS idx_users_session ON users(session_id);
"""


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _init_db_sync():
    with _db_lock:
        conn = _get_conn()
        conn.executescript(SCHEMA)
        now = time.time()
        for song in SONGS_LIST:
            conn.execute(
                "INSERT OR IGNORE INTO song_stats (song_id, total_plays, total_likes, total_skips, last_updated) VALUES (?, 0, 0, 0, ?)",
                (song["id"], now),
            )
        conn.commit()
        conn.close()


async def init_db():
    await asyncio.to_thread(_init_db_sync)


# ─────────────────────────────────────────────────────────────────────────────
# Users
# ─────────────────────────────────────────────────────────────────────────────

def _get_or_create_user_sync(session_id: str) -> dict:
    with _db_lock:
        conn = _get_conn()
        row = conn.execute("SELECT * FROM users WHERE session_id = ?", (session_id,)).fetchone()
        if row:
            result = dict(row)
            conn.close()
            return result
        now = time.time()
        conn.execute(
            "INSERT INTO users (session_id, created_at) VALUES (?, ?)",
            (session_id, now),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM users WHERE session_id = ?", (session_id,)).fetchone()
        result = dict(row)
        conn.close()
        return result


async def get_or_create_user(session_id: str) -> dict:
    return await asyncio.to_thread(_get_or_create_user_sync, session_id)


def _update_user_embedding_sync(user_id: int, embedding: np.ndarray, taste_profile: dict):
    with _db_lock:
        conn = _get_conn()
        conn.execute(
            "UPDATE users SET user_embedding = ?, taste_profile = ? WHERE id = ?",
            (json.dumps(embedding.tolist()), json.dumps(taste_profile), user_id),
        )
        conn.commit()
        conn.close()


async def update_user_embedding(user_id: int, embedding: np.ndarray, taste_profile: dict):
    await asyncio.to_thread(_update_user_embedding_sync, user_id, embedding, taste_profile)


# ─────────────────────────────────────────────────────────────────────────────
# Interactions
# ─────────────────────────────────────────────────────────────────────────────

def _record_interaction_sync(
    user_id: int,
    song_id: str,
    algorithm: str,
    reward: float,
    ucb_score: float = 0.0,
    was_exploration: bool = False,
) -> int:
    with _db_lock:
        conn = _get_conn()
        now = time.time()
        cur = conn.execute(
            "INSERT INTO interactions (user_id, song_id, algorithm, reward, timestamp, ucb_score, was_exploration) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (user_id, song_id, algorithm, reward, now, ucb_score, int(was_exploration)),
        )
        # Update song stats
        if reward >= 0.8:
            conn.execute("UPDATE song_stats SET total_likes = total_likes + 1, last_updated = ? WHERE song_id = ?", (now, song_id))
        elif reward < 0:
            conn.execute("UPDATE song_stats SET total_skips = total_skips + 1, last_updated = ? WHERE song_id = ?", (now, song_id))
        else:
            conn.execute("UPDATE song_stats SET total_plays = total_plays + 1, last_updated = ? WHERE song_id = ?", (now, song_id))
        conn.commit()
        iid = cur.lastrowid
        conn.close()
        return iid


async def record_interaction(
    user_id: int,
    song_id: str,
    algorithm: str,
    reward: float,
    ucb_score: float = 0.0,
    was_exploration: bool = False,
) -> int:
    return await asyncio.to_thread(
        _record_interaction_sync, user_id, song_id, algorithm, reward, ucb_score, was_exploration
    )


def _get_user_history_sync(user_id: int, limit: int = 50) -> List[dict]:
    with _db_lock:
        conn = _get_conn()
        rows = conn.execute(
            "SELECT * FROM interactions WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
        result = [dict(r) for r in rows]
        conn.close()
        return result


async def get_user_history(user_id: int, limit: int = 50) -> List[dict]:
    return await asyncio.to_thread(_get_user_history_sync, user_id, limit)


# ─────────────────────────────────────────────────────────────────────────────
# Song stats
# ─────────────────────────────────────────────────────────────────────────────

def _get_song_stats_sync(song_id: str) -> dict:
    with _db_lock:
        conn = _get_conn()
        row = conn.execute("SELECT * FROM song_stats WHERE song_id = ?", (song_id,)).fetchone()
        result = dict(row) if row else {}
        conn.close()
        return result


async def get_song_stats(song_id: str) -> dict:
    return await asyncio.to_thread(_get_song_stats_sync, song_id)


def _get_all_song_stats_sync() -> List[dict]:
    with _db_lock:
        conn = _get_conn()
        rows = conn.execute("SELECT * FROM song_stats").fetchall()
        result = [dict(r) for r in rows]
        conn.close()
        return result


async def get_all_song_stats() -> List[dict]:
    return await asyncio.to_thread(_get_all_song_stats_sync)


# ─────────────────────────────────────────────────────────────────────────────
# User embedding from interaction history
# ─────────────────────────────────────────────────────────────────────────────

def _compute_user_embedding_sync(user_id: int) -> np.ndarray:
    """Compute user embedding as weighted average of interacted song features."""
    with _db_lock:
        conn = _get_conn()
        rows = conn.execute(
            "SELECT song_id, reward FROM interactions WHERE user_id = ? ORDER BY timestamp DESC LIMIT 100",
            (user_id,),
        ).fetchall()
        conn.close()

    if not rows:
        return np.random.randn(21)

    total_weight = 0.0
    embedding = np.zeros(21)
    for row in rows:
        song = SONGS_DB.get(row["song_id"])
        if song:
            w = max(0.1, row["reward"] + 0.5)  # shift so skips have small weight
            embedding += w * get_song_feature_vector(song)
            total_weight += w

    if total_weight > 0:
        embedding /= total_weight

    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding /= norm
    return embedding


async def compute_user_embedding(user_id: int) -> np.ndarray:
    return await asyncio.to_thread(_compute_user_embedding_sync, user_id)


def _compute_taste_profile_sync(user_id: int) -> dict:
    """Compute taste profile as genre → engagement score."""
    with _db_lock:
        conn = _get_conn()
        rows = conn.execute(
            "SELECT song_id, reward FROM interactions WHERE user_id = ? AND reward > 0",
            (user_id,),
        ).fetchall()
        conn.close()

    genre_scores: dict = {}
    genre_counts: dict = {}
    for row in rows:
        song = SONGS_DB.get(row["song_id"])
        if song:
            g = song["genre"]
            genre_scores[g] = genre_scores.get(g, 0.0) + row["reward"]
            genre_counts[g] = genre_counts.get(g, 0) + 1

    profile = {}
    for g, score in genre_scores.items():
        profile[g] = round(score / genre_counts[g], 3)
    return profile


async def compute_taste_profile(user_id: int) -> dict:
    return await asyncio.to_thread(_compute_taste_profile_sync, user_id)


# ─────────────────────────────────────────────────────────────────────────────
# Leaderboard
# ─────────────────────────────────────────────────────────────────────────────

def _get_leaderboard_sync() -> List[dict]:
    with _db_lock:
        conn = _get_conn()
        rows = conn.execute(
            "SELECT song_id, total_plays, total_likes, total_skips FROM song_stats ORDER BY total_likes DESC LIMIT 50"
        ).fetchall()
        result = [dict(r) for r in rows]
        conn.close()
        return result


async def get_leaderboard() -> List[dict]:
    return await asyncio.to_thread(_get_leaderboard_sync)
