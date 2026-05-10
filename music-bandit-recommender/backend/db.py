"""SQLite database layer for BanditBeats."""
import json, time, sqlite3, asyncio, threading
from typing import List, Dict
import numpy as np

import os, tempfile
DB_PATH = os.path.join(tempfile.gettempdir(), "banditbeats.db")
_db_lock = threading.Lock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT UNIQUE NOT NULL,
    created_at REAL NOT NULL,
    user_embedding TEXT,
    taste_profile TEXT
);
CREATE TABLE IF NOT EXISTS interactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    song_id TEXT NOT NULL,
    algorithm TEXT NOT NULL,
    reward REAL NOT NULL,
    timestamp REAL NOT NULL,
    ucb_score REAL DEFAULT 0,
    was_exploration INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS song_stats (
    song_id TEXT PRIMARY KEY,
    total_plays INTEGER DEFAULT 0,
    total_likes INTEGER DEFAULT 0,
    total_skips INTEGER DEFAULT 0,
    last_updated REAL
);
CREATE INDEX IF NOT EXISTS idx_u ON interactions(user_id);
CREATE INDEX IF NOT EXISTS idx_s ON users(session_id);
"""

def _conn():
    c = sqlite3.connect(DB_PATH, check_same_thread=False)
    c.row_factory = sqlite3.Row
    return c

def _init_sync():
    from music_catalog import SONGS_LIST
    with _db_lock:
        c = _conn()
        c.executescript(SCHEMA)
        now = time.time()
        for song in SONGS_LIST:
            c.execute("INSERT OR IGNORE INTO song_stats (song_id,total_plays,total_likes,total_skips,last_updated) VALUES(?,0,0,0,?)", (song["id"], now))
        c.commit(); c.close()

async def init_db():
    await asyncio.to_thread(_init_sync)

def _get_or_create_sync(session_id):
    with _db_lock:
        c = _conn()
        row = c.execute("SELECT * FROM users WHERE session_id=?", (session_id,)).fetchone()
        if row:
            r = dict(row); c.close(); return r
        c.execute("INSERT INTO users(session_id,created_at) VALUES(?,?)", (session_id, time.time()))
        c.commit()
        row = c.execute("SELECT * FROM users WHERE session_id=?", (session_id,)).fetchone()
        r = dict(row); c.close(); return r

async def get_or_create_user(session_id):
    return await asyncio.to_thread(_get_or_create_sync, session_id)

def _record_sync(user_id, song_id, algorithm, reward, ucb_score=0.0, was_exploration=False):
    with _db_lock:
        c = _conn(); now = time.time()
        cur = c.execute("INSERT INTO interactions(user_id,song_id,algorithm,reward,timestamp,ucb_score,was_exploration) VALUES(?,?,?,?,?,?,?)",
            (user_id, song_id, algorithm, reward, now, ucb_score, int(was_exploration)))
        if reward >= 0.8:
            c.execute("UPDATE song_stats SET total_likes=total_likes+1,last_updated=? WHERE song_id=?", (now,song_id))
        elif reward < 0:
            c.execute("UPDATE song_stats SET total_skips=total_skips+1,last_updated=? WHERE song_id=?", (now,song_id))
        else:
            c.execute("UPDATE song_stats SET total_plays=total_plays+1,last_updated=? WHERE song_id=?", (now,song_id))
        c.commit(); iid = cur.lastrowid; c.close(); return iid

async def record_interaction(user_id, song_id, algorithm, reward, ucb_score=0.0, was_exploration=False):
    return await asyncio.to_thread(_record_sync, user_id, song_id, algorithm, reward, ucb_score, was_exploration)

def _history_sync(user_id, limit=50):
    with _db_lock:
        c = _conn()
        rows = c.execute("SELECT * FROM interactions WHERE user_id=? ORDER BY timestamp DESC LIMIT ?", (user_id,limit)).fetchall()
        r = [dict(x) for x in rows]; c.close(); return r

async def get_user_history(user_id, limit=50):
    return await asyncio.to_thread(_history_sync, user_id, limit)

def _song_stats_sync(song_id):
    with _db_lock:
        c = _conn()
        row = c.execute("SELECT * FROM song_stats WHERE song_id=?", (song_id,)).fetchone()
        r = dict(row) if row else {}; c.close(); return r

async def get_song_stats(song_id):
    return await asyncio.to_thread(_song_stats_sync, song_id)

def _all_stats_sync():
    with _db_lock:
        c = _conn()
        rows = c.execute("SELECT * FROM song_stats").fetchall()
        r = [dict(x) for x in rows]; c.close(); return r

async def get_all_song_stats():
    return await asyncio.to_thread(_all_stats_sync)

def _user_emb_sync(user_id):
    from music_catalog import SONGS_DB, get_song_feature_vector
    with _db_lock:
        c = _conn()
        rows = c.execute("SELECT song_id,reward FROM interactions WHERE user_id=? ORDER BY timestamp DESC LIMIT 100", (user_id,)).fetchall()
        c.close()
    if not rows:
        v = np.random.randn(21); return v/np.linalg.norm(v)
    emb = np.zeros(21); total = 0.0
    for row in rows:
        song = SONGS_DB.get(row["song_id"])
        if song:
            w = max(0.1, row["reward"] + 0.5)
            emb += w * get_song_feature_vector(song); total += w
    if total > 0: emb /= total
    n = np.linalg.norm(emb)
    return emb/n if n > 0 else emb

async def compute_user_embedding(user_id):
    return await asyncio.to_thread(_user_emb_sync, user_id)

def _taste_sync(user_id):
    from music_catalog import SONGS_DB
    with _db_lock:
        c = _conn()
        rows = c.execute("SELECT song_id,reward FROM interactions WHERE user_id=? AND reward>0", (user_id,)).fetchall()
        c.close()
    gs, gc = {}, {}
    for row in rows:
        song = SONGS_DB.get(row["song_id"])
        if song:
            g = song["genre"]
            gs[g] = gs.get(g,0.0)+row["reward"]; gc[g] = gc.get(g,0)+1
    return {g: round(gs[g]/gc[g],3) for g in gs}

async def compute_taste_profile(user_id):
    return await asyncio.to_thread(_taste_sync, user_id)

def _leaderboard_sync():
    with _db_lock:
        c = _conn()
        rows = c.execute("SELECT song_id,total_plays,total_likes,total_skips FROM song_stats ORDER BY total_likes DESC LIMIT 50").fetchall()
        r = [dict(x) for x in rows]; c.close(); return r

async def get_leaderboard():
    return await asyncio.to_thread(_leaderboard_sync)
