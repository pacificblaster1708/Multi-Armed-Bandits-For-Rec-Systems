"""
BanditBeats API — FastAPI backend
Music recommendation via Diag-LinUCB + sparse bipartite graphs.
"""
import asyncio
import json
import logging
import time
import uuid
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from music_catalog import SONGS_LIST, SONGS_DB, get_song_feature_vector
from graph import get_graph, N_CLUSTERS
from bandits import get_bandit_manager
from db import (
    init_db,
    get_or_create_user,
    record_interaction,
    get_user_history,
    get_song_stats,
    get_all_song_stats,
    compute_user_embedding,
    compute_taste_profile,
    get_leaderboard,
)

# ─────────────────────────────────────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("banditbeats")

app = FastAPI(
    title="BanditBeats - Music Recommendation System",
    description="Diag-LinUCB from Google DeepMind RecSys'23",
    version="2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# WebSocket manager
# ─────────────────────────────────────────────────────────────────────────────

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, session_id: str, websocket: WebSocket):
        await websocket.accept()
        self.active_connections[session_id] = websocket
        logger.info(f"WS connected: {session_id}")

    def disconnect(self, session_id: str):
        self.active_connections.pop(session_id, None)
        logger.info(f"WS disconnected: {session_id}")

    async def send_to(self, session_id: str, data: dict):
        ws = self.active_connections.get(session_id)
        if ws:
            try:
                await ws.send_text(json.dumps(data))
            except Exception:
                self.disconnect(session_id)

    async def broadcast(self, data: dict):
        dead = []
        for sid, ws in list(self.active_connections.items()):
            try:
                await ws.send_text(json.dumps(data))
            except Exception:
                dead.append(sid)
        for sid in dead:
            self.disconnect(sid)


ws_manager = ConnectionManager()


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic models
# ─────────────────────────────────────────────────────────────────────────────

class SessionResponse(BaseModel):
    session_id: str
    user_id: int
    created_at: float
    taste_profile: Dict[str, float]
    is_new_user: bool


class SongOut(BaseModel):
    id: str
    title: str
    artist: str
    album: str
    genre: str
    tempo: float
    energy: float
    valence: float
    danceability: float
    acousticness: float
    instrumentalness: float
    loudness: float
    duration_ms: int
    release_year: int
    play_count: int
    cover_color: str
    total_plays: int = 0
    total_likes: int = 0
    total_skips: int = 0


class RecommendationItem(BaseModel):
    song: SongOut
    ucb_score: float
    is_exploration: bool
    rank: int


class RecommendationResponse(BaseModel):
    session_id: str
    algorithm: str
    recommendations: List[RecommendationItem]
    context_weights: Dict[str, float]
    candidate_pool_size: int
    latency_ms: float


class FeedbackRequest(BaseModel):
    session_id: str
    song_id: str
    action: str = Field(..., description="like | skip | play | complete")
    algorithm: str = "diag_linucb"
    ucb_score: Optional[float] = None
    was_exploration: Optional[bool] = False


class FeedbackResponse(BaseModel):
    interaction_id: int
    reward: float
    algorithm: str
    metrics_update: dict


class MetricsSummary(BaseModel):
    algorithm: str
    n_interactions: int
    ctr: float
    avg_reward: float
    cumulative_reward: float
    exploration_ratio: float


# ─────────────────────────────────────────────────────────────────────────────
# Reward function
# ─────────────────────────────────────────────────────────────────────────────

def compute_reward(action: str) -> float:
    rewards = {
        "like": 1.0,
        "complete": 0.8,
        "play": 0.3,
        "skip": -0.2,
    }
    return rewards.get(action, 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Startup
# ─────────────────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    logger.info("Initialising database ...")
    await init_db()

    logger.info("Building sparse bipartite graph ...")
    graph = get_graph()  # triggers build()

    logger.info("Initialising bandit manager ...")
    manager = get_bandit_manager()

    # Build cluster assignments for all songs
    cluster_assignments: Dict[str, List[int]] = {
        song_id: clusters
        for song_id, clusters in graph.item_clusters.items()
    }
    # Ensure all songs have at least one cluster
    for song in SONGS_LIST:
        if song["id"] not in cluster_assignments:
            cluster_assignments[song["id"]] = [0]

    manager.initialize_songs(cluster_assignments)
    logger.info(f"Bandit manager ready — {len(SONGS_LIST)} songs, {N_CLUSTERS} clusters.")


# ─────────────────────────────────────────────────────────────────────────────
# REST endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    graph = get_graph()
    return {
        "status": "ok",
        "graph_built": graph.is_built,
        "n_songs": len(SONGS_LIST),
        "n_clusters": N_CLUSTERS,
        "timestamp": time.time(),
    }


@app.get("/songs", response_model=List[SongOut])
async def list_songs():
    stats_list = await get_all_song_stats()
    stats = {s["song_id"]: s for s in stats_list}
    result = []
    for song in SONGS_LIST:
        s = dict(song)
        stat = stats.get(song["id"], {})
        s["total_plays"] = stat.get("total_plays", 0)
        s["total_likes"] = stat.get("total_likes", 0)
        s["total_skips"] = stat.get("total_skips", 0)
        result.append(SongOut(**s))
    return result


@app.get("/songs/{song_id}", response_model=SongOut)
async def get_song(song_id: str):
    if song_id not in SONGS_DB:
        raise HTTPException(status_code=404, detail="Song not found")
    song = dict(SONGS_DB[song_id])
    stat = await get_song_stats(song_id)
    song["total_plays"] = stat.get("total_plays", 0)
    song["total_likes"] = stat.get("total_likes", 0)
    song["total_skips"] = stat.get("total_skips", 0)
    return SongOut(**song)


@app.post("/session", response_model=SessionResponse)
async def create_session(session_id: Optional[str] = None):
    if not session_id:
        session_id = str(uuid.uuid4())

    existing = False
    try:
        # Check if user already exists
        from db import DB_PATH
        import aiosqlite
        async with aiosqlite.connect(DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT id FROM users WHERE session_id = ?", (session_id,)
            ) as cursor:
                row = await cursor.fetchone()
                existing = row is not None
    except Exception:
        pass

    user = await get_or_create_user(session_id)
    taste = json.loads(user["taste_profile"]) if user.get("taste_profile") else {}

    return SessionResponse(
        session_id=session_id,
        user_id=user["id"],
        created_at=user["created_at"],
        taste_profile=taste,
        is_new_user=not existing,
    )


@app.get("/recommend/{session_id}", response_model=RecommendationResponse)
async def get_recommendations(
    session_id: str,
    algorithm: str = Query("diag_linucb"),
    n: int = Query(12, ge=1, le=50),
    explore_only: bool = Query(False),
    alpha: Optional[float] = Query(None),
):
    t0 = time.perf_counter()

    # Get or create user
    user = await get_or_create_user(session_id)
    user_id = user["id"]

    # Compute user embedding from history
    user_embedding = await compute_user_embedding(user_id)

    # Get sparse graph
    graph = get_graph()

    # Get user context (cluster weights)
    context_weights = graph.get_user_context(user_embedding)

    # Get candidate items from triggered clusters
    candidate_items = graph.get_candidate_items(context_weights)

    # Fallback: if no candidates, use all songs
    if not candidate_items:
        candidate_items = [s["id"] for s in SONGS_LIST]

    # Get bandit manager
    manager = get_bandit_manager()
    if alpha is not None:
        manager.set_alpha(alpha)

    # Get recommendations
    if explore_only:
        import random
        selected = random.sample(candidate_items, min(n, len(candidate_items)))
        recs_raw = [(sid, 0.0, True) for sid in selected]
    else:
        recs_raw = manager.get_recommendations(
            algorithm=algorithm,
            candidate_items=candidate_items,
            context_weights=context_weights,
            top_k=n,
        )

    # Build response
    stats_list = await get_all_song_stats()
    stats = {s["song_id"]: s for s in stats_list}
    recommendations = []
    for rank, (song_id, score, is_exploration) in enumerate(recs_raw):
        if song_id not in SONGS_DB:
            continue
        song_data = dict(SONGS_DB[song_id])
        stat = stats.get(song_id, {})
        song_data["total_plays"] = stat.get("total_plays", 0)
        song_data["total_likes"] = stat.get("total_likes", 0)
        song_data["total_skips"] = stat.get("total_skips", 0)
        recommendations.append(RecommendationItem(
            song=SongOut(**song_data),
            ucb_score=round(score, 4),
            is_exploration=is_exploration,
            rank=rank + 1,
        ))

    latency_ms = (time.perf_counter() - t0) * 1000

    return RecommendationResponse(
        session_id=session_id,
        algorithm=algorithm,
        recommendations=recommendations,
        context_weights={str(k): round(v, 4) for k, v in context_weights.items()},
        candidate_pool_size=len(candidate_items),
        latency_ms=round(latency_ms, 2),
    )


@app.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(req: FeedbackRequest):
    # Validate song
    if req.song_id not in SONGS_DB:
        raise HTTPException(status_code=404, detail="Song not found")

    # Get user
    user = await get_or_create_user(req.session_id)
    user_id = user["id"]

    # Compute reward
    reward = compute_reward(req.action)

    # Get user context for Diag-LinUCB update
    user_embedding = await compute_user_embedding(user_id)
    graph = get_graph()
    context_weights = graph.get_user_context(user_embedding)

    # Record in bandit
    manager = get_bandit_manager()
    manager.record_interaction(
        algorithm=req.algorithm,
        item_id=req.song_id,
        reward=reward,
        context_weights=context_weights if req.algorithm == "diag_linucb" else None,
        was_exploration=req.was_exploration or False,
    )

    # Update user embedding and taste profile
    # Persist interaction
    interaction_id = await record_interaction(
        user_id=user_id,
        song_id=req.song_id,
        algorithm=req.algorithm,
        reward=reward,
        ucb_score=req.ucb_score,
        was_exploration=req.was_exploration or False,
    )

    # Get updated metrics snapshot
    metrics = manager.get_metrics_summary()
    algo_metrics = metrics.get(req.algorithm, {})

    # Broadcast via WebSocket
    ws_payload = {
        "type": "metrics_update",
        "algorithm": req.algorithm,
        "reward": reward,
        "song_id": req.song_id,
        "action": req.action,
        "metrics": {
            k: {
                "ctr": v.get("ctr", 0),
                "cumulative_reward": v.get("cumulative_reward", 0),
                "n_interactions": v.get("n_interactions", 0),
                "avg_reward": v.get("avg_reward", 0),
            }
            for k, v in metrics.items()
        },
    }
    await ws_manager.send_to(req.session_id, ws_payload)

    return FeedbackResponse(
        interaction_id=interaction_id,
        reward=reward,
        algorithm=req.algorithm,
        metrics_update=algo_metrics,
    )


@app.get("/metrics")
async def get_metrics():
    manager = get_bandit_manager()
    summary = manager.get_metrics_summary()
    return summary


@app.get("/metrics/{algorithm}")
async def get_algorithm_metrics(algorithm: str):
    valid = ["diag_linucb", "linucb", "thompson", "ucb1", "epsilon_greedy"]
    if algorithm not in valid:
        raise HTTPException(status_code=400, detail=f"Unknown algorithm. Choose from: {valid}")
    manager = get_bandit_manager()
    summary = manager.get_metrics_summary()
    return summary.get(algorithm, {})


@app.get("/graph/clusters")
async def get_clusters():
    graph = get_graph()
    return graph.get_cluster_info()


@app.get("/graph/user/{session_id}")
async def get_user_graph(session_id: str):
    user = await get_or_create_user(session_id)
    user_id = user["id"]
    user_embedding = await compute_user_embedding(user_id)
    graph = get_graph()
    context_weights = graph.get_user_context(user_embedding)
    candidates = graph.get_candidate_items(context_weights)

    # Enrich with cluster info
    cluster_info = []
    for cluster_id, weight in sorted(context_weights.items(), key=lambda x: -x[1]):
        items = graph.cluster_items.get(cluster_id, [])
        top_songs = []
        for sid in items[:3]:
            if sid in SONGS_DB:
                s = SONGS_DB[sid]
                top_songs.append({"id": sid, "title": s["title"], "artist": s["artist"]})
        cluster_info.append({
            "cluster_id": cluster_id,
            "weight": round(weight, 4),
            "n_items": len(items),
            "top_songs": top_songs,
        })

    return {
        "session_id": session_id,
        "user_id": user_id,
        "cluster_assignments": cluster_info,
        "n_candidates": len(candidates),
        "taste_profile": json.loads(user.get("taste_profile") or "{}"),
    }


@app.get("/leaderboard")
async def get_leaderboard():
    """Top songs by engagement ratio per genre."""
    stats_list = await get_all_song_stats()
    stats = {s["song_id"]: s for s in stats_list}
    result = {}

    for song in SONGS_LIST:
        genre = song["genre"]
        stat = stats.get(song["id"], {})
        plays = stat.get("total_plays", 0)
        likes = stat.get("total_likes", 0)
        engagement = likes / max(plays, 1)

        if genre not in result:
            result[genre] = []
        result[genre].append({
            "song_id": song["id"],
            "title": song["title"],
            "artist": song["artist"],
            "cover_color": song["cover_color"],
            "total_plays": plays,
            "total_likes": likes,
            "engagement_ratio": round(engagement, 4),
        })

    # Sort each genre by engagement
    for genre in result:
        result[genre].sort(key=lambda x: x["engagement_ratio"], reverse=True)
        result[genre] = result[genre][:5]

    return result


# ─────────────────────────────────────────────────────────────────────────────
# WebSocket endpoint
# ─────────────────────────────────────────────────────────────────────────────

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await ws_manager.connect(session_id, websocket)
    try:
        # Send initial state
        manager = get_bandit_manager()
        metrics = manager.get_metrics_summary()
        await websocket.send_text(json.dumps({
            "type": "connected",
            "session_id": session_id,
            "metrics": {
                k: {
                    "ctr": v.get("ctr", 0),
                    "n_interactions": v.get("n_interactions", 0),
                    "cumulative_reward": v.get("cumulative_reward", 0),
                }
                for k, v in metrics.items()
            },
        }))

        # Keep connection alive
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                msg = json.loads(data)
                if msg.get("type") == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))
            except asyncio.TimeoutError:
                # Send heartbeat
                await websocket.send_text(json.dumps({"type": "heartbeat"}))
    except WebSocketDisconnect:
        ws_manager.disconnect(session_id)
    except Exception as e:
        logger.error(f"WS error for {session_id}: {e}")
        ws_manager.disconnect(session_id)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
