#!/bin/bash
# ╔══════════════════════════════════════════════════════════════════╗
# ║  BanditBeats — Music Recommendation via Diag-LinUCB             ║
# ║  Based on: "Online Matching: A Real-time Bandit System"         ║
# ║  Yi et al., Google DeepMind / YouTube — RecSys 2023             ║
# ╚══════════════════════════════════════════════════════════════════╝

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$SCRIPT_DIR/backend"
FRONTEND_DIR="$SCRIPT_DIR/frontend"
PORT=8000

echo ""
echo "🎵  BanditBeats — Bandit-Powered Music Recommender"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📄  Algorithm: Diag-LinUCB (Eq. 7-9, Yi et al. RecSys\'23)"
echo "🧠  Sparse bipartite graph · 20 clusters · 200 songs"
echo ""

# ── Dependencies ──────────────────────────────────────────────────
echo "📦  Installing Python dependencies..."
pip install -r "$BACKEND_DIR/requirements.txt" --break-system-packages -q   2>/dev/null || pip install -r "$BACKEND_DIR/requirements.txt" -q

# ── Start backend ─────────────────────────────────────────────────
echo "🚀  Starting backend on http://localhost:$PORT"
echo "📊  Interactive API docs: http://localhost:$PORT/docs"
echo ""
echo "🌐  Open in your browser:"
echo "    file://$FRONTEND_DIR/index.html"
echo ""
echo "    (Or serve with:  python3 -m http.server 3000 --directory $FRONTEND_DIR)"
echo ""
echo "Press Ctrl+C to stop."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

cd "$BACKEND_DIR"
PYTHONDONTWRITEBYTECODE=1 python3 -m uvicorn main:app \
  --host 0.0.0.0 \
  --port $PORT \
  --reload \
  --log-level info
