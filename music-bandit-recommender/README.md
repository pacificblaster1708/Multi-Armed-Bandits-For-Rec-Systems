# BanditBeats — Music Recommendation via Diag-LinUCB

A full end-to-end music recommendation system implementing the core algorithm
from Google DeepMind's **"Online Matching: A Real-time Bandit System for
Large-Scale Recommendations"** (RecSys 2023).

---

## Paper Citation

```
Yi, J., Ghosh, A., Vartak, M., Hong, L., Chi, E., & Zheng, N. (2023).
Online Matching: A Real-time Bandit System for Large-Scale Recommendations.
Proceedings of the 17th ACM Conference on Recommender Systems (RecSys '23).
Google DeepMind / YouTube.
```

---

## What This Implements

| Component | Description |
|-----------|-------------|
| **Diag-LinUCB** | Diagonal approximation of LinUCB — O(1) update, no matrix inversion |
| **Sparse Bipartite Graph** | Offline graph connecting user clusters to candidate items |
| **Online Policy Update** | Real-time parameter updates on every like/skip/play event |
| **Exploration Bonus** | UCB exploration term calibrated by α slider |
| **5 Algorithms** | Diag-LinUCB, LinUCB, Thompson Sampling, UCB1, ε-Greedy |
| **200 Songs** | 12 genres, rich audio features (tempo, energy, valence, etc.) |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                          OFFLINE PHASE                           │
│                                                                  │
│  200 Songs ──► Feature Vectors ──► K-Means (C=20 clusters)     │
│                                         │                        │
│                          Top-W items per cluster                 │
│                                         │                        │
│                     Sparse Bipartite Graph G(U,V,E)             │
└──────────────────────────────────────────────────────────────────┘
                               │
                    Stored in SparseGraph object
                               │
┌──────────────────────────────────────────────────────────────────┐
│                          ONLINE PHASE                            │
│                                                                  │
│  User Request ──► Compute user embedding                        │
│       │           ──► Assign to top-K clusters (Eq. 10)         │
│       │           ──► Softmax weights w_{u,c}                   │
│       │                    │                                     │
│       │           ──► Retrieve candidate items I_candidate      │
│       │                    │                                     │
│       │           ──► Compute UCB for each item (Eq. 7):        │
│       │               UCB_j = Σ_c w_c·b_{j,c}/d_{j,c}          │
│       │                      + α·√(Σ_c w²_c/d_{j,c})           │
│       │                    │                                     │
│       └──► Recommend top-K items                                │
│                                                                  │
│  Feedback ──► reward r                                          │
│           ──► Update d_{j,c} += w²_{u,c}     (Eq. 8)           │
│           ──► Update b_{j,c} += w_{u,c}·r    (Eq. 9)           │
│           ──► Broadcast via WebSocket                           │
└──────────────────────────────────────────────────────────────────┘

┌──────────────┐    HTTP/WS    ┌──────────────┐
│  Frontend    │◄─────────────►│   FastAPI    │
│  index.html  │               │  Backend     │
│  Chart.js    │               │  :8000       │
│  WebSocket   │               │  SQLite DB   │
└──────────────┘               └──────────────┘
```

---

## Key Algorithm: Diag-LinUCB

### Why Diagonal?

Classic LinUCB maintains a full d×d covariance matrix A_j per item:
- Memory: O(d²) per item — prohibitive at YouTube scale
- Update cost: O(d²) per feedback
- Score computation: requires matrix inversion

Diag-LinUCB approximates A_j with only its diagonal d_j:
- Memory: O(d) per item per cluster
- Update cost: O(1) per feedback (scalar operations only)
- Score computation: O(K) where K = active clusters per user

### Equations

**UCB Score (Eq. 7):**
```
UCB_j(u) = Σ_{c ∈ supp(w_u)} w_{u,c} * b_{j,c} / d_{j,c}
           + α * sqrt(Σ_{c ∈ supp(w_u)} w²_{u,c} / d_{j,c})
```

**Diagonal Update (Eq. 8):**
```
d_{j,c} ← d_{j,c} + w²_{u,c}
```

**Mean Vector Update (Eq. 9):**
```
b_{j,c} ← b_{j,c} + w_{u,c} * r_{u,j}
```

**User Cluster Weights (Eq. 10):**
```
w_{u,c} = softmax(<u, centroid_c> / τ')   for c in top-K clusters
```

Where:
- `w_{u,c}` = weight of user u in cluster c (sparse, only K non-zeros)
- `b_{j,c}` = reward numerator for item j in cluster c
- `d_{j,c}` = diagonal covariance (confidence) for item j in cluster c
- `α` = exploration bonus strength (tunable)
- `τ'` = softmax temperature (default 0.3)
- `K` = clusters per user (default 5)
- `C` = total clusters (default 20)

---

## Project Structure

```
music-bandit-recommender/
├── backend/
│   ├── requirements.txt    # Python dependencies
│   ├── music_catalog.py    # 200 songs with audio features
│   ├── bandits.py          # DiagLinUCB + 4 baseline algorithms
│   ├── graph.py            # Sparse bipartite graph (Algorithm 2)
│   ├── database.py         # Async SQLite (aiosqlite)
│   └── main.py             # FastAPI app with REST + WebSocket
├── frontend/
│   └── index.html          # Complete SPA (Chart.js, WebSocket)
├── start.sh                # One-command startup
└── README.md               # This file
```

---

## Installation & Running

### Prerequisites
- Python 3.9+
- pip

### Quick Start

```bash
cd music-bandit-recommender
bash start.sh
```

Then open `frontend/index.html` in your browser.

### Manual Start

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Open `frontend/index.html` directly in your browser (no web server needed).

---

## API Documentation

After starting the backend, visit: http://localhost:8000/docs

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | System health + graph status |
| GET | `/songs` | All 200 songs with stats |
| GET | `/songs/{id}` | Single song detail |
| POST | `/session` | Create/retrieve user session |
| GET | `/recommend/{session_id}` | Get recommendations |
| POST | `/feedback` | Submit like/skip/play/complete |
| GET | `/metrics` | All algorithm performance metrics |
| GET | `/metrics/{algo}` | Per-algorithm metrics |
| GET | `/graph/clusters` | Cluster info + top songs |
| GET | `/graph/user/{session_id}` | User cluster assignments |
| GET | `/leaderboard` | Top songs by engagement per genre |
| WS | `/ws/{session_id}` | Real-time metric updates |

### Recommendation Query Params

```
GET /recommend/{session_id}
  ?algorithm=diag_linucb    # diag_linucb|linucb|thompson|ucb1|epsilon_greedy
  &n=12                     # number of recommendations
  &alpha=1.0                # exploration bonus (overrides slider)
  &explore_only=false       # pure exploration mode
```

### Feedback Body

```json
{
  "session_id": "...",
  "song_id": "song_042",
  "action": "like",          // like|skip|play|complete
  "algorithm": "diag_linucb",
  "ucb_score": 1.234,
  "was_exploration": false
}
```

### Reward Values

| Action | Reward |
|--------|--------|
| like | +1.0 |
| complete | +0.8 |
| play | +0.3 |
| skip | -0.2 |

---

## Algorithm Descriptions

### 1. Diag-LinUCB (Paper Algorithm)
The core contribution. Maintains only diagonal d_{j,c} instead of full
covariance matrix. Enables O(1) updates and distributes naturally over
sparse graph edges. Trades some statistical efficiency for massive
scalability.

### 2. LinUCB (Baseline)
Classic contextual bandit with full d×d covariance per item. Statistically
optimal but O(d²) memory per item — impractical at scale.

### 3. Thompson Sampling (Baseline)
Bayesian approach using Beta(α, β) prior over binary rewards. Samples
a score from the posterior and recommends highest-scoring items.
No context awareness.

### 4. UCB1 (Baseline)
Non-contextual upper confidence bound. Balances exploration and exploitation
using UCB = mean_reward + sqrt(2 * log(t) / n_pulls). Simple and
theoretically sound.

### 5. ε-Greedy (Baseline)
With probability ε explores randomly, otherwise exploits best known item.
No formal uncertainty quantification.

---

## Performance Metrics

The dashboard shows real-time metrics per algorithm:

- **CTR** (Click-Through Rate): likes / total interactions
- **Avg Reward**: mean reward across all interactions
- **Cumulative Reward**: sum of all rewards over time
- **Exploration Ratio**: fraction of picks that were exploration
- **Cumulative Reward Curve**: shows learning over time

Expected: Diag-LinUCB should converge to higher CTR than baselines
after ~20-50 interactions as it learns user cluster preferences.

---

## Exploration / Exploitation Tradeoff

The α slider (0.1 – 3.0) controls the exploration bonus:

- **Low α (0.1)**: Heavy exploitation — recommends well-known songs
- **High α (3.0)**: Heavy exploration — tries uncertain/new items

The 🔭 icon on song cards indicates **exploration picks** (high UCB
uncertainty). The ⚡ icon indicates **exploitation picks** (high
predicted reward).

---

## Screenshots

*(Place screenshots here after running the system)*

- `screenshots/main-ui.png` — Full 3-column dashboard
- `screenshots/ucb-decomp.png` — UCB formula breakdown panel
- `screenshots/metrics-charts.png` — Real-time Chart.js panels
- `screenshots/cluster-weights.png` — User cluster assignment
