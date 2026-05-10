"""
Bandit Algorithms for Music Recommendation
Implements Diag-LinUCB from "Online Matching: A Real-time Bandit System"
Yi et al., RecSys 2023 (Google DeepMind / YouTube)
"""
import threading
import math
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from music_catalog import SONGS_LIST, get_song_feature_vector, SONGS_DB


# ---------------------------------------------------------------------------
# Diag-LinUCB  (Core contribution of the paper)
# ---------------------------------------------------------------------------

class DiagLinUCB:
    """
    Diagonal LinUCB from "Online Matching: A Real-time Bandit System"
    Yi et al., RecSys 2023 (Google DeepMind / YouTube)

    Instead of maintaining full d×d covariance matrix A_j for each item,
    only maintains the diagonal d_j. This enables:
      - O(1) updates per feedback (no matrix inversion)
      - Fully distributed updates over sparse graph edges
      - Scalable to millions of items

    UCB_j = Σ_{c∈supp(w_u)} w_{u,c} * b_{j,c}/d_{j,c}
            + α * sqrt(Σ w²_{u,c}/d_{j,c})

    Updates:
      d_{j,c} += w²_{u,c}    (Eq. 8 – diagonal covariance increment)
      b_{j,c} += w_{u,c} * r_{u,j}  (Eq. 9 – mean vector increment)
    """

    def __init__(self, n_clusters: int, alpha: float = 1.0):
        self.n_clusters = n_clusters
        self.alpha = alpha
        # item_id -> {cluster_id -> [d, b]}  (d=diagonal, b=numerator)
        self.params: Dict[str, Dict[int, List[float]]] = {}
        self.lock = threading.Lock()

    def init_item(self, item_id: str, cluster_ids: List[int]):
        """Initialize item with d=1.0 (unit confidence) for every supplied cluster."""
        with self.lock:
            if item_id not in self.params:
                self.params[item_id] = {}
            for c in cluster_ids:
                if c not in self.params[item_id]:
                    # d=1.0 → uncertainty = alpha (max exploration at start)
                    # b=0.0 → no prior reward
                    self.params[item_id][c] = [1.0, 0.0]

    def compute_ucb(self, item_id: str, context_weights: Dict[int, float]) -> float:
        """
        Compute UCB score for item given sparse user context (Eq. 7).

        UCB_j = exploitation_term + alpha * exploration_bonus
          exploitation_term = Σ_c w_c * b_{j,c} / d_{j,c}
          exploration_bonus = sqrt(Σ_c w²_c / d_{j,c})
        """
        if item_id not in self.params:
            return float('inf')  # Never seen → explore first

        item_params = self.params[item_id]
        exploitation = 0.0
        exploration_sq = 0.0

        for c, w in context_weights.items():
            if c in item_params:
                d, b = item_params[c]
                exploitation += w * b / d
                exploration_sq += (w * w) / d
            else:
                # Cluster not initialized for this item → high uncertainty
                exploration_sq += w * w  # d=1.0 implicit

        exploration_bonus = math.sqrt(max(exploration_sq, 0.0))
        return exploitation + self.alpha * exploration_bonus

    def update(self, item_id: str, context_weights: Dict[int, float], reward: float):
        """
        Update diagonal covariance and mean vector (Eqs. 8 & 9).

        d_{j,c} += w²_{u,c}
        b_{j,c} += w_{u,c} * r_{u,j}
        """
        with self.lock:
            if item_id not in self.params:
                self.params[item_id] = {}

            for c, w in context_weights.items():
                if c not in self.params[item_id]:
                    self.params[item_id][c] = [1.0, 0.0]
                self.params[item_id][c][0] += w * w          # d update (Eq. 8)
                self.params[item_id][c][1] += w * reward      # b update (Eq. 9)

    def get_recommendations(
        self,
        candidate_items: List[str],
        context_weights: Dict[int, float],
        top_k: int = 10,
        top_k_random: int = 5,
    ) -> List[Tuple[str, float]]:
        """
        Get top-k recommendations using UCB scores.
        top_k_random: randomly replace this many slots for latency/diversity robustness
        (mirrors the paper's approach to handle late-arriving scores).
        """
        scores = []
        for item_id in candidate_items:
            ucb = self.compute_ucb(item_id, context_weights)
            scores.append((item_id, ucb))

        scores.sort(key=lambda x: x[1], reverse=True)

        # Take top_k, then optionally shuffle in some random candidates
        top_items = scores[:top_k]
        if top_k_random > 0 and len(scores) > top_k:
            remaining = scores[top_k:]
            rng = np.random.default_rng()
            random_picks = rng.choice(
                len(remaining),
                size=min(top_k_random, len(remaining)),
                replace=False
            )
            for idx in random_picks:
                top_items.append(remaining[idx])

        return top_items


# ---------------------------------------------------------------------------
# Classic LinUCB (full covariance)
# ---------------------------------------------------------------------------

class LinUCB:
    """Classic LinUCB with full d×d covariance matrix per item."""

    def __init__(self, feature_dim: int, alpha: float = 1.0):
        self.feature_dim = feature_dim
        self.alpha = alpha
        # item_id -> (A: d×d matrix, b: d-vector)
        self.params: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self.lock = threading.Lock()

    def _init_item(self, item_id: str):
        if item_id not in self.params:
            A = np.identity(self.feature_dim)
            b = np.zeros(self.feature_dim)
            self.params[item_id] = (A, b)

    def compute_ucb(self, item_id: str, features: np.ndarray) -> float:
        with self.lock:
            self._init_item(item_id)
            A, b = self.params[item_id]

        try:
            A_inv = np.linalg.inv(A)
        except np.linalg.LinAlgError:
            A_inv = np.linalg.pinv(A)

        theta = A_inv @ b
        exploitation = theta @ features
        exploration = self.alpha * math.sqrt(features @ A_inv @ features)
        return exploitation + exploration

    def update(self, item_id: str, features: np.ndarray, reward: float):
        with self.lock:
            self._init_item(item_id)
            A, b = self.params[item_id]
            A += np.outer(features, features)
            b += reward * features
            self.params[item_id] = (A, b)

    def get_recommendations(
        self,
        candidate_items: List[str],
        features_map: Dict[str, np.ndarray],
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        scores = [
            (item_id, self.compute_ucb(item_id, features_map[item_id]))
            for item_id in candidate_items
            if item_id in features_map
        ]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


# ---------------------------------------------------------------------------
# Thompson Sampling (Beta-Bernoulli)
# ---------------------------------------------------------------------------

class ThompsonSampling:
    """Thompson Sampling with Beta prior for binary rewards."""

    def __init__(self):
        # item_id -> [alpha, beta] parameters
        self.params: Dict[str, List[float]] = {}
        self.lock = threading.Lock()

    def _init_item(self, item_id: str):
        if item_id not in self.params:
            self.params[item_id] = [1.0, 1.0]  # Uniform prior

    def compute_score(self, item_id: str) -> float:
        """Sample from Beta(α, β) distribution."""
        with self.lock:
            self._init_item(item_id)
            a, b = self.params[item_id]
        return float(np.random.beta(a, b))

    def update(self, item_id: str, reward: float):
        """Update: reward=1 → like, reward=0 → skip/neutral."""
        with self.lock:
            self._init_item(item_id)
            if reward > 0.5:
                self.params[item_id][0] += 1.0  # alpha (successes)
            else:
                self.params[item_id][1] += 1.0  # beta (failures)

    def get_recommendations(
        self,
        candidate_items: List[str],
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        scores = [(item_id, self.compute_score(item_id)) for item_id in candidate_items]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


# ---------------------------------------------------------------------------
# UCB1 (non-contextual)
# ---------------------------------------------------------------------------

class UCB1:
    """Classic UCB1 bandit (non-contextual)."""

    def __init__(self):
        # item_id -> [total_reward, n_pulls]
        self.params: Dict[str, List[float]] = {}
        self.t: int = 0  # global time step
        self.lock = threading.Lock()

    def _init_item(self, item_id: str):
        if item_id not in self.params:
            self.params[item_id] = [0.0, 0]

    def compute_ucb(self, item_id: str, t: int) -> float:
        with self.lock:
            self._init_item(item_id)
            total_reward, n_pulls = self.params[item_id]

        if n_pulls == 0:
            return float('inf')

        mean_reward = total_reward / n_pulls
        exploration = math.sqrt(2 * math.log(max(t, 1)) / n_pulls)
        return mean_reward + exploration

    def update(self, item_id: str, reward: float):
        with self.lock:
            self._init_item(item_id)
            self.params[item_id][0] += reward
            self.params[item_id][1] += 1
            self.t += 1

    def get_recommendations(
        self,
        candidate_items: List[str],
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        t = self.t
        scores = [(item_id, self.compute_ucb(item_id, t)) for item_id in candidate_items]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


# ---------------------------------------------------------------------------
# Epsilon-Greedy
# ---------------------------------------------------------------------------

class EpsilonGreedy:
    """ε-greedy exploration strategy."""

    def __init__(self, epsilon: float = 0.1):
        self.epsilon = epsilon
        # item_id -> [total_reward, n_pulls]
        self.params: Dict[str, List[float]] = {}
        self.lock = threading.Lock()

    def _init_item(self, item_id: str):
        if item_id not in self.params:
            self.params[item_id] = [0.0, 0]

    def compute_score(self, item_id: str) -> float:
        with self.lock:
            self._init_item(item_id)
            total_reward, n_pulls = self.params[item_id]
        if n_pulls == 0:
            return float('inf')
        return total_reward / n_pulls

    def update(self, item_id: str, reward: float):
        with self.lock:
            self._init_item(item_id)
            self.params[item_id][0] += reward
            self.params[item_id][1] += 1

    def get_recommendations(
        self,
        candidate_items: List[str],
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        rng = np.random.default_rng()
        scores = []
        for item_id in candidate_items:
            if rng.random() < self.epsilon:
                score = float(rng.random())  # explore
            else:
                score = self.compute_score(item_id)  # exploit
            scores.append((item_id, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


# ---------------------------------------------------------------------------
# BanditManager
# ---------------------------------------------------------------------------

class BanditManager:
    """Manages all bandit instances and tracks performance metrics."""

    ALGORITHM_NAMES = ["diag_linucb", "linucb", "thompson", "ucb1", "epsilon_greedy"]

    def __init__(self, n_clusters: int = 20, feature_dim: int = 21, alpha: float = 1.0):
        self.n_clusters = n_clusters
        self.feature_dim = feature_dim

        self.algorithms: Dict[str, Any] = {
            "diag_linucb": DiagLinUCB(n_clusters=n_clusters, alpha=alpha),
            "linucb": LinUCB(feature_dim=feature_dim, alpha=alpha),
            "thompson": ThompsonSampling(),
            "ucb1": UCB1(),
            "epsilon_greedy": EpsilonGreedy(epsilon=0.1),
        }

        # Per-algorithm metrics storage
        self.metrics: Dict[str, Dict] = {
            name: {
                "rewards": [],
                "cumulative_rewards": [],
                "ctrs": [],          # click-through rates (likes / total)
                "n_likes": 0,
                "n_interactions": 0,
                "n_explorations": 0,
                "n_exploitations": 0,
            }
            for name in self.ALGORITHM_NAMES
        }
        self.lock = threading.Lock()

        # Pre-compute song feature vectors
        self.song_features: Dict[str, np.ndarray] = {
            song["id"]: get_song_feature_vector(song)
            for song in SONGS_LIST
        }

    def initialize_songs(self, cluster_assignments: Dict[str, List[int]]):
        """Initialize all bandit algorithms with all songs at startup."""
        diag = self.algorithms["diag_linucb"]
        for song_id, cluster_ids in cluster_assignments.items():
            diag.init_item(song_id, cluster_ids)

        # LinUCB, Thompson, UCB1, EpsilonGreedy lazily init on first access
        # but we can prime them here
        linucb = self.algorithms["linucb"]
        thompson = self.algorithms["thompson"]
        ucb1 = self.algorithms["ucb1"]
        eg = self.algorithms["epsilon_greedy"]
        for song_id in SONGS_DB:
            linucb._init_item(song_id)
            thompson._init_item(song_id)
            ucb1._init_item(song_id)
            eg._init_item(song_id)

    def set_alpha(self, alpha: float):
        """Dynamically update exploration parameter."""
        self.algorithms["diag_linucb"].alpha = alpha
        self.algorithms["linucb"].alpha = alpha

    def record_interaction(
        self,
        algorithm: str,
        item_id: str,
        reward: float,
        context_weights: Optional[Dict[int, float]] = None,
        was_exploration: bool = False,
    ):
        """Record an interaction and update the appropriate bandit."""
        if algorithm not in self.algorithms:
            return

        # Update the chosen algorithm
        if algorithm == "diag_linucb" and context_weights:
            self.algorithms["diag_linucb"].update(item_id, context_weights, reward)
        elif algorithm == "linucb" and item_id in self.song_features:
            self.algorithms["linucb"].update(item_id, self.song_features[item_id], reward)
        elif algorithm == "thompson":
            self.algorithms["thompson"].update(item_id, reward)
        elif algorithm == "ucb1":
            self.algorithms["ucb1"].update(item_id, reward)
        elif algorithm == "epsilon_greedy":
            self.algorithms["epsilon_greedy"].update(item_id, reward)

        # Update metrics
        with self.lock:
            m = self.metrics[algorithm]
            m["rewards"].append(reward)
            cumsum = (m["cumulative_rewards"][-1] if m["cumulative_rewards"] else 0) + reward
            m["cumulative_rewards"].append(cumsum)
            m["n_interactions"] += 1
            if reward >= 0.8:
                m["n_likes"] += 1
            if was_exploration:
                m["n_explorations"] += 1
            else:
                m["n_exploitations"] += 1
            n = m["n_interactions"]
            ctr = m["n_likes"] / n if n > 0 else 0.0
            m["ctrs"].append(ctr)

    def get_metrics_summary(self) -> dict:
        """Return a summary of all algorithm metrics."""
        summary = {}
        with self.lock:
            for name, m in self.metrics.items():
                n = m["n_interactions"]
                summary[name] = {
                    "n_interactions": n,
                    "n_likes": m["n_likes"],
                    "ctr": m["n_likes"] / n if n > 0 else 0.0,
                    "avg_reward": sum(m["rewards"]) / len(m["rewards"]) if m["rewards"] else 0.0,
                    "cumulative_reward": m["cumulative_rewards"][-1] if m["cumulative_rewards"] else 0.0,
                    "n_explorations": m["n_explorations"],
                    "n_exploitations": m["n_exploitations"],
                    "exploration_ratio": (
                        m["n_explorations"] / n if n > 0 else 0.0
                    ),
                    "recent_rewards": m["rewards"][-20:],
                    "cumulative_rewards_series": m["cumulative_rewards"][-100:],
                    "ctrs_series": m["ctrs"][-100:],
                }
        return summary

    def get_recommendations(
        self,
        algorithm: str,
        candidate_items: List[str],
        context_weights: Optional[Dict[int, float]] = None,
        top_k: int = 12,
    ) -> List[Tuple[str, float, bool]]:
        """
        Get recommendations from the specified algorithm.
        Returns list of (song_id, score, is_exploration).
        """
        if algorithm == "diag_linucb":
            if not context_weights:
                context_weights = {i: 1.0 / self.n_clusters for i in range(self.n_clusters)}
            raw = self.algorithms["diag_linucb"].get_recommendations(
                candidate_items, context_weights, top_k=top_k, top_k_random=3
            )
            # Determine exploration vs exploitation
            results = []
            for item_id, score in raw:
                params = self.algorithms["diag_linucb"].params.get(item_id, {})
                # High exploration if item has been seen few times
                total_d = sum(v[0] for v in params.values()) if params else 1.0
                is_exploration = total_d < 5.0
                results.append((item_id, score, is_exploration))
            return results

        elif algorithm == "linucb":
            raw = self.algorithms["linucb"].get_recommendations(
                candidate_items, self.song_features, top_k=top_k
            )
            return [(item_id, score, False) for item_id, score in raw]

        elif algorithm == "thompson":
            raw = self.algorithms["thompson"].get_recommendations(candidate_items, top_k=top_k)
            return [(item_id, score, False) for item_id, score in raw]

        elif algorithm == "ucb1":
            raw = self.algorithms["ucb1"].get_recommendations(candidate_items, top_k=top_k)
            results = []
            for item_id, score in raw:
                n_pulls = self.algorithms["ucb1"].params.get(item_id, [0, 0])[1]
                results.append((item_id, score, n_pulls == 0))
            return results

        elif algorithm == "epsilon_greedy":
            raw = self.algorithms["epsilon_greedy"].get_recommendations(
                candidate_items, top_k=top_k
            )
            return [(item_id, score, score == float('inf')) for item_id, score in raw]

        else:
            # Fallback: random
            import random
            sampled = random.sample(candidate_items, min(top_k, len(candidate_items)))
            return [(item_id, 0.0, True) for item_id in sampled]


# Module-level singleton
_manager: Optional[BanditManager] = None


def get_bandit_manager() -> BanditManager:
    global _manager
    if _manager is None:
        _manager = BanditManager()
    return _manager
