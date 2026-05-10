"""
Sparse Bipartite Graph Construction (Algorithm 2 from the paper)
Offline component: clusters items, builds cluster → item sparse bipartite graph.

Reference: Yi et al., "Online Matching: A Real-time Bandit System for Large-Scale
Recommendations", RecSys 2023, Google DeepMind / YouTube.
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from typing import Dict, List, Set, Tuple, Optional
from music_catalog import SONGS_LIST, get_song_feature_vector, GENRES

# ── Hyper-parameters (matching paper notation) ──────────────────────────────
N_CLUSTERS = 20          # C  – number of clusters
N_TOP_ITEMS_PER_CLUSTER = 30  # W  – max items per cluster in the graph
K_CLUSTERS_PER_USER = 5  # K  – clusters triggered per user query
SOFTMAX_TEMP = 0.3        # τ' – softmax temperature for cluster weights
# ─────────────────────────────────────────────────────────────────────────────


class SparseGraph:
    """
    Sparse bipartite graph G = (U_clusters, V_items, E).

    Offline phase (build):
      1. Embed all items in feature space.
      2. Run K-Means on item embeddings to get C cluster centroids.
      3. For each cluster c, keep top-W items by cosine similarity to centroid.
      4. Build reverse index: item → list of clusters it belongs to.

    Online phase (get_user_context):
      1. Project user embedding into cluster space.
      2. Compute softmax weights over top-K clusters (Eq. 10).
      3. Return sparse dict {cluster_id: weight}.
    """

    def __init__(self):
        self.kmeans: Optional[KMeans] = None
        self.cluster_items: Dict[int, List[str]] = {}   # cluster_id -> [song_ids]
        self.item_clusters: Dict[str, List[int]] = {}   # song_id   -> [cluster_ids]
        self.centroids: Optional[np.ndarray] = None     # shape (C, feature_dim)
        self.item_embeddings: Dict[str, np.ndarray] = {}
        self.is_built: bool = False

    # ------------------------------------------------------------------ #
    # Offline build                                                        #
    # ------------------------------------------------------------------ #

    def build(self):
        """Build sparse bipartite graph offline using all songs."""
        print("[Graph] Building sparse bipartite graph ...")

        # 1. Compute normalised embeddings for all items
        song_ids = [s["id"] for s in SONGS_LIST]
        raw_embeddings = np.array([get_song_feature_vector(s) for s in SONGS_LIST])
        embeddings = normalize(raw_embeddings, norm="l2")  # unit-length vectors

        for sid, emb in zip(song_ids, embeddings):
            self.item_embeddings[sid] = emb

        # 2. K-Means clustering on item embeddings
        self.kmeans = KMeans(
            n_clusters=N_CLUSTERS,
            n_init=10,
            random_state=42,
            max_iter=300,
        )
        self.kmeans.fit(embeddings)
        self.centroids = normalize(self.kmeans.cluster_centers_, norm="l2")

        # 3. For each cluster, select top-W items by cosine similarity
        for c in range(N_CLUSTERS):
            centroid = self.centroids[c]
            sims = embeddings @ centroid          # cosine similarity (embeddings are unit)
            top_indices = np.argsort(sims)[::-1][:N_TOP_ITEMS_PER_CLUSTER]
            self.cluster_items[c] = [song_ids[i] for i in top_indices]

        # 4. Build reverse index: item → clusters
        for c, items in self.cluster_items.items():
            for item_id in items:
                if item_id not in self.item_clusters:
                    self.item_clusters[item_id] = []
                self.item_clusters[item_id].append(c)

        self.is_built = True
        n_edges = sum(len(v) for v in self.cluster_items.values())
        print(
            f"[Graph] Built: {N_CLUSTERS} clusters, "
            f"{len(SONGS_LIST)} items, "
            f"{n_edges} edges in sparse graph."
        )

    # ------------------------------------------------------------------ #
    # Online inference                                                     #
    # ------------------------------------------------------------------ #

    def get_user_context(self, user_embedding: np.ndarray) -> Dict[int, float]:
        """
        Assign user to top-K clusters and return softmax weights.

        Implements Eq. (10) from the paper:
          w_{u,c} = softmax(<u, centroid_c> / τ')  for c in top-K clusters

        Args:
            user_embedding: raw feature vector for the user (will be normalised)
        Returns:
            Sparse dict {cluster_id: weight} with exactly K entries
        """
        if not self.is_built:
            self.build()

        # Normalise user embedding
        norm = np.linalg.norm(user_embedding)
        if norm > 0:
            user_emb = user_embedding / norm
        else:
            user_emb = user_embedding

        # Cosine similarities to all centroids
        sims = self.centroids @ user_emb  # shape (C,)

        # Select top-K clusters
        top_k_indices = np.argsort(sims)[::-1][:K_CLUSTERS_PER_USER]

        # Softmax weights with temperature τ'
        top_sims = sims[top_k_indices]
        # Numerical stability: subtract max before exp
        top_sims_scaled = top_sims / SOFTMAX_TEMP
        top_sims_scaled -= top_sims_scaled.max()
        exp_sims = np.exp(top_sims_scaled)
        weights = exp_sims / exp_sims.sum()

        return {int(cluster_id): float(w) for cluster_id, w in zip(top_k_indices, weights)}

    def get_candidate_items(self, context_weights: Dict[int, float]) -> List[str]:
        """
        Get the union of items from all triggered clusters.

        I_candidate = ⋃_{c ∈ supp(w_u)} I_c
        """
        seen: Set[str] = set()
        candidates: List[str] = []
        for cluster_id in context_weights:
            for item_id in self.cluster_items.get(cluster_id, []):
                if item_id not in seen:
                    seen.add(item_id)
                    candidates.append(item_id)
        return candidates

    def add_new_item(self, song_id: str, song_dict: dict):
        """
        Streaming update: add a new item to the graph without rebuilding.
        Assigns item to its nearest cluster(s).
        """
        if not self.is_built:
            return

        emb = normalize(get_song_feature_vector(song_dict).reshape(1, -1))[0]
        self.item_embeddings[song_id] = emb

        # Find nearest cluster
        sims = self.centroids @ emb
        top_clusters = np.argsort(sims)[::-1][:3]  # assign to top-3 clusters

        self.item_clusters[song_id] = []
        for c in top_clusters:
            c = int(c)
            # Add to cluster if it has room, else replace worst item
            if len(self.cluster_items[c]) < N_TOP_ITEMS_PER_CLUSTER:
                self.cluster_items[c].append(song_id)
            else:
                # Check if new item is better than current worst
                cluster_sims = [sims[c] if sid == song_id
                                else float(self.centroids[c] @ self.item_embeddings.get(sid, emb))
                                for sid in self.cluster_items[c]]
                worst_idx = int(np.argmin(cluster_sims))
                if sims[c] > cluster_sims[worst_idx]:
                    evicted = self.cluster_items[c][worst_idx]
                    self.cluster_items[c][worst_idx] = song_id
                    # Update reverse index for evicted item
                    if evicted in self.item_clusters and c in self.item_clusters[evicted]:
                        self.item_clusters[evicted].remove(c)

            self.item_clusters[song_id].append(c)

    def get_cluster_info(self) -> List[dict]:
        """Return cluster info for the API."""
        if not self.is_built:
            return []
        result = []
        for c in range(N_CLUSTERS):
            items = self.cluster_items.get(c, [])
            centroid = self.centroids[c].tolist() if self.centroids is not None else []
            # Infer dominant genre from top items
            from music_catalog import SONGS_DB
            genre_counts: Dict[str, int] = {}
            for sid in items[:10]:
                if sid in SONGS_DB:
                    g = SONGS_DB[sid]["genre"]
                    genre_counts[g] = genre_counts.get(g, 0) + 1
            dominant_genre = max(genre_counts, key=genre_counts.get) if genre_counts else "Mixed"
            result.append({
                "cluster_id": c,
                "n_items": len(items),
                "dominant_genre": dominant_genre,
                "top_items": items[:5],
                "centroid_norm": float(np.linalg.norm(self.centroids[c])) if self.centroids is not None else 1.0,
            })
        return result


# ── Module-level singleton ──────────────────────────────────────────────────
_graph = SparseGraph()


def get_graph() -> SparseGraph:
    if not _graph.is_built:
        _graph.build()
    return _graph
