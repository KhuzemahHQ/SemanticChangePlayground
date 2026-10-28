"""
Diachronic Semantic Analyzer (Unified Pipeline)
==============================================
The main entrypoint and high-level orchestrator. Integrates Google Books Ngrams,
historical diachronic embeddings, kinematic metrics, shift classification,
multi-word comparisons, 2D trajectory projections, and SemEval-2020 benchmarking.
"""

from __future__ import annotations
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import spearmanr, pearsonr

from .corpora import HISTORICAL_BENCHMARKS, SEMEVAL_2020_ENGLISH_TARGETS, ALL_ANCHOR_WORDS
from .embeddings import DiachronicEmbeddingSpace
from .metrics import SemanticChangeMetrics, SemanticMeasurement
from .ngrams import GoogleBooksNgramClient, FrequencyTrajectory


class DiachronicAnalyzer:
    """Unified analyzer for lexical semantic change across history."""

    def __init__(
        self,
        embeddings_path: Optional[str] = None,
        ngram_cache_dir: Optional[str] = None,
    ):
        self.ngram_client = GoogleBooksNgramClient(cache_dir=ngram_cache_dir)
        self.embedding_space: Optional[DiachronicEmbeddingSpace] = None

        # Resolve embeddings
        if embeddings_path and os.path.exists(embeddings_path):
            self.load_embeddings(embeddings_path)
        else:
            self._auto_load_embeddings()

    def _auto_load_embeddings(self):
        """Auto-detect downloaded HistWords or bundled curated vectors."""
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # Check standard HistWords download location
        histwords_dir = os.path.join(base_dir, "data", "histwords", "eng-fiction")
        if os.path.exists(histwords_dir):
            loaded = DiachronicEmbeddingSpace.load_from_histwords_directory(histwords_dir)
            if loaded:
                self.embedding_space = loaded
                return

        # Check curated npz
        curated_npz = os.path.join(base_dir, "data", "curated_historical.npz")
        if os.path.exists(curated_npz):
            self._load_curated_npz(curated_npz)

    def load_embeddings(self, path: str):
        """Load embeddings from a user-specified path."""
        if path.endswith(".npz"):
            self._load_curated_npz(path)
        else:
            loaded = DiachronicEmbeddingSpace.load_from_histwords_directory(path)
            if loaded:
                self.embedding_space = loaded

    def _load_curated_npz(self, npz_path: str):
        """Load pre-aligned curated historical embedding archive."""
        data = np.load(npz_path, allow_pickle=True)
        decades = [int(d) for d in data["decades"]]
        vocab = [str(w) for w in data["vocab"]]
        matrices = data["matrices"]  # Shape: (len(decades), len(vocab), dim)

        space = DiachronicEmbeddingSpace(name="Curated-HistWords")
        for i, dec in enumerate(decades):
            space.add_decade(dec, vocab, matrices[i], normalize=True)
        self.embedding_space = space

    def analyze_word(
        self,
        word: str,
        corpus: str = "en-2019",
        start_year: int = 1800,
        end_year: int = 2019,
        top_k_neighbors: int = 8,
    ) -> Dict[str, Any]:
        """
        Comprehensive diachronic analysis of a single word:
        - Live Google Books Ngram frequency trajectory (1800-2019)
        - Vector space displacement, kinematics, and drift
        - Nearest neighbor evolution across decades
        - Semantic shift classification (Amelioration, Pejoration, Domain Transfer, etc.)
        - Curated historical etymological context (if benchmark word)
        """
        word_clean = word.strip().lower()

        # 1. Google Books Ngram Frequency
        freq_traj = self.ngram_client.get_frequency_trajectory(
            word_clean, corpus=corpus, start_year=start_year, end_year=end_year
        )

        # 2. Semantic Displacement & Vector Metrics
        measurement: Optional[SemanticMeasurement] = None
        neighbors_timeline: Dict[int, List[Dict[str, Any]]] = {}

        if self.embedding_space and self.embedding_space.has_word(word_clean):
            measurement = SemanticChangeMetrics.measure_word(
                self.embedding_space, word_clean, top_k_neighbors=top_k_neighbors
            )
            raw_timeline = self.embedding_space.get_all_nearest_neighbors_timeline(
                word_clean, top_k=top_k_neighbors
            )
            neighbors_timeline = {
                d: [{"word": w, "similarity": round(sim, 3)} for w, sim in nns]
                for d, nns in raw_timeline.items()
            }

        # 3. Curated benchmark metadata
        benchmark_meta = HISTORICAL_BENCHMARKS.get(word_clean)

        return {
            "word": word_clean,
            "frequency_trajectory": freq_traj.to_dict(),
            "semantic_measurement": measurement.to_dict() if measurement else None,
            "neighbors_timeline": neighbors_timeline,
            "benchmark_metadata": benchmark_meta,
            "has_embeddings": measurement is not None,
            "embedding_dataset": self.embedding_space.name if self.embedding_space else "None",
        }

    def compare_words(
        self,
        words: List[str],
        corpus: str = "en-2019",
        start_year: int = 1800,
        end_year: int = 2019,
    ) -> Dict[str, Any]:
        """
        Side-by-side comparison of multiple words:
        - Frequency comparisons over 200 years
        - Cumulative drift trajectories
        - Pairwise cross-word semantic similarity over time
        """
        words = [w.strip().lower() for w in words if w.strip()]
        words_data: List[Dict[str, Any]] = []

        for w in words:
            analysis = self.analyze_word(w, corpus=corpus, start_year=start_year, end_year=end_year)
            words_data.append(analysis)

        # Pairwise time similarities
        pairwise: Dict[str, Dict[int, Optional[float]]] = {}
        if self.embedding_space:
            for i in range(len(words)):
                for j in range(i + 1, len(words)):
                    w1, w2 = words[i], words[j]
                    key = f"{w1} ↔ {w2}"
                    sim_dict = self.embedding_space.time_similarity(w1, w2)
                    pairwise[key] = {
                        d: (round(v, 4) if v is not None else None) for d, v in sim_dict.items()
                    }

        return {
            "words": words,
            "analyses": words_data,
            "pairwise_similarities": pairwise,
        }

    def get_trajectory_2d(self, words: List[str]) -> Dict[str, Any]:
        """
        Project words across historical decades into a shared 2D semantic plane
        using Global Principal Component Analysis (PCA) on normalized vectors.
        """
        if not self.embedding_space:
            return {"error": "Embedding space not loaded."}

        words = [w.strip().lower() for w in words if self.embedding_space.has_word(w.strip().lower())]
        if not words:
            return {"error": "None of the specified words exist in the embedding vocabulary."}

        decades = self.embedding_space.decades
        all_vecs: List[np.ndarray] = []
        point_labels: List[Tuple[str, int]] = []

        for w in words:
            for d in decades:
                v = self.embedding_space.get_vector(w, d)
                if v is not None:
                    all_vecs.append(v)
                    point_labels.append((w, d))

        if len(all_vecs) < 3:
            return {"error": "Too few historical vector samples for 2D projection."}

        mat = np.array(all_vecs)
        # Center the data
        mean = np.mean(mat, axis=0)
        centered = mat - mean

        # SVD for top 2 principal components
        u, s, _ = np.linalg.svd(centered, full_matrices=False)
        coords_2d = u[:, :2] * s[:2]

        trajectories: Dict[str, List[Dict[str, Any]]] = {w: [] for w in words}
        for (w, d), (x, y) in zip(point_labels, coords_2d):
            trajectories[w].append({
                "decade": d,
                "x": round(float(x), 4),
                "y": round(float(y), 4),
            })

        return {
            "decades": decades,
            "trajectories": trajectories,
            "words": words,
        }

    def evaluate_semeval(self) -> Dict[str, Any]:
        """
        Evaluate the embedding space against the official SemEval-2020 Task 1 English benchmark:
        - Computes Spearman rank correlation rho on Subtask 2 (graded change)
        - Computes binary classification metrics (Accuracy, Precision, Recall, F1) on Subtask 1
        """
        if not self.embedding_space:
            return {"error": "Embedding space not loaded."}

        decades = self.embedding_space.decades
        if len(decades) < 2:
            return {"error": "Insufficient decades for historical comparison."}

        # Compare 1810-1860s (approx 1830s) against 1960-2010s (approx 1990s)
        early_decades = [d for d in decades if d <= 1860]
        late_decades = [d for d in decades if d >= 1960]

        start_d = early_decades[len(early_decades) // 2] if early_decades else decades[0]
        end_d = late_decades[len(late_decades) // 2] if late_decades else decades[-1]

        evaluated: List[Dict[str, Any]] = []
        model_scores: List[float] = []
        gold_scores: List[float] = []
        binary_preds: List[int] = []
        binary_golds: List[int] = []

        for target_key, item in SEMEVAL_2020_ENGLISH_TARGETS.items():
            lemma = item["lemma"]
            if not self.embedding_space.has_word(lemma, start_d) or not self.embedding_space.has_word(lemma, end_d):
                continue

            v1 = self.embedding_space.get_vector(lemma, start_d)
            v2 = self.embedding_space.get_vector(lemma, end_d)
            if v1 is None or v2 is None:
                continue

            disp = SemanticChangeMetrics.cosine_distance(v1, v2)
            evaluated.append({
                "target": target_key,
                "lemma": lemma,
                "pos": item["pos"],
                "predicted_displacement": round(disp, 4),
                "gold_graded_score": item["graded_gold"],
                "gold_binary_label": item["binary_gold"],
                "category": item["category"],
            })

            model_scores.append(disp)
            gold_scores.append(item["graded_gold"])
            binary_golds.append(item["binary_gold"])

        if len(evaluated) < 5:
            return {
                "error": "Too few SemEval target words found in the embedding vocabulary.",
                "evaluated_count": len(evaluated),
            }

        # Spearman rank correlation
        spearman_rho, spearman_p = spearmanr(model_scores, gold_scores)

        # Threshold for binary prediction (e.g. median displacement)
        median_disp = float(np.median(model_scores))
        binary_preds = [1 if s >= median_disp else 0 for s in model_scores]

        # Binary metrics
        tp = sum(1 for p, g in zip(binary_preds, binary_golds) if p == 1 and g == 1)
        fp = sum(1 for p, g in zip(binary_preds, binary_golds) if p == 1 and g == 0)
        tn = sum(1 for p, g in zip(binary_preds, binary_golds) if p == 0 and g == 0)
        fn = sum(1 for p, g in zip(binary_preds, binary_golds) if p == 0 and g == 1)

        accuracy = (tp + tn) / max(1, len(binary_preds))
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-9, precision + recall)

        # Sort evaluated by predicted displacement descending
        evaluated.sort(key=lambda x: x["predicted_displacement"], reverse=True)

        return {
            "start_decade": start_d,
            "end_decade": end_d,
            "targets_evaluated": len(evaluated),
            "spearman_rank_correlation": round(float(spearman_rho), 4),
            "spearman_p_value": float(spearman_p),
            "binary_classification": {
                "accuracy": round(accuracy, 4),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1_score": round(f1, 4),
                "threshold_displacement": round(median_disp, 4),
            },
            "targets": evaluated,
        }

    def test_laws_of_semantic_change(self, sample_size: int = 500) -> Dict[str, Any]:
        """
        Empirically test Hamilton et al.'s Law of Conformity:
        Semantic displacement scales inversely with word frequency: r(w) ~ f(w)^-alpha
        Computes Pearson and Spearman correlation between log-frequency and semantic displacement.
        """
        if not self.embedding_space:
            return {"error": "Embedding space not loaded."}

        decades = self.embedding_space.decades
        if len(decades) < 2:
            return {"error": "Insufficient decades."}

        d_start, d_end = decades[0], decades[-1]
        shared_words = [
            w for w in self.embedding_space.vocabs[d_start]
            if self.embedding_space.has_word(w, d_end)
        ][:sample_size]

        ranks: List[float] = []
        displacements: List[float] = []

        for idx, w in enumerate(shared_words):
            v1 = self.embedding_space.get_vector(w, d_start)
            v2 = self.embedding_space.get_vector(w, d_end)
            if v1 is None or v2 is None:
                continue

            disp = SemanticChangeMetrics.cosine_distance(v1, v2)
            # Rank in vocabulary approximates log-frequency (Zipf's law)
            log_freq_proxy = -np.log(idx + 1)
            ranks.append(log_freq_proxy)
            displacements.append(disp)

        spearman_rho, s_p = spearmanr(ranks, displacements)
        pearson_r, p_p = pearsonr(ranks, displacements)

        return {
            "law": "Law of Conformity (Hamilton et al., 2016)",
            "description": "More frequent words change their meaning more slowly.",
            "sample_words_evaluated": len(displacements),
            "spearman_rho": round(float(spearman_rho), 4),
            "spearman_p_value": float(s_p),
            "pearson_r": round(float(pearson_r), 4),
            "pearson_p_value": float(p_p),
            "confirmed": bool(pearson_r < 0 and p_p < 0.05),
        }
