"""
Custom Diachronic Corpus Processor
==================================
Allows researchers to upload or supply raw historical text corpora partitioned
by historical era (Period 1 vs Period 2), builds co-occurrence matrices, computes PPMI,
performs Truncated SVD, aligns representations via Orthogonal Procrustes,
and measures lexical semantic change on custom data.
"""

from __future__ import annotations
import collections
import math
import re
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds


class CustomDiachronicCorpusAnalyzer:
    """Trains and aligns diachronic embeddings from user-supplied historical texts."""

    def __init__(
        self,
        window_size: int = 5,
        min_count: int = 2,
        embedding_dim: int = 50,
    ):
        self.window_size = window_size
        self.min_count = min_count
        self.embedding_dim = embedding_dim

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Simple, fast lower-case tokenization."""
        return re.findall(r"\b[a-zA-Z]{2,}\b", text.lower())

    def _build_cooccurrence(
        self, tokens: List[str], vocab: List[str], word2idx: Dict[str, int]
    ) -> Tuple[csr_matrix, np.ndarray]:
        """Construct sliding-window co-occurrence matrix."""
        v_size = len(vocab)
        row_indices: List[int] = []
        col_indices: List[int] = []
        values: List[float] = []
        word_counts = np.zeros(v_size, dtype=np.float64)

        token_indices = [word2idx[t] for t in tokens if t in word2idx]
        n_tokens = len(token_indices)

        for i, center_idx in enumerate(token_indices):
            word_counts[center_idx] += 1
            start = max(0, i - self.window_size)
            end = min(n_tokens, i + self.window_size + 1)
            for j in range(start, end):
                if i == j:
                    continue
                context_idx = token_indices[j]
                # Distance weighting
                weight = 1.0 / abs(i - j)
                row_indices.append(center_idx)
                col_indices.append(context_idx)
                values.append(weight)

        cooc = csr_matrix(
            (values, (row_indices, col_indices)), shape=(v_size, v_size), dtype=np.float64
        )
        return cooc, word_counts

    def _compute_ppmi(self, cooc: csr_matrix, word_counts: np.ndarray) -> csr_matrix:
        """Compute Positive Pointwise Mutual Information (PPMI)."""
        cooc = cooc.tocoo()
        total_pairs = float(cooc.data.sum())
        if total_pairs == 0:
            return cooc.tocsr()

        p_w = word_counts / total_pairs
        ppmi_values: List[float] = []
        rows: List[int] = []
        cols: List[int] = []

        for r, c, val in zip(cooc.row, cooc.col, cooc.data):
            p_wc = val / total_pairs
            denominator = p_w[r] * p_w[c]
            if denominator > 0 and p_wc > 0:
                pmi = math.log2(p_wc / denominator)
                if pmi > 0:
                    ppmi_values.append(pmi)
                    rows.append(r)
                    cols.append(c)

        return csr_matrix(
            (ppmi_values, (rows, cols)), shape=cooc.shape, dtype=np.float32
        )

    def _embed_matrix(self, ppmi: csr_matrix, dim: int) -> np.ndarray:
        """Factorize PPMI matrix using Truncated SVD to obtain dense representations."""
        v_size = ppmi.shape[0]
        actual_dim = min(dim, v_size - 2)
        if actual_dim < 2:
            # Fallback to random normalized or identity
            m = np.random.randn(v_size, dim).astype(np.float32)
            norms = np.linalg.norm(m, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            return m / norms

        try:
            u, s, _ = svds(ppmi, k=actual_dim)
            # Embeddings = U * sqrt(Sigma)
            vecs = u * np.sqrt(s)
            # Pad if needed
            if actual_dim < dim:
                pad = np.zeros((v_size, dim - actual_dim), dtype=np.float32)
                vecs = np.hstack([vecs, pad])
        except Exception:
            vecs = np.random.randn(v_size, dim).astype(np.float32)

        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (vecs / norms).astype(np.float32)

    def analyze_two_corpora(
        self,
        corpus_t1: str,
        corpus_t2: str,
        target_words: Optional[List[str]] = None,
        label_t1: str = "Period 1",
        label_t2: str = "Period 2",
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Process two raw text historical corpora, align embeddings via Orthogonal Procrustes,
        and measure semantic displacement for target words.
        """
        tokens_1 = self.tokenize(corpus_t1)
        tokens_2 = self.tokenize(corpus_t2)

        counts_1 = collections.Counter(tokens_1)
        counts_2 = collections.Counter(tokens_2)

        # Shared vocabulary passing min_count in both
        min_c = self.min_count
        shared_vocab = [
            w for w, c1 in counts_1.items()
            if c1 >= min_c and counts_2.get(w, 0) >= min_c
        ]
        if len(shared_vocab) < 5 and min_c > 1:
            # Fallback to min_count = 1 for concise user sample texts
            shared_vocab = [
                w for w, c1 in counts_1.items()
                if c1 >= 1 and counts_2.get(w, 0) >= 1
            ]
        shared_vocab.sort()

        if len(shared_vocab) < 5:
            return {
                "error": "Insufficient shared vocabulary between the two text corpora.",
                "shared_vocab_count": len(shared_vocab),
                "tokens_1_count": len(tokens_1),
                "tokens_2_count": len(tokens_2),
            }

        word2idx = {w: i for i, w in enumerate(shared_vocab)}

        cooc_1, wc_1 = self._build_cooccurrence(tokens_1, shared_vocab, word2idx)
        cooc_2, wc_2 = self._build_cooccurrence(tokens_2, shared_vocab, word2idx)

        ppmi_1 = self._compute_ppmi(cooc_1, wc_1)
        ppmi_2 = self._compute_ppmi(cooc_2, wc_2)

        vecs_1 = self._embed_matrix(ppmi_1, self.embedding_dim)
        vecs_2 = self._embed_matrix(ppmi_2, self.embedding_dim)

        # Orthogonal Procrustes Alignment: align vecs_1 to vecs_2
        m = np.dot(vecs_1.T, vecs_2)
        u, _, vt = np.linalg.svd(m)
        rot = np.dot(u, vt)
        vecs_1_aligned = np.dot(vecs_1, rot)
        norms = np.linalg.norm(vecs_1_aligned, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vecs_1_aligned = (vecs_1_aligned / norms).astype(np.float32)

        # Select target words
        if not target_words:
            # Pick top words by frequency
            target_words = [w for w in shared_vocab if counts_1[w] >= 5 and counts_2[w] >= 5][:20]
        else:
            target_words = [w.lower() for w in target_words if w.lower() in word2idx]

        results: List[Dict[str, Any]] = []

        for tw in target_words:
            idx = word2idx[tw]
            v1 = vecs_1_aligned[idx]
            v2 = vecs_2[idx]

            cos_sim = float(np.dot(v1, v2))
            disp = float(max(0.0, 1.0 - cos_sim))

            # Neighbors in T1
            sims_1 = np.dot(vecs_1_aligned, v1)
            top_1_indices = np.argsort(-sims_1)[: top_k + 2]
            nn_1 = [
                (shared_vocab[i], float(sims_1[i]))
                for i in top_1_indices
                if i != idx
            ][:top_k]

            # Neighbors in T2
            sims_2 = np.dot(vecs_2, v2)
            top_2_indices = np.argsort(-sims_2)[: top_k + 2]
            nn_2 = [
                (shared_vocab[i], float(sims_2[i]))
                for i in top_2_indices
                if i != idx
            ][:top_k]

            # Jaccard turnover
            set1 = set(w for w, _ in nn_1)
            set2 = set(w for w, _ in nn_2)
            union = len(set1 | set2)
            jaccard = 1.0 - (len(set1 & set2) / union) if union > 0 else 0.0

            results.append({
                "word": tw,
                "displacement": round(disp, 4),
                "cosine_similarity": round(cos_sim, 4),
                "frequency_t1": counts_1[tw],
                "frequency_t2": counts_2[tw],
                "neighbors_t1": nn_1,
                "neighbors_t2": nn_2,
                "neighborhood_turnover": round(jaccard, 3),
            })

        # Sort by displacement descending
        results.sort(key=lambda x: x["displacement"], reverse=True)

        return {
            "label_t1": label_t1,
            "label_t2": label_t2,
            "tokens_count_t1": len(tokens_1),
            "tokens_count_t2": len(tokens_2),
            "shared_vocab_size": len(shared_vocab),
            "target_count": len(results),
            "words": results,
        }
