"""
Diachronic Vector Spaces & Orthogonal Procrustes Alignment
==========================================================
Manages multi-decadal vector spaces (1800s-2000s), nearest-neighbor queries,
dynamic Procrustes alignment across time slices, and fast vector retrieval.
"""

from __future__ import annotations
import glob
import os
import pickle
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np


class DiachronicEmbeddingSpace:
    """Manages sequential word embeddings aligned across historical decades."""

    def __init__(self, name: str = "HistWords"):
        self.name = name
        self.decades: List[int] = []
        self.matrices: Dict[int, np.ndarray] = {}
        self.vocabs: Dict[int, List[str]] = {}
        self.word_to_indices: Dict[int, Dict[str, int]] = {}
        self.dimensions: int = 0

    def add_decade(self, decade: int, vocab: List[str], vectors: np.ndarray, normalize: bool = True):
        """Add embedding matrix and vocabulary for a given decade."""
        if normalize:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            vectors = vectors / norms

        self.decades.append(decade)
        self.decades.sort()
        self.matrices[decade] = vectors.astype(np.float32)
        self.vocabs[decade] = vocab
        self.word_to_indices[decade] = {w.lower(): i for i, w in enumerate(vocab)}
        self.dimensions = vectors.shape[1]

    def has_word(self, word: str, decade: Optional[int] = None) -> bool:
        """Check if a word exists in vocabulary with valid non-zero representation."""
        word = word.lower()
        if decade is not None:
            if decade not in self.word_to_indices or word not in self.word_to_indices[decade]:
                return False
            idx = self.word_to_indices[decade][word]
            vec = self.matrices[decade][idx]
            return float(np.linalg.norm(vec)) > 0.05
        # Exists in any decade
        return any(self.has_word(word, d) for d in self.decades)

    def get_vector(self, word: str, decade: int) -> Optional[np.ndarray]:
        """Retrieve L2-normalized vector for a word in a specific decade."""
        word = word.lower()
        if decade not in self.word_to_indices or word not in self.word_to_indices[decade]:
            return None
        idx = self.word_to_indices[decade][word]
        vec = self.matrices[decade][idx]
        norm = float(np.linalg.norm(vec))
        if norm <= 0.05:
            return None
        return vec / norm

    def cosine_similarity(self, word1: str, word2: str, decade: int) -> Optional[float]:
        """Compute cosine similarity between two words in a single decade."""
        v1 = self.get_vector(word1, decade)
        v2 = self.get_vector(word2, decade)
        if v1 is None or v2 is None:
            return None
        return float(np.dot(v1, v2))

    def time_similarity(self, word1: str, word2: str) -> Dict[int, Optional[float]]:
        """Compute cosine similarity between two words across all available decades."""
        res: Dict[int, Optional[float]] = {}
        for d in self.decades:
            res[d] = self.cosine_similarity(word1, word2, d)
        return res

    def nearest_neighbors(self, word: str, decade: int, top_k: int = 10) -> List[Tuple[str, float]]:
        """Retrieve top-k nearest semantic neighbors for a word in a specific decade."""
        v = self.get_vector(word, decade)
        if v is None:
            return []

        mat = self.matrices[decade]
        vocab = self.vocabs[decade]
        sims = np.dot(mat, v)

        # Get top indices (excluding the query word itself)
        top_indices = np.argsort(-sims)[: top_k + 5]
        res: List[Tuple[str, float]] = []
        word_lower = word.lower()

        for idx in top_indices:
            w = vocab[idx]
            if w.lower() == word_lower:
                continue
            res.append((w, float(sims[idx])))
            if len(res) == top_k:
                break
        return res

    def get_all_nearest_neighbors_timeline(self, word: str, top_k: int = 6) -> Dict[int, List[Tuple[str, float]]]:
        """Get nearest neighbors for a word across all decades."""
        timeline: Dict[int, List[Tuple[str, float]]] = {}
        for d in self.decades:
            if self.has_word(word, d):
                timeline[d] = self.nearest_neighbors(word, d, top_k=top_k)
            else:
                timeline[d] = []
        return timeline

    def align_all_to_reference(self, reference_decade: int = 1990, anchor_words: Optional[List[str]] = None):
        """
        Align all decade vector spaces to a shared reference coordinate frame
        using Orthogonal Procrustes on high-frequency anchor vocabulary.
        """
        if reference_decade not in self.matrices:
            if not self.decades:
                return
            reference_decade = self.decades[-1]

        ref_mat = self.matrices[reference_decade]
        ref_dict = self.word_to_indices[reference_decade]

        for d in self.decades:
            if d == reference_decade:
                continue

            cur_mat = self.matrices[d]
            cur_dict = self.word_to_indices[d]

            # Find shared anchor words
            if anchor_words is not None:
                shared = [w for w in anchor_words if w in cur_dict and w in ref_dict]
            else:
                # Top shared words
                shared = [w for w in self.vocabs[d][:1000] if w in ref_dict]

            if len(shared) < 20:
                continue

            cur_indices = [cur_dict[w] for w in shared]
            ref_indices = [ref_dict[w] for w in shared]

            a = cur_mat[cur_indices]
            b = ref_mat[ref_indices]

            # Orthogonal Procrustes: min ||A W - B||_F s.t. W^T W = I
            # Solution: A^T B = U \Sigma V^T => W = U V^T
            m = np.dot(a.T, b)
            u, _, vt = np.linalg.svd(m)
            w = np.dot(u, vt)

            # Rotate current matrix
            rotated = np.dot(cur_mat, w)
            # Re-normalize
            norms = np.linalg.norm(rotated, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self.matrices[d] = (rotated / norms).astype(np.float32)

    @classmethod
    def load_from_histwords_directory(cls, dir_path: str, max_words_per_decade: int = 15000) -> Optional[DiachronicEmbeddingSpace]:
        """
        Load historical embeddings from a directory structured like Stanford HistWords
        containing <decade>-w.npy and <decade>-vocab.pkl.
        """
        if not os.path.exists(dir_path):
            return None

        # Check for sgns subdirectory
        sgns_path = os.path.join(dir_path, "sgns")
        if os.path.exists(sgns_path):
            dir_path = sgns_path

        npy_files = glob.glob(os.path.join(dir_path, "*-w.npy"))
        if not npy_files:
            return None

        space = cls(name="HistWords-Loaded")

        for fpath in sorted(npy_files):
            base = os.path.basename(fpath).replace("-w.npy", "")
            try:
                decade = int(base)
            except ValueError:
                continue

            vocab_path = os.path.join(dir_path, f"{decade}-vocab.pkl")
            if not os.path.exists(vocab_path):
                continue

            try:
                with open(vocab_path, "rb") as vf:
                    vocab = pickle.load(vf)
                mat = np.load(fpath, mmap_mode="r")

                if max_words_per_decade and len(vocab) > max_words_per_decade:
                    vocab = vocab[:max_words_per_decade]
                    mat = np.array(mat[:max_words_per_decade])
                else:
                    mat = np.array(mat)

                space.add_decade(decade, vocab, mat, normalize=True)
            except Exception as e:
                print(f"Warning: Failed to load decade {decade}: {e}")

        if space.decades:
            return space
        return None
