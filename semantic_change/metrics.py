"""
Diachronic Semantic Change Metrics & Scientific Measurement
============================================================
Implements:
1. Cosine displacement & cumulative drift
2. Semantic velocity & acceleration
3. Neighborhood turnover (Jaccard & Rank Overlap)
4. Second-order anchor drift (CDCS - Contextual Drift Cosine Similarity)
5. Semantic shift taxonomy: Amelioration, Pejoration, Broadening, Narrowing, Domain Transfer
6. Statistical validation for Hamilton's Laws of Semantic Change
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .corpora import DOMAIN_ANCHORS, VALENCE_MARKERS, ALL_ANCHOR_WORDS


@dataclass
class ShiftClassification:
    primary_category: str  # e.g., 'Pejoration', 'Amelioration', 'Broadening', 'Narrowing', 'Domain Shift'
    confidence: float
    valence_shift: float  # Positive = amelioration, Negative = pejoration
    valence_start: float
    valence_end: float
    dispersion_shift: float  # Positive = broadening, Negative = narrowing
    top_domains_start: List[str]
    top_domains_end: List[str]
    explanation: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "primary_category": self.primary_category,
            "confidence": round(self.confidence, 3),
            "valence_shift": round(self.valence_shift, 4),
            "valence_start": round(self.valence_start, 4),
            "valence_end": round(self.valence_end, 4),
            "dispersion_shift": round(self.dispersion_shift, 4),
            "top_domains_start": self.top_domains_start,
            "top_domains_end": self.top_domains_end,
            "explanation": self.explanation,
        }


@dataclass
class SemanticMeasurement:
    word: str
    decades: List[int]
    total_displacement: float  # Cosine distance from first decade to last decade
    step_displacements: Dict[int, float]  # Distance from t to t+1
    cumulative_drift: Dict[int, float]  # Distance from base decade to t
    velocities: Dict[int, float]  # Displacement / delta_t
    accelerations: Dict[int, float]  # delta_v / delta_t
    peak_velocity_decade: int
    peak_velocity: float
    neighborhood_turnovers: Dict[int, float]  # Jaccard distance of top-k neighbors
    mean_neighborhood_turnover: float
    second_order_drift: float  # CDCS against invariant anchor words
    classification: ShiftClassification

    def to_dict(self) -> Dict[str, Any]:
        return {
            "word": self.word,
            "decades": self.decades,
            "total_displacement": round(self.total_displacement, 4),
            "step_displacements": {d: round(v, 4) for d, v in self.step_displacements.items()},
            "cumulative_drift": {d: round(v, 4) for d, v in self.cumulative_drift.items()},
            "velocities": {d: round(v, 4) for d, v in self.velocities.items()},
            "accelerations": {d: round(v, 4) for d, v in self.accelerations.items()},
            "peak_velocity_decade": self.peak_velocity_decade,
            "peak_velocity": round(self.peak_velocity, 4),
            "neighborhood_turnovers": {d: round(v, 4) for d, v in self.neighborhood_turnovers.items()},
            "mean_neighborhood_turnover": round(self.mean_neighborhood_turnover, 4),
            "second_order_drift": round(self.second_order_drift, 4),
            "classification": self.classification.to_dict(),
        }


class SemanticChangeMetrics:
    """Scientific calculator for diachronic semantic change metrics."""

    @staticmethod
    def cosine_distance(v1: np.ndarray, v2: np.ndarray) -> float:
        """Compute cosine distance between two vectors: 1 - cos(v1, v2)."""
        dot = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 1.0
        cos_sim = float(dot / (norm1 * norm2))
        cos_sim = max(-1.0, min(1.0, cos_sim))
        return float(1.0 - cos_sim)

    @staticmethod
    def jaccard_distance(set1: set, set2: set) -> float:
        """Compute Jaccard distance between two sets: 1 - |A ∩ B| / |A ∪ B|."""
        if not set1 and not set2:
            return 0.0
        union = len(set1 | set2)
        if union == 0:
            return 0.0
        intersection = len(set1 & set2)
        return float(1.0 - (intersection / union))

    @classmethod
    def compute_second_order_drift(
        cls,
        space: Any,
        word: str,
        start_decade: int,
        end_decade: int,
        anchor_words: Optional[List[str]] = None,
    ) -> float:
        """
        Compute alignment-free second-order drift (CDCS - Contextual Drift Cosine Similarity).
        Word profile is its vector of similarities to invariant anchor words at each time point.
        """
        if anchor_words is None:
            anchor_words = ALL_ANCHOR_WORDS

        v_start = space.get_vector(word, start_decade)
        v_end = space.get_vector(word, end_decade)
        if v_start is None or v_end is None:
            return 0.0

        profile_start: List[float] = []
        profile_end: List[float] = []

        for aw in anchor_words:
            aw_start = space.get_vector(aw, start_decade)
            aw_end = space.get_vector(aw, end_decade)
            if aw_start is not None and aw_end is not None:
                profile_start.append(float(np.dot(v_start, aw_start)))
                profile_end.append(float(np.dot(v_end, aw_end)))

        if len(profile_start) < 5:
            return 0.0

        p1 = np.array(profile_start)
        p2 = np.array(profile_end)
        return cls.cosine_distance(p1, p2)

    @classmethod
    def classify_semantic_shift(
        cls,
        space: Any,
        word: str,
        start_decade: int,
        end_decade: int,
        displacement: float,
    ) -> ShiftClassification:
        """
        Classify the type of semantic shift:
        - Amelioration (valence improvement)
        - Pejoration (valence degradation)
        - Broadening (dispersion increase)
        - Narrowing (dispersion decrease)
        - Domain Transfer / Technological Transition
        """
        v_start = space.get_vector(word, start_decade)
        v_end = space.get_vector(word, end_decade)

        if v_start is None or v_end is None:
            return ShiftClassification(
                primary_category="Stable / Unchanged",
                confidence=0.0,
                valence_shift=0.0,
                valence_start=0.0,
                valence_end=0.0,
                dispersion_shift=0.0,
                top_domains_start=[],
                top_domains_end=[],
                explanation="Insufficient historical embedding representations.",
            )

        # 1. Valence Shift (Positive vs Negative Lexicon)
        pos_sims_start = [
            float(np.dot(v_start, space.get_vector(w, start_decade)))
            for w in VALENCE_MARKERS["positive"]
            if space.get_vector(w, start_decade) is not None
        ]
        neg_sims_start = [
            float(np.dot(v_start, space.get_vector(w, start_decade)))
            for w in VALENCE_MARKERS["negative"]
            if space.get_vector(w, start_decade) is not None
        ]
        pos_sims_end = [
            float(np.dot(v_end, space.get_vector(w, end_decade)))
            for w in VALENCE_MARKERS["positive"]
            if space.get_vector(w, end_decade) is not None
        ]
        neg_sims_end = [
            float(np.dot(v_end, space.get_vector(w, end_decade)))
            for w in VALENCE_MARKERS["negative"]
            if space.get_vector(w, end_decade) is not None
        ]

        val_start = (np.mean(pos_sims_start) if pos_sims_start else 0.0) - (np.mean(neg_sims_start) if neg_sims_start else 0.0)
        val_end = (np.mean(pos_sims_end) if pos_sims_end else 0.0) - (np.mean(neg_sims_end) if neg_sims_end else 0.0)
        valence_shift = float(val_end - val_start)

        # 2. Neighborhood Dispersion (Broadening vs Narrowing)
        nn_start = [w for w, _ in space.nearest_neighbors(word, start_decade, top_k=8)]
        nn_end = [w for w, _ in space.nearest_neighbors(word, end_decade, top_k=8)]

        def get_inter_similarity(words: List[str], decade: int) -> float:
            vecs = [space.get_vector(w, decade) for w in words if space.get_vector(w, decade) is not None]
            if len(vecs) < 2:
                return 0.0
            mat = np.array(vecs)
            sim_matrix = np.dot(mat, mat.T)
            # average off-diagonal
            n = len(vecs)
            return float((np.sum(sim_matrix) - n) / (n * (n - 1)))

        cohesion_start = get_inter_similarity(nn_start, start_decade)
        cohesion_end = get_inter_similarity(nn_end, end_decade)
        # Higher cohesion => narrower/specialized. Lower cohesion => broader/generalized.
        dispersion_shift = float(cohesion_start - cohesion_end)

        # 3. Domain Association Shift
        domain_scores_start: Dict[str, float] = {}
        domain_scores_end: Dict[str, float] = {}
        for dom, terms in DOMAIN_ANCHORS.items():
            s_start = [
                float(np.dot(v_start, space.get_vector(t, start_decade)))
                for t in terms
                if space.get_vector(t, start_decade) is not None
            ]
            s_end = [
                float(np.dot(v_end, space.get_vector(t, end_decade)))
                for t in terms
                if space.get_vector(t, end_decade) is not None
            ]
            domain_scores_start[dom] = float(np.mean(s_start)) if s_start else 0.0
            domain_scores_end[dom] = float(np.mean(s_end)) if s_end else 0.0

        sorted_dom_start = sorted(domain_scores_start.items(), key=lambda x: x[1], reverse=True)
        sorted_dom_end = sorted(domain_scores_end.items(), key=lambda x: x[1], reverse=True)

        top_dom_start = [k for k, _ in sorted_dom_start[:2]]
        top_dom_end = [k for k, _ in sorted_dom_end[:2]]

        # Determine Primary Category
        if displacement < 0.15:
            cat = "Semantic Stability"
            conf = 0.90
            exp = f"'{word}' exhibited high semantic stability (cosine distance {displacement:.3f}) over history."
        elif abs(valence_shift) > 0.045:
            if valence_shift > 0:
                cat = "Amelioration"
                conf = min(0.95, 0.5 + abs(valence_shift) * 4.0)
                exp = f"'{word}' underwent amelioration: connotation shifted toward positive evaluation (valence delta {valence_shift:+.3f})."
            else:
                cat = "Pejoration"
                conf = min(0.95, 0.5 + abs(valence_shift) * 4.0)
                exp = f"'{word}' underwent pejoration: connotation degraded toward negative evaluation (valence delta {valence_shift:+.3f})."
        elif abs(dispersion_shift) > 0.06:
            if dispersion_shift > 0:
                cat = "Broadening (Generalization)"
                conf = min(0.92, 0.5 + abs(dispersion_shift) * 3.5)
                exp = f"'{word}' underwent broadening: its semantic neighborhood expanded across diverse, less tightly clustered contexts."
            else:
                cat = "Narrowing (Specialization)"
                conf = min(0.92, 0.5 + abs(dispersion_shift) * 3.5)
                exp = f"'{word}' underwent narrowing: its semantic usage became concentrated around a specialized domain cluster."
        else:
            cat = "Metaphorical / Domain Transfer"
            conf = min(0.95, 0.4 + displacement * 1.2)
            exp = (
                f"'{word}' underwent significant domain transfer (displacement {displacement:.3f}), "
                f"transitioning semantic affinity across historical eras."
            )

        return ShiftClassification(
            primary_category=cat,
            confidence=conf,
            valence_shift=valence_shift,
            valence_start=val_start,
            valence_end=val_end,
            dispersion_shift=dispersion_shift,
            top_domains_start=top_dom_start,
            top_domains_end=top_dom_end,
            explanation=exp,
        )

    @classmethod
    def measure_word(cls, space: Any, word: str, top_k_neighbors: int = 10) -> Optional[SemanticMeasurement]:
        """Compute full suite of diachronic semantic change metrics for a word across all available decades."""
        word = word.lower()
        decades = [d for d in space.decades if space.has_word(word, d)]
        if len(decades) < 2:
            return None

        # Base vectors
        v_base = space.get_vector(word, decades[0])
        v_end = space.get_vector(word, decades[-1])
        if v_base is None or v_end is None:
            return None

        total_displacement = cls.cosine_distance(v_base, v_end)

        step_displacements: Dict[int, float] = {}
        cumulative_drift: Dict[int, float] = {}
        velocities: Dict[int, float] = {}
        accelerations: Dict[int, float] = {}

        prev_decade = decades[0]
        prev_vec = v_base
        prev_velocity = 0.0

        peak_v = -1.0
        peak_v_decade = decades[0]

        for i, d in enumerate(decades):
            cur_vec = space.get_vector(word, d)
            if cur_vec is None:
                continue

            # Cumulative drift from base decade
            drift = cls.cosine_distance(v_base, cur_vec)
            cumulative_drift[d] = drift

            if i > 0:
                delta_t = (d - prev_decade) / 10.0  # Normalized in decades
                step_dist = cls.cosine_distance(prev_vec, cur_vec)
                step_displacements[d] = step_dist

                velocity = (step_dist / delta_t) if delta_t > 0 else step_dist
                velocities[d] = velocity

                if velocity > peak_v:
                    peak_v = velocity
                    peak_v_decade = d

                accel = (velocity - prev_velocity) / delta_t if delta_t > 0 else 0.0
                accelerations[d] = accel

                prev_velocity = velocity

            prev_decade = d
            prev_vec = cur_vec

        # Neighborhood Turnover (Jaccard Distance)
        turnovers: Dict[int, float] = {}
        prev_nn: Optional[set] = None

        for d in decades:
            nn_list = space.nearest_neighbors(word, d, top_k=top_k_neighbors)
            nn_set = set(w.lower() for w, _ in nn_list)
            if prev_nn is not None:
                j_dist = cls.jaccard_distance(prev_nn, nn_set)
                turnovers[d] = j_dist
            prev_nn = nn_set

        mean_turnover = float(np.mean(list(turnovers.values()))) if turnovers else 0.0

        # Second-order drift
        second_order = cls.compute_second_order_drift(space, word, decades[0], decades[-1])

        # Semantic shift classification
        classification = cls.classify_semantic_shift(
            space, word, decades[0], decades[-1], total_displacement
        )

        return SemanticMeasurement(
            word=word,
            decades=decades,
            total_displacement=total_displacement,
            step_displacements=step_displacements,
            cumulative_drift=cumulative_drift,
            velocities=velocities,
            accelerations=accelerations,
            peak_velocity_decade=peak_v_decade,
            peak_velocity=peak_v if peak_v >= 0 else 0.0,
            neighborhood_turnovers=turnovers,
            mean_neighborhood_turnover=mean_turnover,
            second_order_drift=second_order,
            classification=classification,
        )
