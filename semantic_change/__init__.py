"""
Diachronic Semantic Change Toolkit & Research Playground
=========================================================
A comprehensive framework and interactive studio for measuring lexical semantic change
of words across history using Google Books Ngrams, HistWords diachronic embeddings,
SemEval benchmarks, and custom historical corpora.
"""

from .analyzer import DiachronicAnalyzer
from .ngrams import GoogleBooksNgramClient
from .embeddings import DiachronicEmbeddingSpace
from .metrics import SemanticChangeMetrics
from .corpora import HISTORICAL_BENCHMARKS, SEMEVAL_2020_ENGLISH_TARGETS

__version__ = "1.0.0"
__all__ = [
    "DiachronicAnalyzer",
    "GoogleBooksNgramClient",
    "DiachronicEmbeddingSpace",
    "SemanticChangeMetrics",
    "HISTORICAL_BENCHMARKS",
    "SEMEVAL_2020_ENGLISH_TARGETS",
]
