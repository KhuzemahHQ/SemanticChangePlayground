"""
FastAPI Web Application & REST API
==================================
Serves the diachronic semantic change REST API and interactive web interface.
"""

from __future__ import annotations
import os
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from semantic_change.analyzer import DiachronicAnalyzer
from semantic_change.corpora import HISTORICAL_BENCHMARKS, SEMEVAL_2020_ENGLISH_TARGETS
from semantic_change.custom_corpus import CustomDiachronicCorpusAnalyzer
from semantic_change.ngrams import CORPORA_MAP

app = FastAPI(
    title="Diachronic Semantic Change Studio",
    description="Scientific tools to measure the semantic change of words over history using Google Books Ngrams and historical embeddings",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global analyzer instance
analyzer: Optional[DiachronicAnalyzer] = None


@app.on_event("startup")
def startup_event():
    global analyzer
    analyzer = DiachronicAnalyzer()


def get_analyzer() -> DiachronicAnalyzer:
    global analyzer
    if analyzer is None:
        analyzer = DiachronicAnalyzer()
    return analyzer


@app.get("/api/status")
def get_status() -> Dict[str, Any]:
    az = get_analyzer()
    space = az.embedding_space
    return {
        "status": "online",
        "embeddings_loaded": space is not None,
        "embedding_name": space.name if space else None,
        "decades": space.decades if space else [],
        "vocab_size": len(space.vocabs[space.decades[0]]) if space and space.decades else 0,
        "dimensions": space.dimensions if space else 0,
        "corpora": CORPORA_MAP,
        "benchmark_words": list(HISTORICAL_BENCHMARKS.keys()),
        "semeval_target_count": len(SEMEVAL_2020_ENGLISH_TARGETS),
    }


@app.get("/api/analyze")
def analyze_word(
    word: str = Query(..., description="Target word (e.g. gay, mouse, apple, cell)"),
    corpus: str = Query("en-2019", description="Google Books Ngram corpus"),
    start_year: int = Query(1800, description="Start year"),
    end_year: int = Query(2019, description="End year"),
    top_k: int = Query(8, description="Number of top nearest neighbors per decade"),
) -> Dict[str, Any]:
    az = get_analyzer()
    return az.analyze_word(word, corpus=corpus, start_year=start_year, end_year=end_year, top_k_neighbors=top_k)


@app.get("/api/compare")
def compare_words(
    words: str = Query(..., description="Comma-separated list of words (e.g. gay,mouse,apple)"),
    corpus: str = Query("en-2019", description="Google Books Ngram corpus"),
    start_year: int = Query(1800, description="Start year"),
    end_year: int = Query(2019, description="End year"),
) -> Dict[str, Any]:
    az = get_analyzer()
    word_list = [w.strip() for w in words.split(",") if w.strip()]
    if not word_list:
        raise HTTPException(status_code=400, detail="No words provided.")
    return az.compare_words(word_list, corpus=corpus, start_year=start_year, end_year=end_year)


@app.get("/api/trajectory")
def get_trajectory_2d(
    words: str = Query(..., description="Comma-separated list of words"),
) -> Dict[str, Any]:
    az = get_analyzer()
    word_list = [w.strip() for w in words.split(",") if w.strip()]
    return az.get_trajectory_2d(word_list)


@app.get("/api/benchmarks")
def get_benchmarks() -> Dict[str, Any]:
    return {
        "count": len(HISTORICAL_BENCHMARKS),
        "benchmarks": HISTORICAL_BENCHMARKS,
    }


@app.get("/api/semeval")
def get_semeval_evaluation() -> Dict[str, Any]:
    az = get_analyzer()
    return az.evaluate_semeval()


@app.get("/api/laws")
def get_laws_evaluation(sample_size: int = Query(500, description="Sample vocabulary size")) -> Dict[str, Any]:
    az = get_analyzer()
    return az.test_laws_of_semantic_change(sample_size=sample_size)


class CustomCorpusPayload(BaseModel):
    corpus_t1: str
    corpus_t2: str
    target_words: Optional[List[str]] = None
    label_t1: Optional[str] = "Historical Period 1"
    label_t2: Optional[str] = "Historical Period 2"
    window_size: Optional[int] = 5
    min_count: Optional[int] = 2
    dim: Optional[int] = 50


@app.post("/api/custom-corpus")
def process_custom_corpus(payload: CustomCorpusPayload) -> Dict[str, Any]:
    proc = CustomDiachronicCorpusAnalyzer(
        window_size=payload.window_size or 5,
        min_count=payload.min_count or 2,
        embedding_dim=payload.dim or 50,
    )
    return proc.analyze_two_corpora(
        corpus_t1=payload.corpus_t1,
        corpus_t2=payload.corpus_t2,
        target_words=payload.target_words,
        label_t1=payload.label_t1 or "Period 1",
        label_t2=payload.label_t2 or "Period 2",
    )


# Mount static directory
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
os.makedirs(static_dir, exist_ok=True)

app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
def serve_index():
    index_file = os.path.join(static_dir, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "Diachronic Semantic Change Studio API is running. UI building in progress."}
