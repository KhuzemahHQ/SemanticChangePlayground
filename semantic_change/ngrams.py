"""
Google Books Ngram Client & Historical Frequency Kinematics
===========================================================
Queries historical word frequency curves from Google Books Ngram Viewer (1800-2019),
computes decadal aggregates, velocity, growth rates, and frequency inflection points.
"""

from __future__ import annotations
import json
import os
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


CORPORA_MAP = {
    "en-2019": "English (2019)",
    "en-US-2019": "American English (2019)",
    "en-GB-2019": "British English (2019)",
    "en-fiction-2019": "English Fiction (2019)",
    "eng_2012": "English (2012 Historical)",
    "eng_fiction_2012": "English Fiction (2012)",
}


@dataclass
class DecadalFrequency:
    decade: int
    mean_frequency: float
    min_frequency: float
    max_frequency: float
    growth_rate: float  # Percentage change from previous decade
    velocity: float  # Absolute change per decade


@dataclass
class FrequencyTrajectory:
    ngram: str
    corpus: str
    start_year: int
    end_year: int
    smoothing: int
    annual_years: List[int]
    annual_frequencies: List[float]
    decadal: List[DecadalFrequency]
    peak_year: int
    peak_frequency: float
    total_change_ratio: float  # Frequency in end_decade / frequency in start_decade
    pattern: str  # Classification: 'Exponential Surge', 'Steady Decline', 'Resurgence', 'Stable', etc.

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ngram": self.ngram,
            "corpus": self.corpus,
            "corpus_name": CORPORA_MAP.get(self.corpus, self.corpus),
            "start_year": self.start_year,
            "end_year": self.end_year,
            "smoothing": self.smoothing,
            "annual_years": self.annual_years,
            "annual_frequencies": self.annual_frequencies,
            "decadal": [
                {
                    "decade": d.decade,
                    "mean_frequency": d.mean_frequency,
                    "min_frequency": d.min_frequency,
                    "max_frequency": d.max_frequency,
                    "growth_rate": d.growth_rate,
                    "velocity": d.velocity,
                }
                for d in self.decadal
            ],
            "peak_year": self.peak_year,
            "peak_frequency": self.peak_frequency,
            "total_change_ratio": self.total_change_ratio,
            "pattern": self.pattern,
        }


class GoogleBooksNgramClient:
    """Client for Google Books Ngram Viewer with decadal kinematic analysis."""

    BASE_URL = "https://books.google.com/ngrams/json"

    def __init__(self, cache_dir: Optional[str] = None):
        if cache_dir is None:
            cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "cache", "ngrams")
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_cache_path(self, query: str, corpus: str, start_year: int, end_year: int, smoothing: int) -> str:
        safe_query = re.sub(r"[^\w\-]", "_", query.lower())
        filename = f"{safe_query}_{corpus}_{start_year}_{end_year}_s{smoothing}.json"
        return os.path.join(self.cache_dir, filename)

    def fetch_ngram_raw(
        self,
        query: str,
        corpus: str = "en-2019",
        start_year: int = 1800,
        end_year: int = 2019,
        smoothing: int = 3,
        use_cache: bool = True,
    ) -> List[Dict[str, Any]]:
        """Fetch raw Ngram data from Google Books API or local cache."""
        cache_path = self._get_cache_path(query, corpus, start_year, end_year, smoothing)
        if use_cache and os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass

        params = {
            "content": query,
            "year_start": str(start_year),
            "year_end": str(end_year),
            "corpus": corpus,
            "smoothing": str(smoothing),
        }
        url = f"{self.BASE_URL}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                ),
                "Accept": "application/json",
            },
        )

        try:
            with urllib.request.urlopen(req, timeout=12) as response:
                payload = response.read().decode("utf-8")
                data = json.loads(payload)
                if use_cache and data:
                    with open(cache_path, "w", encoding="utf-8") as f:
                        json.dump(data, f)
                return data
        except Exception as e:
            # Check if we have stale cache
            if os.path.exists(cache_path):
                with open(cache_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            # If completely offline, return empty list or fallback
            return []

    def get_frequency_trajectory(
        self,
        word: str,
        corpus: str = "en-2019",
        start_year: int = 1800,
        end_year: int = 2019,
        smoothing: int = 3,
    ) -> FrequencyTrajectory:
        """Retrieve and compute comprehensive historical frequency trajectory and kinematics."""
        raw_list = self.fetch_ngram_raw(word, corpus, start_year, end_year, smoothing)
        
        years = list(range(start_year, end_year + 1))
        freqs: List[float] = []

        if raw_list and len(raw_list) > 0 and "timeseries" in raw_list[0]:
            ts = raw_list[0]["timeseries"]
            # Align length if discrepancy occurs
            if len(ts) >= len(years):
                freqs = ts[: len(years)]
            else:
                freqs = ts + [0.0] * (len(years) - len(ts))
        else:
            freqs = [0.0] * len(years)

        # Aggregate by decade
        decades: Dict[int, List[float]] = {}
        for y, f in zip(years, freqs):
            dec = (y // 10) * 10
            if dec not in decades:
                decades[dec] = []
            decades[dec].append(f)

        decadal_results: List[DecadalFrequency] = []
        sorted_decades = sorted(decades.keys())
        prev_mean = 0.0

        for idx, dec in enumerate(sorted_decades):
            vals = decades[dec]
            mean_f = sum(vals) / len(vals) if vals else 0.0
            min_f = min(vals) if vals else 0.0
            max_f = max(vals) if vals else 0.0
            growth = 0.0
            velocity = 0.0

            if idx > 0 and prev_mean > 0:
                growth = (mean_f - prev_mean) / prev_mean
                velocity = mean_f - prev_mean
            elif idx > 0 and prev_mean == 0:
                growth = 1.0 if mean_f > 0 else 0.0
                velocity = mean_f

            decadal_results.append(
                DecadalFrequency(
                    decade=dec,
                    mean_frequency=mean_f,
                    min_frequency=min_f,
                    max_frequency=max_f,
                    growth_rate=growth,
                    velocity=velocity,
                )
            )
            prev_mean = mean_f

        # Calculate peak
        peak_idx = 0
        peak_f = -1.0
        for i, f in enumerate(freqs):
            if f > peak_f:
                peak_f = f
                peak_idx = i
        peak_year = years[peak_idx] if years else start_year

        # Ratio between end decade and start decade
        start_dec_mean = decadal_results[0].mean_frequency if decadal_results else 0.0
        end_dec_mean = decadal_results[-1].mean_frequency if decadal_results else 0.0
        ratio = (end_dec_mean / start_dec_mean) if start_dec_mean > 1e-12 else (end_dec_mean * 1e6)

        # Pattern classification
        pattern = self._classify_pattern(decadal_results, peak_year)

        return FrequencyTrajectory(
            ngram=word,
            corpus=corpus,
            start_year=start_year,
            end_year=end_year,
            smoothing=smoothing,
            annual_years=years,
            annual_frequencies=freqs,
            decadal=decadal_results,
            peak_year=peak_year,
            peak_frequency=peak_f,
            total_change_ratio=ratio,
            pattern=pattern,
        )

    def _classify_pattern(self, decadal: List[DecadalFrequency], peak_year: int) -> str:
        """Classify historical frequency trajectory pattern."""
        if not decadal or len(decadal) < 3:
            return "Indeterminate"

        means = [d.mean_frequency for d in decadal]
        if max(means) < 1e-9:
            return "Ultra Low Frequency / Archival"

        first_third = sum(means[: len(means) // 3]) / (len(means) // 3)
        last_third = sum(means[-len(means) // 3 :]) / (len(means) // 3)
        mid_third = sum(means[len(means) // 3 : -len(means) // 3]) / max(1, len(means) - 2 * (len(means) // 3))

        if last_third > 3.0 * first_third and last_third > 1.5 * mid_third:
            return "Modern Exponential Surge"
        elif first_third > 3.0 * last_third and first_third > 1.5 * mid_third:
            return "Historical Archaisation / Decline"
        elif mid_third > 1.8 * first_third and mid_third > 1.8 * last_third:
            return f"Mid-Era Peak (c. {peak_year})"
        elif last_third > 2.0 * mid_third and first_third > 2.0 * mid_third:
            return "U-Shaped Resurgence"
        elif max(means) / (min(means) + 1e-9) < 2.5:
            return "Diachronic Frequency Stability"
        else:
            return f"Fluctuating Dynamic (Peak {peak_year})"
