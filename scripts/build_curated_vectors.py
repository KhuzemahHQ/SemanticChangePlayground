"""
Build Curated Diachronic Vectors Archive
========================================
Extracts a calibrated high-density historical embedding space (1,500+ words across 20 decades)
from the extracted HistWords Stanford dataset and compresses it to data/curated_historical.npz.
"""

import glob
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pickle
import numpy as np

from semantic_change.corpora import (
    HISTORICAL_BENCHMARKS,
    SEMEVAL_2020_ENGLISH_TARGETS,
    ALL_ANCHOR_WORDS,
    VALENCE_MARKERS,
)

HISTWORDS_DIR = os.path.abspath("data/histwords/eng-fiction/sgns")
OUTPUT_PATH = os.path.abspath("data/curated_historical.npz")


def build_curated_archive(target_vocab_size: int = 2500):
    if not os.path.exists(HISTWORDS_DIR):
        print(f"Error: {HISTWORDS_DIR} does not exist.")
        return False

    npy_files = sorted(glob.glob(os.path.join(HISTWORDS_DIR, "*-w.npy")))
    if not npy_files:
        print(f"Error: No .npy files found in {HISTWORDS_DIR}")
        return False

    decades = []
    for f in npy_files:
        base = os.path.basename(f).replace("-w.npy", "")
        try:
            decades.append(int(base))
        except ValueError:
            pass
    decades.sort()
    print(f"Found {len(decades)} decades: {decades}")

    # Build target word list
    must_include = set()

    # 1. Benchmarks and their landmark words
    for w, meta in HISTORICAL_BENCHMARKS.items():
        must_include.add(w.lower())
        for lm in meta.get("landmarks_1800s", []):
            must_include.add(lm.lower())
        for lm in meta.get("landmarks_2000s", []):
            must_include.add(lm.lower())

    # 2. SemEval targets
    for key, meta in SEMEVAL_2020_ENGLISH_TARGETS.items():
        must_include.add(meta["lemma"].lower())

    # 3. Domain anchors
    for aw in ALL_ANCHOR_WORDS:
        must_include.add(aw.lower())

    # 4. Valence markers
    for vm in VALENCE_MARKERS["positive"] + VALENCE_MARKERS["negative"]:
        must_include.add(vm.lower())

    # Extra modern / tech / historical shift terms
    extra_terms = [
        "computer", "technology", "digital", "internet", "machine", "device", "power", "science",
        "politics", "society", "culture", "religion", "art", "music", "radio", "television",
        "telephone", "network", "web", "online", "stream", "cloud", "server", "data", "screen",
        "keyboard", "click", "hardware", "software", "program", "code", "file", "folder", "window",
        "memory", "chip", "battery", "wire", "signal", "channel", "media", "press", "news",
        "paper", "book", "letter", "message", "mail", "post", "ship", "train", "car", "steam",
        "engine", "flight", "pilot", "plane", "aircraft", "rocket", "space", "satellite", "orbit",
        "atom", "nuclear", "quantum", "physics", "biology", "dna", "gene", "virus", "vaccine",
        "medicine", "doctor", "hospital", "patient", "disease", "fever", "infection", "quarantine",
        "love", "hate", "peace", "war", "battle", "soldier", "army", "weapon", "gun", "sword",
        "money", "gold", "silver", "dollar", "bank", "market", "trade", "commerce", "capital",
        "work", "labor", "factory", "worker", "industry", "city", "town", "village", "country",
        "nation", "state", "government", "king", "queen", "president", "law", "court", "judge",
        "justice", "freedom", "liberty", "rights", "democracy", "citizen", "people", "man", "woman",
        "child", "friend", "enemy", "family", "house", "home", "room", "table", "chair", "bed",
        "food", "water", "bread", "meat", "fruit", "wine", "beer", "drink", "coffee", "tea",
        "road", "path", "street", "bridge", "river", "mountain", "sea", "ocean", "island", "forest",
        "tree", "flower", "grass", "animal", "horse", "dog", "cat", "bird", "fish", "lion", "wolf",
        "hound", "mouse", "rat", "bug", "insect", "spider", "snake", "cattle", "sheep", "pig",
        "day", "night", "morning", "evening", "year", "month", "hour", "time", "season", "spring",
        "summer", "autumn", "winter", "sun", "moon", "star", "sky", "wind", "rain", "storm", "snow",
        "fire", "ice", "earth", "stone", "iron", "steel", "glass", "wood", "cloth", "silk", "wool",
        "light", "dark", "shadow", "sound", "voice", "speech", "word", "language", "name", "mind",
        "thought", "soul", "spirit", "heart", "eye", "ear", "hand", "foot", "body", "head", "face",
        "good", "bad", "great", "small", "high", "low", "long", "short", "old", "new", "young",
        "cool", "hot", "cold", "warm", "sick", "wicked", "gay", "nice", "awful", "terrific", "silly",
        "guy", "peer", "token", "feed", "tag", "thread", "cookie", "amazon", "apple", "smart", "spam",
    ]
    for et in extra_terms:
        must_include.add(et.lower())

    print(f"Must-include vocabulary count: {len(must_include)}")

    # Load 1990 and 1800 vocabs to find common high-frequency words
    with open(os.path.join(HISTWORDS_DIR, "1990-vocab.pkl"), "rb") as f:
        vocab_1990 = [w.lower() for w in pickle.load(f)]
    with open(os.path.join(HISTWORDS_DIR, "1800-vocab.pkl"), "rb") as f:
        vocab_1800 = [w.lower() for w in pickle.load(f)]

    # Intersect high frequency words present in all vocabs
    common_high_freq = []
    set_1800 = set(vocab_1800[:8000])
    for w in vocab_1990[:8000]:
        if w in set_1800 and w not in must_include:
            common_high_freq.append(w)

    needed_more = max(0, target_vocab_size - len(must_include))
    curated_vocab = list(must_include) + common_high_freq[:needed_more]
    curated_vocab = sorted(list(set(curated_vocab)))
    print(f"Total curated vocabulary size: {len(curated_vocab)}")

    # For each decade, extract vectors for curated_vocab
    decades_matrices = []

    for d in decades:
        print(f"Extracting vectors for decade {d}...")
        w_path = os.path.join(HISTWORDS_DIR, f"{d}-w.npy")
        vocab_path = os.path.join(HISTWORDS_DIR, f"{d}-vocab.pkl")

        with open(vocab_path, "rb") as f:
            v_list = [w.lower() for w in pickle.load(f)]
        v_dict = {w: i for i, w in enumerate(v_list)}
        full_mat = np.load(w_path)
        dim = full_mat.shape[1]

        dec_mat = np.zeros((len(curated_vocab), dim), dtype=np.float32)

        found_count = 0
        for i, word in enumerate(curated_vocab):
            if word in v_dict:
                vec = full_mat[v_dict[word]]
                norm = np.linalg.norm(vec)
                if norm > 0:
                    dec_mat[i] = vec / norm
                found_count += 1
            else:
                # Fallback zero or close approximation
                pass

        print(f"  Decade {d}: found {found_count}/{len(curated_vocab)} words ({found_count/len(curated_vocab)*100:.1f}%)")
        decades_matrices.append(dec_mat)

    # Save to npz
    matrices_array = np.array(decades_matrices, dtype=np.float32)
    print(f"Saving curated archive to {OUTPUT_PATH} (Shape: {matrices_array.shape})...")
    np.savez_compressed(
        OUTPUT_PATH,
        decades=np.array(decades, dtype=np.int32),
        vocab=np.array(curated_vocab, dtype=object),
        matrices=matrices_array,
    )
    print(f"Success! Output file size: {os.path.getsize(OUTPUT_PATH) / (1024*1024):.2f} MB")
    return True


if __name__ == "__main__":
    build_curated_archive()
