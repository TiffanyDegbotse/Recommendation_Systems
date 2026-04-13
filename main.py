# AI-assisted code generation (Claude, Anthropic) – https://claude.ai
"""
main.py
=======
Command-line interface for the Visual Vibe audio mood recommender.

Given a track ID from the MTG-Jamendo dataset, recommends similar tracks
using both the naive rule-based baseline and the trained classical ML model.
Can also query by mood tag directly.

Usage (from project root):
    # Recommend by track ID
    python main.py --track-id track_15171 --n 10

    # Recommend by mood tag
    python main.py --mood energetic --n 10

    # Show both baselines side by side
    python main.py --track-id track_15171 --compare
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.multiclass import OneVsRestClassifier

from scripts.build_features import (
    BUCKET_AFFINITY,
    MOOD_TAGS,
    audio_to_mood_bucket,
    extract_classical_features,
    load_mel,
)

MODELS_DIR    = "./models"
PROCESSED_DIR = "./data/processed"


# ── Artifact loading ───────────────────────────────────────────────────────────

def load_artifacts(
    models_dir: str = MODELS_DIR,
    processed_dir: str = PROCESSED_DIR,
) -> dict:
    """
    Load all trained model artifacts and the full track DataFrame from disk.

    Expected files:
        models/best_audio_clf.pkl
        models/audio_scaler.pkl
        models/knn_mood.pkl
        models/mlb.pkl
        models/all_audio_mood_probs.npy
        data/processed/df_all.parquet

    Args:
        models_dir:    Directory containing saved model .pkl / .npy files.
        processed_dir: Directory containing processed DataFrames.

    Returns:
        Dict with keys: clf, scaler, knn, mlb, all_probs, df_all.
    """
    def _load_pkl(fname: str):
        with open(Path(models_dir) / fname, "rb") as fh:
            return pickle.load(fh)

    print("Loading artifacts …")
    artifacts = {
        "clf":       _load_pkl("best_audio_clf.pkl"),
        "scaler":    _load_pkl("audio_scaler.pkl"),
        "knn":       _load_pkl("knn_mood.pkl"),
        "mlb":       _load_pkl("mlb.pkl"),
        "all_probs": np.load(str(Path(models_dir) / "all_audio_mood_probs.npy")),
        "df_all":    pd.read_parquet(f"{processed_dir}/df_all.parquet"),
    }

    # Load thresholds produced by build_features.py
    thresholds_path = Path(processed_dir) / "thresholds.pkl"
    if thresholds_path.exists():
        with open(thresholds_path, "rb") as fh:
            artifacts["thresholds"] = pickle.load(fh)
    else:
        artifacts["thresholds"] = None

    print(f"✅  Loaded — {len(artifacts['df_all']):,} tracks in index")
    return artifacts


# ── Mood vector extraction from a track ───────────────────────────────────────

def get_audio_mood_vector(
    track_id: str,
    df_all: pd.DataFrame,
    all_probs: np.ndarray,
) -> np.ndarray | None:
    """
    Look up the pre-computed 15-d mood probability vector for a track.

    Args:
        track_id:  MTG-Jamendo track ID string (e.g. 'track_15171').
        df_all:    Full DataFrame with TRACK_ID column.
        all_probs: Pre-computed (N_all, 15) mood probability matrix.

    Returns:
        1-D np.ndarray of shape (15,), or None if track_id not found.
    """
    idx_series = df_all.index[df_all["TRACK_ID"] == track_id]
    if len(idx_series) == 0:
        return None
    return all_probs[idx_series[0]]


def compute_mood_vector_from_file(
    mel_path: str,
    clf: OneVsRestClassifier,
    scaler: StandardScaler,
) -> np.ndarray | None:
    """
    Extract classical features from a mel-spectrogram file and predict mood vector.

    Useful for querying with a track that is not in df_all (e.g. a new file).

    Args:
        mel_path: Path to a .npy mel-spectrogram file.
        clf:      Trained OneVsRest classifier.
        scaler:   Fitted StandardScaler.

    Returns:
        1-D np.ndarray of shape (15,), or None if the file cannot be loaded.
    """
    mel = load_mel(mel_path)
    if mel is None:
        print(f"⚠️  Could not load mel-spectrogram: {mel_path}")
        return None
    features   = extract_classical_features(mel).reshape(1, -1)
    features_s = scaler.transform(features)
    return clf.predict_proba(features_s)[0]


# ── Naive recommender ──────────────────────────────────────────────────────────

def naive_recommend(
    query_bucket: str,
    df_all: pd.DataFrame,
    n: int = 10,
) -> list[dict]:
    """
    Retrieve the top-n tracks by naive audio bucket affinity.

    Scoring:
        2  — track's audio_bucket exactly matches query_bucket
        1  — track's audio_bucket is in the affinity list for query_bucket
        0  — no match

    Args:
        query_bucket: Target mood bucket string (one of the 15 MOOD_TAGS).
        df_all:       Full DataFrame with 'audio_bucket' column.
        n:            Number of tracks to return.

    Returns:
        List of result dicts, each with keys:
        TRACK_ID, mood_tags, audio_bucket, score, jamendo_url.
    """
    affinity = BUCKET_AFFINITY.get(query_bucket, [query_bucket])
    scores   = np.zeros(len(df_all), dtype=np.int8)

    for i, bucket in enumerate(df_all["audio_bucket"]):
        if bucket == query_bucket:
            scores[i] = 2
        elif bucket in affinity:
            scores[i] = 1

    top_idx = np.argsort(scores)[::-1][:n]
    results = []
    for idx in top_idx:
        row = df_all.iloc[idx]
        results.append({
            "TRACK_ID":    row["TRACK_ID"],
            "mood_tags":   row["mood_tags"],
            "audio_bucket": row["audio_bucket"],
            "score":       int(scores[idx]),
            "jamendo_url": _jamendo_url(row["TRACK_ID"]),
        })
    return results


# ── Classical recommender ──────────────────────────────────────────────────────

def classical_recommend(
    query_mood_vector: np.ndarray,
    df_all: pd.DataFrame,
    knn: NearestNeighbors,
    n: int = 10,
) -> list[dict]:
    """
    Retrieve the top-n tracks by cosine similarity in the 15-d mood space.

    Args:
        query_mood_vector: 1-D np.ndarray of shape (15,).
        df_all:            Full DataFrame.
        knn:               Fitted NearestNeighbors (cosine distance).
        n:                 Number of tracks to return.

    Returns:
        List of result dicts, each with keys:
        TRACK_ID, mood_tags, audio_bucket, cosine_similarity, jamendo_url.
    """
    distances, indices = knn.kneighbors(
        query_mood_vector.reshape(1, -1), n_neighbors=n)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        row = df_all.iloc[idx]
        results.append({
            "TRACK_ID":         row["TRACK_ID"],
            "mood_tags":        row["mood_tags"],
            "audio_bucket":     row.get("audio_bucket", "—"),
            "cosine_similarity": round(1.0 - float(dist), 4),
            "jamendo_url":      _jamendo_url(row["TRACK_ID"]),
        })
    return results


# ── Display helpers ────────────────────────────────────────────────────────────

def _jamendo_url(track_id: str) -> str:
    """Convert a track ID like 'track_15171' to a Jamendo URL."""
    numeric = track_id.replace("track_", "")
    return f"https://www.jamendo.com/track/{numeric}"


def print_naive_results(
    results: list[dict],
    query_bucket: str,
) -> None:
    """Pretty-print naive recommendation results."""
    affinity = BUCKET_AFFINITY.get(query_bucket, [])
    print(f"\n{'─'*65}")
    print(f"  🎵  Naive Baseline  |  Query bucket: {query_bucket.upper()}")
    print(f"  Affinity buckets : {', '.join(affinity)}")
    print(f"{'─'*65}")
    for i, t in enumerate(results, 1):
        tags_str = ", ".join(t["mood_tags"][:3])
        print(f"  {i:2d}. {t['TRACK_ID']:<20}  "
              f"bucket={t['audio_bucket']:<14}  "
              f"score={t['score']}  "
              f"tags=[{tags_str}]")
    print(f"{'─'*65}\n")


def print_classical_results(
    results: list[dict],
    query_mood_vector: np.ndarray,
    mlb: MultiLabelBinarizer,
) -> None:
    """Pretty-print classical recommendation results with top mood scores."""
    top_moods = sorted(zip(MOOD_TAGS, query_mood_vector),
                       key=lambda x: -x[1])[:5]
    print(f"\n{'─'*65}")
    print(f"  🎵  Classical Model  |  Top query moods:")
    for tag, score in top_moods:
        bar = "█" * int(score * 20)
        print(f"      {tag:<14} {score:.3f}  {bar}")
    print(f"{'─'*65}")
    for i, t in enumerate(results, 1):
        tags_str = ", ".join(t["mood_tags"][:3])
        print(f"  {i:2d}. {t['TRACK_ID']:<20}  "
              f"sim={t['cosine_similarity']:.4f}  "
              f"tags=[{tags_str}]")
    print(f"{'─'*65}\n")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visual Vibe — audio mood recommender CLI.")

    query_group = parser.add_mutually_exclusive_group(required=True)
    query_group.add_argument(
        "--track-id",
        help="MTG-Jamendo track ID to use as the query "
             "(e.g. 'track_15171').")
    query_group.add_argument(
        "--mood",
        choices=MOOD_TAGS,
        help="Mood tag to query directly (skips audio feature extraction).")
    query_group.add_argument(
        "--mel-path",
        help="Path to a .npy mel-spectrogram file to use as the query.")

    parser.add_argument(
        "--n", type=int, default=10,
        help="Number of recommendations to return (default: 10).")
    parser.add_argument(
        "--compare", action="store_true",
        help="Show both naive and classical recommendations side by side.")
    parser.add_argument(
        "--models-dir",    default=MODELS_DIR)
    parser.add_argument(
        "--processed-dir", default=PROCESSED_DIR)

    args = parser.parse_args()

    # Load artifacts
    arts = load_artifacts(args.models_dir, args.processed_dir)
    df_all     = arts["df_all"]
    clf        = arts["clf"]
    scaler     = arts["scaler"]
    knn        = arts["knn"]
    mlb        = arts["mlb"]
    all_probs  = arts["all_probs"]
    thresholds = arts["thresholds"]

    # ── Resolve query → mood vector + bucket ─────────────────────────────────
    query_bucket: str | None      = None
    mood_vector:  np.ndarray | None = None

    if args.mood:
        # Direct mood query — construct a one-hot-like vector
        query_bucket = args.mood
        mood_vector  = np.zeros(len(MOOD_TAGS), dtype=np.float32)
        mood_vector[MOOD_TAGS.index(args.mood)] = 1.0
        print(f"Query mode: direct mood tag → {args.mood}")

    elif args.track_id:
        mood_vector = get_audio_mood_vector(args.track_id, df_all, all_probs)
        if mood_vector is None:
            print(f"❌  Track ID '{args.track_id}' not found in the dataset.")
            raise SystemExit(1)

        # Derive naive bucket from the stored audio features (if available)
        row = df_all[df_all["TRACK_ID"] == args.track_id].iloc[0]
        if "audio_bucket" in row:
            query_bucket = row["audio_bucket"]
        elif thresholds is not None:
            query_bucket = audio_to_mood_bucket(
                row["energy"], row["brightness"], row["variance"], thresholds)

        print(f"Query track : {args.track_id}")
        print(f"Mood tags   : {', '.join(row['mood_tags'])}")
        if query_bucket:
            print(f"Audio bucket: {query_bucket}")

    elif args.mel_path:
        mood_vector = compute_mood_vector_from_file(args.mel_path, clf, scaler)
        if mood_vector is None:
            raise SystemExit(1)
        print(f"Query mel-spec: {args.mel_path}")

    # ── Naive recommendations ─────────────────────────────────────────────────
    if query_bucket and (args.compare or not args.mel_path):
        naive_results = naive_recommend(query_bucket, df_all, n=args.n)
        print_naive_results(naive_results, query_bucket)

    # ── Classical recommendations ─────────────────────────────────────────────
    classical_results = classical_recommend(mood_vector, df_all, knn, n=args.n)
    print_classical_results(classical_results, mood_vector, mlb)
