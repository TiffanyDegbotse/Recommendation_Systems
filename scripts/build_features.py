# AI-assisted code generation (Claude, Anthropic) – https://claude.ai
"""
scripts/build_features.py
=========================
Extracts two levels of audio features from pre-computed mel-spectrogram .npy
files and caches the results to data/processed/.

Naive features (3 scalars per track — no ML):
    energy     : mean log-power across all mel-bands and time frames
    brightness : ratio of high-frequency (top 64 bands) to low-frequency energy
    variance   : std of per-frame mean energy (captures rhythmic complexity)

Classical features (261-d vector per track):
    128 mel-band means  +  128 mel-band stds  +  energy mean/std/range
    +  spectral brightness  +  temporal flux

Usage (from project root):
    python scripts/build_features.py
    python scripts/build_features.py --force   # overwrite existing caches
"""

import argparse
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# ── Feature-space dimensionality constants ────────────────────────────────────
N_MELS: int          = 128    # mel bands per spectrogram
TIME_FRAMES: int     = 1300   # frames — pad/crop to this fixed length
CLASSICAL_DIM: int   = 261    # 128 means + 128 stds + 5 global scalars

MOOD_TAGS: list[str] = [
    "happy", "sad", "energetic", "calm", "dark", "epic",
    "romantic", "aggressive", "relaxing", "melancholic",
    "uplifting", "dramatic", "peaceful", "tense", "fun",
]

# Bucket affinity map used by the naive recommender (audio-side only)
BUCKET_AFFINITY: dict[str, list[str]] = {
    "energetic":   ["energetic", "aggressive", "epic",       "fun"],
    "aggressive":  ["aggressive","energetic",  "dramatic",   "tense"],
    "calm":        ["calm",      "peaceful",   "relaxing",   "happy"],
    "peaceful":    ["peaceful",  "calm",       "relaxing",   "romantic"],
    "dark":        ["dark",      "dramatic",   "tense",      "melancholic"],
    "dramatic":    ["dramatic",  "dark",       "tense",      "epic"],
    "happy":       ["happy",     "fun",        "energetic",  "uplifting"],
    "sad":         ["sad",       "melancholic","calm",       "romantic"],
    "epic":        ["epic",      "dramatic",   "energetic",  "uplifting"],
    "romantic":    ["romantic",  "peaceful",   "calm",       "sad"],
    "relaxing":    ["relaxing",  "calm",       "peaceful",   "happy"],
    "melancholic": ["melancholic","sad",        "dark",       "calm"],
    "fun":         ["fun",       "happy",      "energetic",  "uplifting"],
    "tense":       ["tense",     "dark",       "dramatic",   "aggressive"],
    "uplifting":   ["uplifting", "happy",      "epic",       "energetic"],
}


# ── Mel-spectrogram loading ────────────────────────────────────────────────────

def load_mel(path: str, time_frames: int = TIME_FRAMES) -> np.ndarray | None:
    """
    Load a mel-spectrogram .npy file and return a fixed-length float32 array.

    The spectrogram is zero-padded (if shorter) or cropped (if longer) to
    exactly `time_frames` columns.

    Args:
        path:        Absolute or relative path to the .npy file.
        time_frames: Target number of time frames (default 1300).

    Returns:
        np.ndarray of shape (N_MELS, time_frames) and dtype float32,
        or None if the file is missing / malformed.
    """
    try:
        mel = np.load(path).astype(np.float32)
        if mel.ndim != 2 or mel.size == 0:
            return None

        t = mel.shape[1]
        if t < time_frames:
            mel = np.pad(mel, ((0, 0), (0, time_frames - t)))
        else:
            mel = mel[:, :time_frames]
        return mel
    except Exception:
        return None


# ── Naive feature extraction (3 scalars) ──────────────────────────────────────

def extract_naive_features(mel: np.ndarray) -> dict[str, float]:
    """
    Compute three interpretable scalars directly from the mel-spectrogram.

    The input is assumed to already be in log-power scale (as provided by the
    MTG-Jamendo pre-computed .npy files), so no additional log transform is applied.

    Args:
        mel: np.ndarray of shape (N_MELS, T) in log-power scale.

    Returns:
        Dict with keys 'energy', 'brightness', 'variance'.
    """
    energy     = float(mel.mean())
    brightness = float(mel[64:, :].mean() / (mel[:64, :].mean() + 1e-8))
    variance   = float(mel.mean(axis=0).std())
    return {"energy": energy, "brightness": brightness, "variance": variance}


# ── Classical feature extraction (261-d) ──────────────────────────────────────

def extract_classical_features(mel: np.ndarray) -> np.ndarray:
    """
    Build a 261-dimensional hand-crafted feature vector from a mel-spectrogram.

    Feature breakdown:
        [  0:128]  Per-band mean log-power  (128-d)
        [128:256]  Per-band std  log-power  (128-d)
        [256]      Global energy mean
        [257]      Global energy std
        [258]      Global energy dynamic range  (max - min)
        [259]      Spectral brightness  (high/low band ratio)
        [260]      Mean temporal flux  (mean |diff| across frames)

    Args:
        mel: np.ndarray of shape (N_MELS, T) in log-power scale.

    Returns:
        np.ndarray of shape (261,) and dtype float32.
    """
    band_mean   = mel.mean(axis=1)                              # (128,)
    band_std    = mel.std(axis=1)                               # (128,)
    frame_e     = mel.mean(axis=0)                              # (T,)
    energy_mean = float(frame_e.mean())
    energy_std  = float(frame_e.std())
    energy_rng  = float(frame_e.max() - frame_e.min())
    brightness  = float(mel[64:, :].mean() / (mel[:64, :].mean() + 1e-8))
    flux        = float(np.abs(np.diff(mel, axis=1)).mean())

    return np.concatenate([
        band_mean, band_std,
        [energy_mean, energy_std, energy_rng, brightness, flux],
    ]).astype(np.float32)


# ── Rule-based mood assignment (naive baseline, audio side) ───────────────────

def compute_thresholds(
    df: pd.DataFrame,
    low_q: float = 0.33,
    high_q: float = 0.67,
) -> dict[str, float]:
    """
    Compute energy/brightness/variance percentile thresholds from a DataFrame.

    Args:
        df:     DataFrame containing 'energy', 'brightness', 'variance' columns.
        low_q:  Lower percentile (default 0.33).
        high_q: Upper percentile (default 0.67).

    Returns:
        Dict with keys: energy_lo, energy_hi, brightness_lo, brightness_hi,
        variance_hi.
    """
    return {
        "energy_lo":     float(df["energy"].quantile(low_q)),
        "energy_hi":     float(df["energy"].quantile(high_q)),
        "brightness_lo": float(df["brightness"].quantile(low_q)),
        "brightness_hi": float(df["brightness"].quantile(high_q)),
        "variance_hi":   float(df["variance"].quantile(high_q)),
    }


def audio_to_mood_bucket(
    energy: float,
    brightness: float,
    variance: float,
    thresholds: dict[str, float],
) -> str:
    """
    Map three audio scalars to one of the 15 mood tags using rule-based logic.

    Rules encode musical intuitions calibrated from training-split percentiles:
        high energy + bright + dynamic  →  aggressive
        high energy + bright            →  energetic
        high energy + dynamic           →  dramatic
        high energy                     →  epic
        low energy  + dark + stable     →  peaceful
        low energy  + dark              →  calm
        low energy  + dynamic           →  tense
        low energy                      →  relaxing
        dynamic + dark                  →  dark
        dynamic                         →  melancholic
        bright + not loud               →  fun
        dark                            →  sad
        (default)                       →  happy

    Args:
        energy, brightness, variance: Scalar audio features.
        thresholds: Dict from compute_thresholds().

    Returns:
        One of the 15 mood tag strings.
    """
    high_e = energy     > thresholds["energy_hi"]
    low_e  = energy     < thresholds["energy_lo"]
    high_b = brightness > thresholds["brightness_hi"]
    low_b  = brightness < thresholds["brightness_lo"]
    high_v = variance   > thresholds["variance_hi"]

    if high_e and high_b and high_v:  return "aggressive"
    if high_e and high_b:             return "energetic"
    if high_e and high_v:             return "dramatic"
    if high_e:                        return "epic"
    if low_e  and low_b and not high_v: return "peaceful"
    if low_e  and low_b:              return "calm"
    if low_e  and high_v:             return "tense"
    if low_e:                         return "relaxing"
    if high_v and low_b:              return "dark"
    if high_v:                        return "melancholic"
    if high_b and not high_e:         return "fun"
    if low_b:                         return "sad"
    return "happy"


# ── Batch extraction with caching ─────────────────────────────────────────────

def batch_extract_features(
    df_all: pd.DataFrame,
    processed_dir: str,
    force: bool = False,
) -> tuple[list[dict], np.ndarray]:
    """
    Extract naive and classical features for every track, with disk caching.

    Skips extraction and loads from cache if both cache files already exist
    (unless force=True).

    Args:
        df_all:        Full DataFrame with a 'mel_path' column.
        processed_dir: Directory for reading/writing cache files.
        force:         Re-extract and overwrite caches if True.

    Returns:
        naive_records   : list of dicts, one per track, with keys
                          'energy', 'brightness', 'variance'.
        X_classical_all : np.ndarray of shape (N, 261).
    """
    naive_cache     = Path(processed_dir) / "naive_features.pkl"
    classical_cache = Path(processed_dir) / "classical_features.npy"

    if not force and naive_cache.exists() and classical_cache.exists():
        print("Loading cached features …")
        with open(naive_cache, "rb") as fh:
            naive_records = pickle.load(fh)
        X_classical_all = np.load(str(classical_cache))
        print(f"✅  Loaded from cache — {len(naive_records):,} tracks, "
              f"classical matrix {X_classical_all.shape}")
        return naive_records, X_classical_all

    # ── Extract from scratch ──────────────────────────────────────────────────
    naive_records: list[dict]   = []
    classical_feats: list[np.ndarray] = []

    for _, row in tqdm(df_all.iterrows(), total=len(df_all),
                       desc="Extracting features"):
        mel = load_mel(row["mel_path"])
        if mel is None:
            # Use zero-feature fallback for missing/corrupt files
            naive_records.append({"energy": 0.0, "brightness": 1.0,
                                   "variance": 0.0})
            classical_feats.append(np.zeros(CLASSICAL_DIM, dtype=np.float32))
        else:
            naive_records.append(extract_naive_features(mel))
            classical_feats.append(extract_classical_features(mel))

    X_classical_all = np.stack(classical_feats)

    # ── Write caches ──────────────────────────────────────────────────────────
    os.makedirs(processed_dir, exist_ok=True)
    with open(naive_cache, "wb") as fh:
        pickle.dump(naive_records, fh)
    np.save(str(classical_cache), X_classical_all)

    print(f"✅  Extraction complete — classical matrix {X_classical_all.shape}")
    return naive_records, X_classical_all


def attach_naive_features(
    df_all: pd.DataFrame,
    naive_records: list[dict],
    train_ids: set[str],
) -> tuple[pd.DataFrame, dict[str, float]]:
    """
    Add energy/brightness/variance/audio_bucket columns to df_all in-place.

    Thresholds are computed solely from the training split to prevent data leakage.

    Args:
        df_all:       Full DataFrame.
        naive_records: List of feature dicts aligned to df_all rows.
        train_ids:    Set of training track IDs.

    Returns:
        Updated df_all and the thresholds dict.
    """
    df_all = df_all.copy()
    df_all["energy"]     = [r["energy"]     for r in naive_records]
    df_all["brightness"] = [r["brightness"] for r in naive_records]
    df_all["variance"]   = [r["variance"]   for r in naive_records]

    # Compute thresholds on training split only
    train_mask = df_all["TRACK_ID"].isin(train_ids)
    thresholds = compute_thresholds(df_all.loc[train_mask])

    df_all["audio_bucket"] = df_all.apply(
        lambda r: audio_to_mood_bucket(
            r["energy"], r["brightness"], r["variance"], thresholds),
        axis=1,
    )
    return df_all, thresholds


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract naive and classical audio features from mel-spectrograms.")
    parser.add_argument(
        "--force", action="store_true",
        help="Re-extract features even if caches already exist.")
    parser.add_argument(
        "--processed-dir", default="./data/processed",
        help="Directory containing processed DataFrames and for writing caches.")
    args = parser.parse_args()

    processed_dir = args.processed_dir

    # Load processed data produced by make_dataset.py
    df_all    = pd.read_parquet(f"{processed_dir}/df_all.parquet")
    train_ids = set(
        pd.read_parquet(f"{processed_dir}/df_train.parquet")["TRACK_ID"])

    print(f"Loaded {len(df_all):,} tracks from {processed_dir}/")

    # Extract features
    naive_records, X_classical_all = batch_extract_features(
        df_all, processed_dir, force=args.force)

    # Attach naive scalars + mood buckets (thresholds from training split)
    df_all, thresholds = attach_naive_features(df_all, naive_records, train_ids)

    # Save updated df_all with feature columns
    df_all.to_parquet(f"{processed_dir}/df_all.parquet", index=False)

    # Save thresholds for use by model.py and main.py
    with open(f"{processed_dir}/thresholds.pkl", "wb") as fh:
        pickle.dump(thresholds, fh)

    print(f"\nThresholds (from training split):")
    for k, v in thresholds.items():
        print(f"  {k:<16} {v:.4f}")

    print(f"\nAudio bucket distribution:")
    print(df_all["audio_bucket"].value_counts().to_string())

    print(f"\n✅  Features saved to {processed_dir}/")
