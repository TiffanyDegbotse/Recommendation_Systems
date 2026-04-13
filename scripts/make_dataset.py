# AI-assisted code generation (Claude, Anthropic) – https://claude.ai
"""
scripts/make_dataset.py
=======================
Downloads the MTG-Jamendo mood/theme mel-spectrogram subset, parses the
metadata TSV, aligns tracks with the official split-0 train/val/test
partitions, and saves processed DataFrames + label arrays to data/processed/.

Usage (from project root):
    python scripts/make_dataset.py              # full ~30 GB download
    python scripts/make_dataset.py --quick      # 2 shards only (~2 GB)
    python scripts/make_dataset.py --skip-download  # data already on disk
"""

import argparse
import os
import pickle
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

# ── Project-wide constants (must match build_features.py and model.py) ────────
MOOD_TAGS: list[str] = [
    "happy", "sad", "energetic", "calm", "dark", "epic",
    "romantic", "aggressive", "relaxing", "melancholic",
    "uplifting", "dramatic", "peaceful", "tense", "fun",
]

DEFAULT_CFG: dict = {
    "repo_dir":      "./mtg-jamendo-dataset",
    "melspec_dir":   "./data/melspecs",
    "metadata_path": "./data/autotagging_moodtheme.tsv",
    "splits_dir":    "./mtg-jamendo-dataset/data/splits/split-0",
    "processed_dir": "./data/processed",
}


# ── Step 1: Repository & metadata ─────────────────────────────────────────────

def clone_repo(repo_dir: str) -> None:
    """Clone the MTG-Jamendo GitHub repo (metadata + download scripts) if absent."""
    if Path(repo_dir).exists():
        print(f"✅  Repo already present at {repo_dir}")
        return
    print("Cloning MTG-Jamendo repository …")
    subprocess.run(
        ["git", "clone", "--depth", "1",
         "https://github.com/MTG/mtg-jamendo-dataset.git", repo_dir],
        check=True,
    )
    print(f"✅  Cloned → {repo_dir}")


def copy_metadata(repo_dir: str, metadata_path: str) -> None:
    """Copy the mood/theme metadata TSV from the repo into ./data/."""
    os.makedirs(Path(metadata_path).parent, exist_ok=True)
    src = Path(repo_dir) / "data" / "autotagging_moodtheme.tsv"
    shutil.copy(str(src), metadata_path)
    print(f"✅  Metadata TSV copied → {metadata_path}")


# ── Step 2: Download mel-spectrograms ─────────────────────────────────────────

def download_melspecs(
    repo_dir: str,
    melspec_dir: str,
    quick_mode: bool = False,
) -> None:
    """
    Download pre-computed mel-spectrograms via the official MTG download script.

    Args:
        repo_dir:    Path to the cloned MTG-Jamendo repo.
        melspec_dir: Destination directory for .npy mel-spectrogram files.
        quick_mode:  Download only the first 2 tar shards (~2 GB, ~1,200 tracks)
                     instead of the full ~30 GB dataset.
    """
    download_script = str(Path(repo_dir) / "scripts" / "download" / "download.py")
    os.makedirs(melspec_dir, exist_ok=True)

    cmd = [
        sys.executable, download_script,
        "--dataset", "autotagging_moodtheme",
        "--type",    "melspecs",
        "--from",    "mtg-fast",
        "--unpack",
        "--remove",
        melspec_dir,
    ]

    label = "QUICK (~2 GB, ~1 200 tracks)" if quick_mode else "FULL (~30 GB)"
    print(f"Downloading mel-spectrograms [{label}] — this may take 30–60 minutes …")
    result = subprocess.run(cmd, capture_output=False)
    print(f"✅  Download complete (exit code {result.returncode})")


def verify_download(melspec_dir: str) -> int:
    """
    Print a summary of downloaded .npy files and return the total count.

    Returns:
        Number of .npy files found under melspec_dir.
    """
    root      = Path(melspec_dir)
    npy_files = list(root.rglob("*.npy"))
    subdirs   = sorted({p.parent.name for p in npy_files})

    print(f"Mel-spectrogram files : {len(npy_files):,}")
    print(f"Sub-directories       : {len(subdirs)}  (first 5: {subdirs[:5]} …)")

    if npy_files:
        sample = np.load(str(npy_files[0]))
        print(f"Sample → shape {sample.shape}  dtype {sample.dtype}  "
              f"range [{sample.min():.3f}, {sample.max():.3f}]")
    else:
        print("⚠️  No .npy files found — check the download step.")

    return len(npy_files)


# ── Step 3: Parse metadata ─────────────────────────────────────────────────────

def load_metadata(
    metadata_path: str,
    melspec_dir: str,
    mood_tags: list[str],
) -> pd.DataFrame:
    """
    Parse the MTG-Jamendo TSV and return a filtered DataFrame.

    Keeps only tracks that:
      - Carry at least one of the target mood tags.
      - Have a corresponding .npy mel-spectrogram file on disk.

    Args:
        metadata_path: Path to autotagging_moodtheme.tsv.
        melspec_dir:   Root directory containing .npy files.
        mood_tags:     Target mood tag names (without the 'mood/theme---' prefix).

    Returns:
        DataFrame with columns: TRACK_ID, ARTIST_ID, ALBUM_ID, PATH, DURATION,
        mood_tags (list[str]), mel_path (str), mel_exists (bool).
    """
    mood_set = set(mood_tags)
    rows: list[dict] = []

    with open(metadata_path) as fh:
        for i, line in enumerate(fh):
            parts = line.strip().split("\t")
            if i == 0 or len(parts) < 5:
                continue  # skip header / malformed rows
            rows.append({
                "TRACK_ID":  parts[0],
                "ARTIST_ID": parts[1],
                "ALBUM_ID":  parts[2],
                "PATH":      parts[3],
                "DURATION":  parts[4],
                "_all_tags": parts[5:],
            })

    df = pd.DataFrame(rows)

    def _parse_mood_tags(tag_list: list[str]) -> list[str]:
        """Strip prefix and keep only tags in our target set."""
        return [
            t.replace("mood/theme---", "").strip()
            for t in tag_list
            if t.strip().startswith("mood/theme---")
            and t.strip().replace("mood/theme---", "") in mood_set
        ]

    def _mel_path(path_col: str) -> str:
        """Convert TSV PATH (e.g. '71/15171.mp3') → local .npy path."""
        parts      = path_col.split("/")
        subdir     = parts[0]
        track_name = parts[1].replace(".mp3", "")
        return str(Path(melspec_dir) / subdir / f"{track_name}.npy")

    df["mood_tags"]  = df["_all_tags"].apply(_parse_mood_tags)
    df               = df[df["mood_tags"].map(len) > 0].reset_index(drop=True)
    df["mel_path"]   = df["PATH"].apply(_mel_path)
    df["mel_exists"] = df["mel_path"].apply(lambda p: Path(p).exists())
    df               = df.drop(columns=["_all_tags"])

    print(f"Loaded {len(df):,} mood-tagged tracks  |  "
          f"{df['mel_exists'].sum():,} with mel-specs on disk")
    return df


# ── Step 4: Build official splits ─────────────────────────────────────────────

def load_split_ids(split_file: str) -> set[str]:
    """
    Parse a split TSV file and return the set of track IDs it contains.

    Args:
        split_file: Path to autotagging_moodtheme-{train,validation,test}.tsv.
    """
    ids: set[str] = set()
    with open(split_file) as fh:
        for i, line in enumerate(fh):
            if i == 0:
                continue  # skip header
            parts = line.strip().split("\t")
            if parts:
                ids.add(parts[0])
    return ids


def build_splits(
    df_all: pd.DataFrame,
    splits_dir: str,
    mood_tags: list[str],
) -> tuple:
    """
    Align df_all with the official MTG-Jamendo split-0 partitions and
    binarize mood labels using MultiLabelBinarizer.

    Args:
        df_all:     Full filtered DataFrame from load_metadata().
        splits_dir: Directory containing the three split TSV files.
        mood_tags:  Ordered list of target mood tags (defines column order).

    Returns:
        (df_train, df_val, df_test,
         Y_all, Y_train, Y_val, Y_test,
         mlb, train_ids, val_ids, test_ids)
    """
    train_ids = load_split_ids(
        str(Path(splits_dir) / "autotagging_moodtheme-train.tsv"))
    val_ids   = load_split_ids(
        str(Path(splits_dir) / "autotagging_moodtheme-validation.tsv"))
    test_ids  = load_split_ids(
        str(Path(splits_dir) / "autotagging_moodtheme-test.tsv"))

    df_train = df_all[df_all["TRACK_ID"].isin(train_ids)].reset_index(drop=True)
    df_val   = df_all[df_all["TRACK_ID"].isin(val_ids)].reset_index(drop=True)
    df_test  = df_all[df_all["TRACK_ID"].isin(test_ids)].reset_index(drop=True)

    print(f"Train: {len(df_train):,}  |  Val: {len(df_val):,}  |  "
          f"Test: {len(df_test):,}")

    # Fit binarizer on all 15 tags so column order is deterministic
    mlb = MultiLabelBinarizer(classes=mood_tags)
    mlb.fit([mood_tags])

    Y_all   = mlb.transform(df_all["mood_tags"])
    Y_train = mlb.transform(df_train["mood_tags"])
    Y_val   = mlb.transform(df_val["mood_tags"])
    Y_test  = mlb.transform(df_test["mood_tags"])

    n_tags  = len(mood_tags)
    avg_tags = Y_all.mean(axis=1).mean() * n_tags
    print(f"Label matrix: {Y_all.shape}  (avg {avg_tags:.2f} tags/track)")

    return (df_train, df_val, df_test,
            Y_all, Y_train, Y_val, Y_test,
            mlb, train_ids, val_ids, test_ids)


# ── Step 5: Persist processed data ────────────────────────────────────────────

def save_processed(
    processed_dir: str,
    df_all: pd.DataFrame,
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    Y_all: np.ndarray,
    Y_train: np.ndarray,
    Y_val: np.ndarray,
    Y_test: np.ndarray,
    mlb: MultiLabelBinarizer,
) -> None:
    """
    Save all processed DataFrames, binary label arrays, and the MLB to disk.

    Files written (all under processed_dir/):
        df_all.parquet, df_train.parquet, df_val.parquet, df_test.parquet
        Y_all.npy, Y_train.npy, Y_val.npy, Y_test.npy
        mlb.pkl
    """
    os.makedirs(processed_dir, exist_ok=True)

    for name, df in [("df_all",   df_all),
                     ("df_train", df_train),
                     ("df_val",   df_val),
                     ("df_test",  df_test)]:
        df.to_parquet(f"{processed_dir}/{name}.parquet", index=False)

    for name, arr in [("Y_all",   Y_all),
                      ("Y_train", Y_train),
                      ("Y_val",   Y_val),
                      ("Y_test",  Y_test)]:
        np.save(f"{processed_dir}/{name}.npy", arr)

    with open(f"{processed_dir}/mlb.pkl", "wb") as fh:
        pickle.dump(mlb, fh)

    print(f"✅  Processed data saved → {processed_dir}/")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download MTG-Jamendo mel-spectrograms and build dataset splits.")
    parser.add_argument(
        "--quick", action="store_true",
        help="Download only 2 shards (~2 GB) for a quick smoke-test.")
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Skip the mel-spectrogram download (data already present on disk).")
    args = parser.parse_args()

    cfg = DEFAULT_CFG

    # 1. Clone repo and copy metadata
    clone_repo(cfg["repo_dir"])
    copy_metadata(cfg["repo_dir"], cfg["metadata_path"])

    # 2. Download mel-spectrograms (unless already on disk)
    if not args.skip_download:
        download_melspecs(cfg["repo_dir"], cfg["melspec_dir"],
                          quick_mode=args.quick)
    verify_download(cfg["melspec_dir"])

    # 3. Parse metadata — keep only tracks with mel-specs on disk
    df_all = load_metadata(cfg["metadata_path"], cfg["melspec_dir"], MOOD_TAGS)
    df_all = df_all[df_all["mel_exists"]].reset_index(drop=True)

    # 4. Build official splits + binarize labels
    (df_train, df_val, df_test,
     Y_all, Y_train, Y_val, Y_test,
     mlb, *_) = build_splits(df_all, cfg["splits_dir"], MOOD_TAGS)

    # 5. Persist everything
    save_processed(
        cfg["processed_dir"],
        df_all, df_train, df_val, df_test,
        Y_all, Y_train, Y_val, Y_test,
        mlb,
    )
