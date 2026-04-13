# AI-assisted code generation (Claude, Anthropic) – https://claude.ai
"""
scripts/model.py
================
Trains two audio mood classifiers (Logistic Regression and Random Forest,
both in a One-vs-Rest multi-label setup), selects the best one, builds a
cosine-similarity KNN retrieval index, evaluates on the test set, and saves
all model artifacts to models/.

Usage (from project root):
    python scripts/model.py
    python scripts/model.py --models-dir ./models --processed-dir ./data/processed
"""

import argparse
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

MOOD_TAGS: list[str] = [
    "happy", "sad", "energetic", "calm", "dark", "epic",
    "romantic", "aggressive", "relaxing", "melancholic",
    "uplifting", "dramatic", "peaceful", "tense", "fun",
]
N_MOODS: int = len(MOOD_TAGS)

SEED: int = 42


# ── Feature preprocessing ──────────────────────────────────────────────────────

def scale_features(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    X_all: np.ndarray,
) -> tuple[StandardScaler, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit a StandardScaler on the training set and transform all splits.

    Args:
        X_train: Training feature matrix (N_train, D).
        X_val:   Validation feature matrix (N_val, D).
        X_test:  Test feature matrix (N_test, D).
        X_all:   Full dataset feature matrix (N_all, D).

    Returns:
        (scaler, X_train_s, X_val_s, X_test_s, X_all_s)
    """
    scaler     = StandardScaler()
    X_train_s  = scaler.fit_transform(X_train)
    X_val_s    = scaler.transform(X_val)
    X_test_s   = scaler.transform(X_test)
    X_all_s    = scaler.transform(X_all)
    return scaler, X_train_s, X_val_s, X_test_s, X_all_s


# ── Classifier training ────────────────────────────────────────────────────────

def build_classifiers(seed: int = SEED) -> dict[str, OneVsRestClassifier]:
    """
    Return a dict of untrained One-vs-Rest multi-label classifiers.

    Args:
        seed: Random seed for reproducibility.

    Returns:
        Dict mapping a human-readable name → sklearn estimator.
    """
    return {
        "Logistic Regression": OneVsRestClassifier(
            LogisticRegression(C=1.0, max_iter=500, solver="lbfgs",
                               random_state=seed),
            n_jobs=-1,
        ),
        "Random Forest": OneVsRestClassifier(
            RandomForestClassifier(n_estimators=150, max_depth=12,
                                   min_samples_leaf=2, random_state=seed),
            n_jobs=-1,
        ),
    }


def train_and_validate(
    classifiers: dict[str, OneVsRestClassifier],
    X_train_s: np.ndarray,
    Y_train: np.ndarray,
    X_val_s: np.ndarray,
    Y_val: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Train each classifier and compute validation-set probability predictions.

    Prints a summary line per model with micro-F1 and mean ROC-AUC.

    Args:
        classifiers: Dict of untrained estimators (modified in-place after fit).
        X_train_s:   Scaled training features.
        Y_train:     Binary label matrix for training.
        X_val_s:     Scaled validation features.
        Y_val:       Binary label matrix for validation.

    Returns:
        Dict mapping model name → val probability matrix (N_val, N_MOODS).
    """
    val_probs: dict[str, np.ndarray] = {}

    for name, clf in classifiers.items():
        print(f"Training {name} …", end=" ", flush=True)
        clf.fit(X_train_s, Y_train)

        probs = clf.predict_proba(X_val_s)
        preds = (probs >= 0.5).astype(int)

        mean_auc = np.nanmean([
            roc_auc_score(Y_val[:, j], probs[:, j])
            if Y_val[:, j].sum() > 0 else np.nan
            for j in range(N_MOODS)
        ])
        micro_f1 = f1_score(Y_val, preds, average="micro", zero_division=0)

        print(f"micro-F1={micro_f1:.4f}  mean-AUC={mean_auc:.4f}")
        val_probs[name] = probs

    return val_probs


def select_best_classifier(
    classifiers: dict[str, OneVsRestClassifier],
    val_probs: dict[str, np.ndarray],
    Y_val: np.ndarray,
) -> tuple[str, OneVsRestClassifier]:
    """
    Pick the classifier with the highest validation micro-F1.

    Args:
        classifiers: Trained classifier dict.
        val_probs:   Val probability predictions from train_and_validate().
        Y_val:       Binary label matrix for validation.

    Returns:
        (best_name, best_clf)
    """
    best_name = max(
        classifiers,
        key=lambda n: f1_score(
            Y_val, (val_probs[n] >= 0.5).astype(int),
            average="micro", zero_division=0,
        ),
    )
    print(f"✅  Best classifier: {best_name}")
    return best_name, classifiers[best_name]


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate_classifier(
    clf: OneVsRestClassifier,
    X_test_s: np.ndarray,
    Y_test: np.ndarray,
    clf_name: str = "",
) -> dict[str, float]:
    """
    Compute test-set metrics for a trained audio classifier.

    Metrics: ROC-AUC (macro), PR-AUC (macro), micro-F1, macro-F1, BCE loss.

    Args:
        clf:       Trained OneVsRest estimator.
        X_test_s:  Scaled test features.
        Y_test:    Binary label matrix for the test set.
        clf_name:  Optional name for display purposes.

    Returns:
        Dict with keys: roc_auc, pr_auc, micro_f1, macro_f1, bce_loss.
    """
    probs = clf.predict_proba(X_test_s)
    preds = (probs >= 0.5).astype(int)

    roc_auc  = float(np.nanmean([
        roc_auc_score(Y_test[:, j], probs[:, j])
        if Y_test[:, j].sum() > 0 else np.nan
        for j in range(N_MOODS)
    ]))
    pr_auc   = float(np.nanmean([
        average_precision_score(Y_test[:, j], probs[:, j])
        if Y_test[:, j].sum() > 0 else np.nan
        for j in range(N_MOODS)
    ]))
    micro_f1 = float(f1_score(Y_test, preds, average="micro",  zero_division=0))
    macro_f1 = float(f1_score(Y_test, preds, average="macro",  zero_division=0))
    bce      = float(log_loss(Y_test, np.clip(probs, 1e-15, 1 - 1e-15)))

    label = f" [{clf_name}]" if clf_name else ""
    print(f"Test metrics{label}:")
    print(f"  ROC-AUC  = {roc_auc:.4f}")
    print(f"  PR-AUC   = {pr_auc:.4f}")
    print(f"  micro-F1 = {micro_f1:.4f}")
    print(f"  macro-F1 = {macro_f1:.4f}")
    print(f"  BCE loss = {bce:.4f}")

    return {
        "roc_auc":  roc_auc,
        "pr_auc":   pr_auc,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "bce_loss": bce,
    }


def evaluate_naive_baseline(
    df_test: pd.DataFrame,
    bucket_affinity: dict[str, list[str]],
) -> dict[str, float]:
    """
    Evaluate the naive rule-based baseline on the test split.

    A track is counted as a hit if any of its ground-truth mood tags appears
    in the affinity list of its predicted audio_bucket.

    Args:
        df_test:          Test-split DataFrame with 'audio_bucket' and
                          'mood_tags' columns (populated by build_features.py).
        bucket_affinity:  BUCKET_AFFINITY dict from build_features.py.

    Returns:
        Dict with key 'affinity_match_rate' (float in [0, 1]).
    """
    hits = sum(
        1 for _, row in df_test.iterrows()
        if any(t in bucket_affinity.get(row["audio_bucket"], [])
               for t in row["mood_tags"])
    )
    rate = hits / len(df_test) if len(df_test) > 0 else 0.0
    print(f"Naive affinity match: {hits:,} / {len(df_test):,}  ({rate:.1%})")
    return {"affinity_match_rate": rate}


# ── Retrieval index ────────────────────────────────────────────────────────────

def build_retrieval_index(
    mood_prob_matrix: np.ndarray,
    n_neighbors: int = 30,
) -> NearestNeighbors:
    """
    Fit a brute-force cosine-similarity KNN index over the mood probability space.

    At query time, a 15-d mood vector is used to retrieve the n closest songs.

    Args:
        mood_prob_matrix: (N_all, 15) array of per-track mood probabilities.
        n_neighbors:      Number of neighbours to store (can retrieve fewer
                          at query time).

    Returns:
        Fitted NearestNeighbors estimator.
    """
    knn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine",
                           algorithm="brute")
    knn.fit(mood_prob_matrix)
    print(f"✅  KNN index built over {mood_prob_matrix.shape[0]:,} tracks "
          f"({mood_prob_matrix.shape[1]}-d mood space)")
    return knn


# ── Artifact persistence ───────────────────────────────────────────────────────

def save_models(
    models_dir: str,
    best_audio_clf: OneVsRestClassifier,
    scaler: StandardScaler,
    knn_mood: NearestNeighbors,
    mlb,
    all_audio_probs: np.ndarray,
    metrics: dict,
) -> None:
    """
    Save all trained model objects and the pre-computed mood probability matrix.

    Files written (all under models_dir/):
        best_audio_clf.pkl    – best OneVsRest audio classifier
        audio_scaler.pkl      – StandardScaler fitted on training features
        knn_mood.pkl          – KNN retrieval index
        mlb.pkl               – MultiLabelBinarizer
        all_audio_mood_probs.npy – pre-computed (N_all, 15) mood matrix
        metrics.pkl           – evaluation metrics dict

    Args:
        models_dir:       Target directory (created if absent).
        best_audio_clf:   Trained best classifier.
        scaler:           Fitted StandardScaler.
        knn_mood:         Fitted NearestNeighbors.
        mlb:              Fitted MultiLabelBinarizer.
        all_audio_probs:  Pre-computed mood probabilities for all tracks.
        metrics:          Dict of evaluation metrics from evaluate_classifier().
    """
    os.makedirs(models_dir, exist_ok=True)

    artifacts = {
        "best_audio_clf.pkl": best_audio_clf,
        "audio_scaler.pkl":   scaler,
        "knn_mood.pkl":       knn_mood,
        "mlb.pkl":            mlb,
    }
    for fname, obj in artifacts.items():
        with open(f"{models_dir}/{fname}", "wb") as fh:
            pickle.dump(obj, fh)

    np.save(f"{models_dir}/all_audio_mood_probs.npy", all_audio_probs)

    with open(f"{models_dir}/metrics.pkl", "wb") as fh:
        pickle.dump(metrics, fh)

    print(f"✅  Models saved → {models_dir}/")
    for fname in Path(models_dir).iterdir():
        print(f"  {fname.name:<38}  {fname.stat().st_size // 1024} KB")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train audio mood classifiers and build the retrieval index.")
    parser.add_argument("--processed-dir", default="./data/processed",
                        help="Directory containing features and label arrays.")
    parser.add_argument("--models-dir",    default="./models",
                        help="Output directory for trained model artifacts.")
    args = parser.parse_args()

    processed_dir = args.processed_dir
    models_dir    = args.models_dir

    # ── Load processed data ───────────────────────────────────────────────────
    print("Loading processed data …")
    df_all    = pd.read_parquet(f"{processed_dir}/df_all.parquet")
    df_test   = pd.read_parquet(f"{processed_dir}/df_test.parquet")

    Y_all   = np.load(f"{processed_dir}/Y_all.npy")
    Y_train = np.load(f"{processed_dir}/Y_train.npy")
    Y_val   = np.load(f"{processed_dir}/Y_val.npy")
    Y_test  = np.load(f"{processed_dir}/Y_test.npy")

    with open(f"{processed_dir}/mlb.pkl", "rb") as fh:
        mlb = pickle.load(fh)

    X_all = np.load(f"{processed_dir}/classical_features.npy")

    train_ids = set(pd.read_parquet(f"{processed_dir}/df_train.parquet")["TRACK_ID"])
    val_ids   = set(pd.read_parquet(f"{processed_dir}/df_val.parquet")["TRACK_ID"])
    test_ids  = set(pd.read_parquet(f"{processed_dir}/df_test.parquet")["TRACK_ID"])

    print(f"Dataset: {len(df_all):,} tracks  |  "
          f"features shape: {X_all.shape}")

    # ── Build split masks ─────────────────────────────────────────────────────
    train_mask = df_all["TRACK_ID"].isin(train_ids).values
    val_mask   = df_all["TRACK_ID"].isin(val_ids).values
    test_mask  = df_all["TRACK_ID"].isin(test_ids).values

    X_train = X_all[train_mask]
    X_val   = X_all[val_mask]
    X_test  = X_all[test_mask]

    # ── Scale features (fit on training only to avoid leakage) ───────────────
    scaler, X_train_s, X_val_s, X_test_s, X_all_s = scale_features(
        X_train, X_val, X_test, X_all)

    # ── Train & validate classifiers ─────────────────────────────────────────
    classifiers = build_classifiers(seed=SEED)
    val_probs   = train_and_validate(
        classifiers, X_train_s, Y_train, X_val_s, Y_val)

    best_name, best_audio_clf = select_best_classifier(
        classifiers, val_probs, Y_val)

    # ── Pre-compute mood probabilities for every track in the dataset ─────────
    all_audio_probs = best_audio_clf.predict_proba(X_all_s)
    print(f"Pre-computed mood probs: {all_audio_probs.shape}")

    # ── Evaluate on test set ──────────────────────────────────────────────────
    print("\n=== Classical Audio Classifier — Test Set ===")
    metrics = evaluate_classifier(best_audio_clf, X_test_s, Y_test,
                                  clf_name=best_name)

    # ── Naive baseline evaluation ─────────────────────────────────────────────
    from scripts.build_features import BUCKET_AFFINITY  # noqa: E402
    print("\n=== Naive Baseline — Test Set ===")
    test_df_with_buckets = df_all[test_mask].reset_index(drop=True)
    naive_metrics = evaluate_naive_baseline(test_df_with_buckets, BUCKET_AFFINITY)
    metrics["naive_affinity_match"] = naive_metrics["affinity_match_rate"]

    # ── Build KNN retrieval index ─────────────────────────────────────────────
    knn_mood = build_retrieval_index(all_audio_probs)

    # ── Save everything ───────────────────────────────────────────────────────
    save_models(
        models_dir,
        best_audio_clf=best_audio_clf,
        scaler=scaler,
        knn_mood=knn_mood,
        mlb=mlb,
        all_audio_probs=all_audio_probs,
        metrics=metrics,
    )
