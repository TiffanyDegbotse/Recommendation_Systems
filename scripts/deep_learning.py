# AI-assisted code generation (Claude, Anthropic) – https://claude.ai
"""
scripts/deep_learning.py
========================
Deep learning pipeline for the Visual Vibe image → song recommender.

Architecture:
    Image  → CLIP ViT-B/32 (frozen) → 512-d embedding
                                           ↓  CLIPToMoodProjection MLP
    Song   → Mel-spec → ResNet-18 CNN  → 512-d embedding → 15-d mood probs
                                           ↓  cosine similarity
                                       Top-K songs returned

Usage (from project root):
    python scripts/deep_learning.py
    python scripts/deep_learning.py --epochs 30 --batch-size 32
    python scripts/deep_learning.py --quick   # 5 epochs for smoke-test
"""

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import models
from tqdm import tqdm

import open_clip

# ── Constants (must match make_dataset.py) ────────────────────────────────────
MOOD_TAGS: list[str] = [
    "happy", "sad", "energetic", "calm", "dark", "epic",
    "romantic", "aggressive", "relaxing", "melancholic",
    "uplifting", "dramatic", "peaceful", "tense", "fun",
]
N_MOODS: int = len(MOOD_TAGS)
SEED: int    = 42


# ── Reproducibility helpers ───────────────────────────────────────────────────

def set_seeds(seed: int = SEED) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Return CUDA device if available, otherwise CPU."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


# ── Dataset ───────────────────────────────────────────────────────────────────

class JamendoMelspecDataset(Dataset):
    """
    PyTorch Dataset that loads pre-computed MTG-Jamendo mel-spectrograms.

    Each .npy file is stored as  melspec_dir/<subdir>/<numeric_id>.npy
    where the subdir and id are derived from the TSV PATH column
    (e.g. '71/15171.mp3' → subdir='71', id='15171').

    Returns single-channel tensors of shape (1, n_mels, time_frames)
    with values normalised to [-1, 1].
    """

    def __init__(
        self,
        df: pd.DataFrame,
        melspec_dir: str,
        n_mels: int = 128,
        time_frames: int = 1300,
        augment: bool = False,
    ):
        """
        Args:
            df:           Metadata DataFrame with TRACK_ID, PATH,
                          label_vector columns.
            melspec_dir:  Root directory of .npy mel-spectrogram files.
            n_mels:       Number of mel frequency bands.
            time_frames:  Fixed time dimension; shorter clips are zero-padded,
                          longer clips are cropped.
            augment:      If True, applies random crop + SpecAugment masking.
        """
        self.df          = df.reset_index(drop=True)
        self.melspec_dir = Path(melspec_dir)
        self.n_mels      = n_mels
        self.time_frames = time_frames
        self.augment     = augment

    def __len__(self) -> int:
        return len(self.df)

    def _load_melspec(self, path_col: str) -> np.ndarray:
        """
        Resolve and load the .npy file for a track.

        Args:
            path_col: TSV PATH value (e.g. '71/15171.mp3').

        Returns:
            float32 ndarray of shape (n_mels, T), or zeros on failure.
        """
        subdir     = path_col.split("/")[0]
        numeric_id = path_col.split("/")[1].replace(".mp3", "")
        npy_path   = self.melspec_dir / subdir / f"{numeric_id}.npy"

        if not npy_path.exists():
            return np.zeros((self.n_mels, self.time_frames), dtype=np.float32)
        return np.load(str(npy_path)).astype(np.float32)

    def _pad_or_crop(self, mel: np.ndarray) -> np.ndarray:
        """Pad (right) or crop mel-spectrogram to self.time_frames width."""
        t = mel.shape[1]
        if t < self.time_frames:
            mel = np.pad(mel, ((0, 0), (0, self.time_frames - t)))
        else:
            start = (random.randint(0, t - self.time_frames)
                     if self.augment else 0)
            mel = mel[:, start : start + self.time_frames]
        return mel

    def _specaugment(self, mel: np.ndarray) -> np.ndarray:
        """
        Apply SpecAugment: random frequency and time masking.
        Operates on a copy to avoid mutating the cached array.
        """
        mel = mel.copy()
        # Frequency masking
        f_mask  = random.randint(0, 20)
        f_start = random.randint(0, max(self.n_mels - f_mask, 0))
        mel[f_start : f_start + f_mask, :] = 0
        # Time masking
        t_mask  = random.randint(0, 100)
        t_start = random.randint(0, max(self.time_frames - t_mask, 0))
        mel[:, t_start : t_start + t_mask] = 0
        return mel

    def __getitem__(self, idx: int) -> dict:
        row       = self.df.iloc[idx]
        track_id  = str(row["TRACK_ID"])
        path      = str(row["PATH"])

        mel = self._load_melspec(path)
        mel = self._pad_or_crop(mel)
        if self.augment:
            mel = self._specaugment(mel)

        # Normalise to [-1, 1]
        mel_min, mel_max = mel.min(), mel.max()
        if mel_max > mel_min:
            mel = 2.0 * (mel - mel_min) / (mel_max - mel_min) - 1.0

        mel_tensor = torch.from_numpy(mel).unsqueeze(0)          # (1, n_mels, T)
        label      = torch.tensor(row["label_vector"], dtype=torch.float32)

        return {"melspec": mel_tensor, "label": label, "track_id": track_id}


# ── CLIP image encoder ────────────────────────────────────────────────────────

def load_clip_model(
    model_name: str = "ViT-B-32",
    pretrained: str = "openai",
    device: torch.device | None = None,
) -> tuple:
    """
    Load a pretrained CLIP model via open_clip.  All weights are frozen;
    CLIP is used purely as a fixed feature extractor.

    Args:
        model_name: CLIP architecture string (ViT-B-32 → 512-d embeddings).
        pretrained: Weight source identifier.
        device:     Target device (defaults to CPU if not provided).

    Returns:
        (clip_model, preprocess_fn)
    """
    if device is None:
        device = torch.device("cpu")

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained
    )
    model = model.to(device).eval()

    for param in model.parameters():
        param.requires_grad = False

    total = sum(p.numel() for p in model.parameters())
    print(f"CLIP loaded: {model_name}  ({total/1e6:.1f}M params, frozen)")
    return model, preprocess


@torch.no_grad()
def encode_images(
    images: list,
    clip_model: nn.Module,
    preprocess,
    device: torch.device,
) -> torch.Tensor:
    """
    Encode a list of PIL Images through CLIP.

    Args:
        images:      List of PIL Image objects.
        clip_model:  Loaded (frozen) CLIP model.
        preprocess:  CLIP preprocessing transform.
        device:      Target device.

    Returns:
        L2-normalised tensor of shape (N, 512).
    """
    tensors = torch.stack([preprocess(img) for img in images]).to(device)
    embs    = clip_model.encode_image(tensors)
    return F.normalize(embs, dim=-1)


# ── Audio CNN encoder ─────────────────────────────────────────────────────────

class MelspecCNNEncoder(nn.Module):
    """
    ResNet-18 based CNN that maps mel-spectrograms to a shared embedding space.

    The first convolutional layer is adapted to accept single-channel input.
    A projection head maps the ResNet features to embed_dim dimensions, and
    an auxiliary mood classifier head enables BCE supervision during training.

    Outputs are L2-normalised to unit norm for cosine similarity retrieval.
    """

    def __init__(self, embed_dim: int = 512, pretrained: bool = True):
        """
        Args:
            embed_dim:  Output embedding dimension (must match CLIP dim = 512).
            pretrained: Initialise ResNet backbone with ImageNet weights.
        """
        super().__init__()

        backbone = models.resnet18(pretrained=pretrained)

        # Replace first conv to handle 1-channel mel-spectrograms
        backbone.conv1 = nn.Conv2d(
            in_channels=1, out_channels=64,
            kernel_size=7, stride=2, padding=3, bias=False,
        )
        if pretrained:
            with torch.no_grad():
                # Warm-start: average ImageNet RGB weights across channels
                backbone.conv1.weight = nn.Parameter(
                    backbone.conv1.weight.mean(dim=1, keepdim=True))

        feature_dim  = backbone.fc.in_features   # 512 for ResNet-18
        backbone.fc  = nn.Identity()
        self.backbone = backbone

        # Projection head: feature_dim → embed_dim
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(feature_dim, embed_dim),
        )

        # Auxiliary multi-label classifier (BCE supervision)
        self.mood_classifier = nn.Linear(embed_dim, N_MOODS)

    def forward(self, x: torch.Tensor) -> dict:
        """
        Args:
            x: Mel-spectrogram tensor (B, 1, n_mels, time_frames).

        Returns:
            dict with:
                'embedding': L2-normalised embedding (B, embed_dim)
                'logits':    Raw mood logits (B, N_MOODS)
        """
        features  = self.backbone(x)              # (B, 512)
        embedding = self.projection(features)     # (B, embed_dim)
        embedding = F.normalize(embedding, dim=-1)
        logits    = self.mood_classifier(embedding)
        return {"embedding": embedding, "logits": logits}


# ── CLIP → mood projection ────────────────────────────────────────────────────

class CLIPToMoodProjection(nn.Module):
    """
    Small MLP that maps CLIP image embeddings (512-d) to mood probabilities.

    Trained via pseudo-label distillation: CLIP embeddings of mel-spectrogram
    thumbnails are paired with mood predictions from the trained audio encoder,
    so at inference any natural photo can be mapped into the mood space.
    """

    def __init__(self, clip_dim: int = 512, n_moods: int = N_MOODS):
        """
        Args:
            clip_dim: Input dimension (CLIP embedding size = 512).
            n_moods:  Output dimension (number of mood tags = 15).
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(clip_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_moods),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: CLIP embedding tensor (B, clip_dim).

        Returns:
            Raw mood logits (B, n_moods) — apply sigmoid for probabilities.
        """
        return self.net(x)


# ── Combined loss ─────────────────────────────────────────────────────────────

class CombinedLoss(nn.Module):
    """
    Weighted sum of multi-label BCE and NT-Xent contrastive loss.

    BCE loss provides direct mood tag supervision.
    NT-Xent loss pulls same-mood songs closer in embedding space, encouraging
    a structure that CLIP embeddings can be matched to via cosine similarity.

    Set alpha=0.0 to use BCE only (the default per the original notebook).
    """

    def __init__(self, temperature: float = 0.07, alpha: float = 0.0):
        """
        Args:
            temperature: Softmax temperature for contrastive loss.
            alpha:       Weight of contrastive loss (1-alpha for BCE).
        """
        super().__init__()
        self.temperature = temperature
        self.alpha       = alpha
        self.bce         = nn.BCEWithLogitsLoss()

    def _contrastive_loss(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        NT-Xent loss using multi-label tag overlap as positive-pair criterion.
        Two tracks are positives if they share at least one mood tag.

        Args:
            embeddings: L2-normalised embeddings (B, D).
            labels:     Multi-hot label vectors (B, N_MOODS).

        Returns:
            Scalar contrastive loss.
        """
        B      = embeddings.shape[0]
        sim    = torch.mm(embeddings, embeddings.T) / self.temperature  # (B, B)

        # Positive mask: shared tag overlap, excluding self-pairs
        overlap  = torch.mm(labels, labels.T)
        eye      = torch.eye(B, device=embeddings.device)
        pos_mask = (overlap > 0).float() * (1 - eye)

        if pos_mask.sum() == 0:
            return torch.tensor(0.0, device=embeddings.device)

        exp_sim  = torch.exp(sim) * (1 - eye)
        pos_sum  = (exp_sim * pos_mask).sum(dim=1)
        all_sum  = exp_sim.sum(dim=1)

        has_pos  = pos_mask.sum(dim=1) > 0
        loss     = -torch.log(pos_sum[has_pos] / (all_sum[has_pos] + 1e-8))
        return loss.mean()

    def forward(
        self,
        logits: torch.Tensor,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> dict:
        """
        Args:
            logits:     Raw classification logits (B, N_MOODS).
            embeddings: L2-normalised embeddings (B, D).
            labels:     Multi-hot label vectors (B, N_MOODS).

        Returns:
            dict with 'total', 'bce', 'contrastive' loss tensors.
        """
        bce_loss  = self.bce(logits, labels)
        cont_loss = self._contrastive_loss(embeddings, labels)
        total     = (1 - self.alpha) * bce_loss + self.alpha * cont_loss
        return {"total": total, "bce": bce_loss, "contrastive": cont_loss}


# ── Training & evaluation ─────────────────────────────────────────────────────

def train_one_epoch(
    model: MelspecCNNEncoder,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: CombinedLoss,
    device: torch.device,
) -> dict:
    """
    Run one training epoch over the given DataLoader.

    Args:
        model:     MelspecCNNEncoder in training mode.
        loader:    Training DataLoader.
        optimizer: Gradient descent optimiser.
        criterion: CombinedLoss instance.
        device:    Target device.

    Returns:
        Dict with average 'total', 'bce', 'contrastive' loss values.
    """
    model.train()
    total_loss = bce_loss = cont_loss = 0.0

    for batch in tqdm(loader, desc="  Train", leave=False):
        mel    = batch["melspec"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        out    = model(mel)
        losses = criterion(out["logits"], out["embedding"], labels)
        losses["total"].backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += losses["total"].item()
        bce_loss   += losses["bce"].item()
        cont_loss  += losses["contrastive"].item()

    n = len(loader)
    return {"total": total_loss/n, "bce": bce_loss/n, "contrastive": cont_loss/n}


@torch.no_grad()
def evaluate(
    model: MelspecCNNEncoder,
    loader: DataLoader,
    criterion: CombinedLoss,
    device: torch.device,
) -> dict:
    """
    Evaluate the model on a validation or test DataLoader.

    Computes loss values and macro-averaged ROC-AUC and PR-AUC,
    skipping tags with no positive examples in the split.

    Args:
        model:     Trained MelspecCNNEncoder.
        loader:    Evaluation DataLoader.
        criterion: CombinedLoss instance.
        device:    Target device.

    Returns:
        Dict with 'total', 'bce', 'contrastive', 'roc_auc', 'pr_auc'.
    """
    model.eval()
    total_loss = bce_loss = cont_loss = 0.0
    all_logits: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    for batch in tqdm(loader, desc="  Eval ", leave=False):
        mel    = batch["melspec"].to(device)
        labels = batch["label"].to(device)

        out    = model(mel)
        losses = criterion(out["logits"], out["embedding"], labels)

        total_loss += losses["total"].item()
        bce_loss   += losses["bce"].item()
        cont_loss  += losses["contrastive"].item()

        all_logits.append(torch.sigmoid(out["logits"]).cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    all_logits_arr = np.vstack(all_logits)
    all_labels_arr = np.vstack(all_labels)

    roc_aucs, pr_aucs = [], []
    for j in range(N_MOODS):
        if all_labels_arr[:, j].sum() > 0:
            roc_aucs.append(
                roc_auc_score(all_labels_arr[:, j], all_logits_arr[:, j]))
            pr_aucs.append(
                average_precision_score(all_labels_arr[:, j], all_logits_arr[:, j]))

    n = len(loader)
    return {
        "total":       total_loss / n,
        "bce":         bce_loss / n,
        "contrastive": cont_loss / n,
        "roc_auc":     float(np.mean(roc_aucs)) if roc_aucs else 0.0,
        "pr_auc":      float(np.mean(pr_aucs))  if pr_aucs  else 0.0,
    }


def train_audio_encoder(
    model: MelspecCNNEncoder,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: CombinedLoss,
    device: torch.device,
    epochs: int = 30,
    lr: float = 1e-4,
    models_dir: str = "./models",
) -> tuple[MelspecCNNEncoder, dict]:
    """
    Full training loop for the audio CNN encoder with early stopping on
    validation ROC-AUC.  Saves the best checkpoint to models_dir.

    Args:
        model:        Freshly initialised MelspecCNNEncoder.
        train_loader: Training DataLoader.
        val_loader:   Validation DataLoader.
        criterion:    CombinedLoss instance.
        device:       Target device.
        epochs:       Number of training epochs.
        lr:           Initial learning rate for AdamW.
        models_dir:   Directory where checkpoints are saved.

    Returns:
        (trained_model, history) where history is a dict of train/val metric lists.
    """
    os.makedirs(models_dir, exist_ok=True)
    best_ckpt = os.path.join(models_dir, "audio_encoder_best.pt")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6)

    history   = {"train": [], "val": []}
    best_roc  = 0.0

    print(f"Training audio encoder for {epochs} epochs …\n")
    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch:02d}/{epochs}")

        train_m = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_m   = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history["train"].append(train_m)
        history["val"].append(val_m)

        print(f"  Train loss: {train_m['total']:.4f}  "
              f"(BCE={train_m['bce']:.4f} Cont={train_m['contrastive']:.4f})")
        print(f"  Val   loss: {val_m['total']:.4f}  "
              f"ROC-AUC={val_m['roc_auc']:.4f}  PR-AUC={val_m['pr_auc']:.4f}")

        if val_m["roc_auc"] > best_roc:
            best_roc = val_m["roc_auc"]
            torch.save(model.state_dict(), best_ckpt)
            print(f"  ✅ New best saved  (ROC-AUC={best_roc:.4f})")
        print()

    print(f"Training complete. Best val ROC-AUC: {best_roc:.4f}")
    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    return model, history


# ── CLIP projection training ──────────────────────────────────────────────────

def build_clip_mood_pairs(
    df_train: pd.DataFrame,
    audio_encoder: MelspecCNNEncoder,
    clip_model: nn.Module,
    preprocess,
    melspec_dir: str,
    device: torch.device,
    batch_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pre-compute (CLIP embedding, audio mood prediction) pairs for all
    training tracks by converting each mel-spectrogram to an RGB thumbnail
    and encoding it with CLIP.  Audio encoder pseudo-labels are used as
    training targets for the projection MLP.

    Args:
        df_train:      Training-split DataFrame.
        audio_encoder: Trained MelspecCNNEncoder (produces pseudo-labels).
        clip_model:    Frozen CLIP model.
        preprocess:    CLIP image preprocessing transform.
        melspec_dir:   Root directory of .npy mel-spectrogram files.
        device:        Target device.
        batch_size:    Batch size for CLIP embedding computation.

    Returns:
        (clip_embeddings, mood_labels) — both as CPU float tensors,
        shapes (N, 512) and (N, N_MOODS) respectively.
    """
    ds     = JamendoMelspecDataset(df_train, melspec_dir, augment=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    all_clip_embs: list[torch.Tensor]  = []
    all_mood_lbls: list[torch.Tensor]  = []

    audio_encoder.eval()
    clip_model.eval()

    print("Pre-computing CLIP embeddings + audio pseudo-labels …")
    with torch.no_grad():
        for batch in tqdm(loader, desc="CLIP pairs"):
            mel    = batch["melspec"].to(device)
            out    = audio_encoder(mel)
            probs  = torch.sigmoid(out["logits"])  # (B, N_MOODS)

            # Convert each mel-spec to a 3-channel RGB image for CLIP
            mel_np      = mel.cpu().numpy()         # (B, 1, n_mels, T)
            clip_batch: list[torch.Tensor] = []

            for i in range(mel_np.shape[0]):
                m       = mel_np[i, 0]
                m       = (m - m.min()) / (m.max() - m.min() + 1e-8)
                m_rgb   = Image.fromarray(
                    np.stack([(m * 255).astype(np.uint8)] * 3, axis=-1))
                img_t   = preprocess(m_rgb).unsqueeze(0).to(device)
                emb     = clip_model.encode_image(img_t)
                clip_batch.append(F.normalize(emb, dim=-1))

            all_clip_embs.append(torch.cat(clip_batch, dim=0).cpu())
            all_mood_lbls.append(probs.cpu())

    return torch.vstack(all_clip_embs), torch.vstack(all_mood_lbls)


def train_clip_projection(
    clip_embeddings: torch.Tensor,
    mood_labels: torch.Tensor,
    device: torch.device,
    epochs: int = 10,
    lr: float = 1e-3,
    batch_size: int = 64,
    models_dir: str = "./models",
) -> CLIPToMoodProjection:
    """
    Train the CLIPToMoodProjection MLP on pre-computed pseudo-label pairs.

    Args:
        clip_embeddings: Pre-computed CLIP embeddings (N, 512) on CPU.
        mood_labels:     Audio encoder pseudo-labels (N, N_MOODS) on CPU.
        device:          Target device.
        epochs:          Training epochs (default 10).
        lr:              Adam learning rate.
        batch_size:      Mini-batch size.
        models_dir:      Directory to save the trained projection weights.

    Returns:
        Trained CLIPToMoodProjection model in eval mode.
    """
    dataset    = TensorDataset(clip_embeddings, mood_labels)
    loader     = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            num_workers=0)
    projection = CLIPToMoodProjection().to(device)
    optimizer  = torch.optim.Adam(projection.parameters(), lr=lr)
    criterion  = nn.BCEWithLogitsLoss()

    print("\nTraining CLIP → mood projection …")
    for epoch in range(1, epochs + 1):
        projection.train()
        total_loss = 0.0
        for clip_emb, mood_lbl in loader:
            clip_emb = clip_emb.to(device)
            mood_lbl = mood_lbl.to(device)
            optimizer.zero_grad()
            logits = projection(clip_emb)
            loss   = criterion(logits, mood_lbl)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"  Epoch {epoch:2d}/{epochs}  loss={total_loss/len(loader):.4f}")

    proj_path = os.path.join(models_dir, "clip_projection.pt")
    torch.save(projection.state_dict(), proj_path)
    print(f"✅  Projection saved → {proj_path}")
    return projection.eval()


# ── Song embedding index ──────────────────────────────────────────────────────

@torch.no_grad()
def build_song_index(
    model: MelspecCNNEncoder,
    df_all: pd.DataFrame,
    melspec_dir: str,
    device: torch.device,
    embed_dir: str = "./data/embeddings",
    batch_size: int = 64,
) -> dict:
    """
    Pre-compute and cache 512-d embeddings for every track in the dataset.

    These embeddings are used at inference time so recommendation queries
    only require a single forward pass through CLIP + the projection layer.

    Args:
        model:       Trained MelspecCNNEncoder.
        df_all:      Full metadata DataFrame (train + val + test).
        melspec_dir: Root directory of mel-spectrogram .npy files.
        device:      Target device.
        embed_dir:   Directory to save embedding .npy and metadata .json.
        batch_size:  Batch size for embedding computation.

    Returns:
        dict with 'embeddings' (np.ndarray), 'track_ids' (list[str]),
        'metadata' (list[dict]).
    """
    os.makedirs(embed_dir, exist_ok=True)
    model.eval()

    ds     = JamendoMelspecDataset(df_all, melspec_dir, augment=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)

    all_embs:      list[np.ndarray] = []
    all_track_ids: list[str]        = []

    for batch in tqdm(loader, desc="Building song index"):
        mel = batch["melspec"].to(device)
        out = model(mel)
        all_embs.append(out["embedding"].cpu().numpy())
        all_track_ids.extend(batch["track_id"])

    embeddings = np.vstack(all_embs)

    # Build JSON-serialisable metadata
    id_to_row  = df_all.set_index("TRACK_ID").to_dict("index")
    metadata   = [
        {
            "track_id":   tid,
            "mood_tags":  id_to_row.get(tid, {}).get("mood_tags", []),
            "jamendo_url": f"https://www.jamendo.com/track/{tid}",
        }
        for tid in all_track_ids
    ]

    np.save(os.path.join(embed_dir, "song_embeddings.npy"), embeddings)
    with open(os.path.join(embed_dir, "song_metadata.json"), "w") as fh:
        json.dump({"track_ids": all_track_ids, "metadata": metadata}, fh)

    print(f"✅  Song index: {embeddings.shape[0]:,} tracks × {embeddings.shape[1]}-d")
    return {"embeddings": embeddings, "track_ids": all_track_ids, "metadata": metadata}


# ── Inference ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def recommend_from_image(
    image: Image.Image,
    clip_model: nn.Module,
    projection: CLIPToMoodProjection,
    audio_encoder: MelspecCNNEncoder,
    df_all: pd.DataFrame,
    melspec_dir: str,
    device: torch.device,
    top_k: int = 10,
) -> pd.DataFrame:
    """
    Recommend top-K songs for a query image.

    Pipeline:
        1. CLIP encodes the image → 512-d embedding.
        2. CLIPToMoodProjection maps it → 15-d mood probability vector.
        3. Audio encoder scores all songs in the same 15-d mood space.
        4. Cosine similarity ranks songs by mood match.

    Args:
        image:         PIL Image query.
        clip_model:    Frozen CLIP model + preprocess stored in model.preprocess.
        projection:    Trained CLIPToMoodProjection.
        audio_encoder: Trained MelspecCNNEncoder.
        df_all:        Full metadata DataFrame.
        melspec_dir:   Root directory of mel-spectrogram .npy files.
        device:        Target device.
        top_k:         Number of recommendations to return.

    Returns:
        DataFrame with columns: rank, track_id, mood_tags, similarity, jamendo_url.
    """
    # Step 1 & 2: image → CLIP → mood probs
    preprocess  = open_clip.image_transform(224, is_train=False)
    img_tensor  = preprocess(image).unsqueeze(0).to(device)
    clip_emb    = clip_model.encode_image(img_tensor)
    clip_emb    = F.normalize(clip_emb, dim=-1)
    mood_logits = projection(clip_emb)
    mood_probs  = torch.sigmoid(mood_logits).squeeze().cpu().numpy()

    print("Image mood scores:")
    for tag, score in sorted(zip(MOOD_TAGS, mood_probs), key=lambda x: -x[1]):
        print(f"  {tag:<14} {score:.3f}")

    # Step 3: audio encoder mood probs for all songs
    ds     = JamendoMelspecDataset(df_all, melspec_dir, augment=False)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=4)

    all_song_probs: list[np.ndarray] = []
    all_track_ids:  list[str]        = []

    audio_encoder.eval()
    for batch in tqdm(loader, desc="Scoring songs"):
        mel   = batch["melspec"].to(device)
        out   = audio_encoder(mel)
        probs = torch.sigmoid(out["logits"]).cpu().numpy()
        all_song_probs.append(probs)
        all_track_ids.extend(batch["track_id"])

    all_song_probs = np.vstack(all_song_probs)

    # Step 4: cosine similarity in mood space
    img_vec  = mood_probs / (np.linalg.norm(mood_probs) + 1e-8)
    norms    = np.linalg.norm(all_song_probs, axis=1, keepdims=True) + 1e-8
    sims     = (all_song_probs / norms) @ img_vec
    top_idx  = np.argsort(sims)[::-1][:top_k]

    id_to_meta = df_all.set_index("TRACK_ID").to_dict("index")
    results    = []
    for rank, idx in enumerate(top_idx, 1):
        tid  = all_track_ids[idx]
        meta = id_to_meta.get(tid, {})
        results.append({
            "rank":        rank,
            "track_id":    tid,
            "mood_tags":   ", ".join(meta.get("mood_tags", [])),
            "similarity":  float(sims[idx]),
            "jamendo_url": f"https://www.jamendo.com/track/{tid}",
        })
    return pd.DataFrame(results)


# ── Per-tag evaluation ────────────────────────────────────────────────────────

@torch.no_grad()
def per_tag_metrics(
    model: MelspecCNNEncoder,
    loader: DataLoader,
    device: torch.device,
) -> pd.DataFrame:
    """
    Compute ROC-AUC and PR-AUC for each mood tag on a given DataLoader.

    Args:
        model:  Trained MelspecCNNEncoder.
        loader: DataLoader (typically test set).
        device: Target device.

    Returns:
        DataFrame sorted by ROC-AUC (descending) with columns:
        tag, n_positive, roc_auc, pr_auc.
    """
    model.eval()
    all_logits: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    for batch in tqdm(loader, desc="Per-tag metrics"):
        mel    = batch["melspec"].to(device)
        labels = batch["label"]
        out    = model(mel)
        all_logits.append(torch.sigmoid(out["logits"]).cpu().numpy())
        all_labels.append(labels.numpy())

    all_logits_arr = np.vstack(all_logits)
    all_labels_arr = np.vstack(all_labels)

    rows = []
    for j, tag in enumerate(MOOD_TAGS):
        n_pos = int(all_labels_arr[:, j].sum())
        if n_pos > 0:
            roc = float(roc_auc_score(all_labels_arr[:, j], all_logits_arr[:, j]))
            pr  = float(average_precision_score(all_labels_arr[:, j], all_logits_arr[:, j]))
        else:
            roc = pr = float("nan")
        rows.append({"tag": tag, "n_positive": n_pos, "roc_auc": roc, "pr_auc": pr})

    return pd.DataFrame(rows).sort_values("roc_auc", ascending=False)


# ── Error analysis ─────────────────────────────────────────────────────────────

@torch.no_grad()
def find_mispredictions(
    model: MelspecCNNEncoder,
    loader: DataLoader,
    device: torch.device,
    n: int = 5,
    threshold: float = 0.5,
) -> list[dict]:
    """
    Identify the n tracks with the largest mean absolute prediction error.

    A high error indicates the model's predicted mood probabilities diverge
    significantly from the ground-truth binary labels.

    Args:
        model:     Trained MelspecCNNEncoder.
        loader:    DataLoader (typically test set).
        device:    Target device.
        n:         Number of worst mispredictions to return.
        threshold: Probability threshold for binary prediction labels.

    Returns:
        List of dicts with keys: track_id, true_tags, pred_tags, error.
    """
    model.eval()
    all_probs:  list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_ids:    list[str]        = []

    for batch in tqdm(loader, desc="Error analysis"):
        mel    = batch["melspec"].to(device)
        labels = batch["label"]
        out    = model(mel)
        all_probs.append(torch.sigmoid(out["logits"]).cpu().numpy())
        all_labels.append(labels.numpy())
        all_ids.extend(batch["track_id"])

    all_probs_arr  = np.vstack(all_probs)
    all_labels_arr = np.vstack(all_labels)
    errors         = np.abs(all_probs_arr - all_labels_arr).mean(axis=1)
    worst          = np.argsort(errors)[::-1][:n]

    results = []
    for idx in worst:
        true_tags = [MOOD_TAGS[j] for j in range(N_MOODS)
                     if all_labels_arr[idx, j] == 1]
        pred_tags = [MOOD_TAGS[j] for j in range(N_MOODS)
                     if all_probs_arr[idx, j] >= threshold]
        results.append({
            "track_id":  all_ids[idx],
            "true_tags": true_tags,
            "pred_tags": pred_tags,
            "error":     float(errors[idx]),
        })
    return results


# ── Training set size experiment ──────────────────────────────────────────────

def run_size_experiment(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    melspec_dir: str,
    device: torch.device,
    fractions: tuple = (0.1, 0.25, 0.5, 0.75, 1.0),
    quick_epochs: int = 5,
    batch_size: int = 32,
) -> pd.DataFrame:
    """
    Experiment: how does training set size affect validation ROC-AUC?

    Trains a fresh model on subsets of the training data and records the
    validation ROC-AUC at the end of quick_epochs epochs.

    Args:
        df_train:     Full training DataFrame.
        df_val:       Validation DataFrame.
        melspec_dir:  Root directory of mel-spectrogram .npy files.
        device:       Target device.
        fractions:    Dataset size fractions to test.
        quick_epochs: Epochs per run (keep small for speed).
        batch_size:   Mini-batch size.

    Returns:
        DataFrame with columns: fraction, n_samples, val_roc_auc.
    """
    val_ds  = JamendoMelspecDataset(df_val, melspec_dir, augment=False)
    val_ldr = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    crit    = CombinedLoss()
    results = []

    for frac in fractions:
        n     = max(int(len(df_train) * frac), 32)
        df_sub = df_train.sample(n=n, random_state=SEED)
        print(f"\n─── Fraction {frac:.0%}  n={n:,} ───")

        model    = MelspecCNNEncoder().to(device)
        opt      = torch.optim.AdamW(model.parameters(), lr=1e-4)
        train_ds = JamendoMelspecDataset(df_sub, melspec_dir, augment=True)
        train_ldr= DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=2)

        for _ in range(quick_epochs):
            train_one_epoch(model, train_ldr, opt, crit, device)

        m = evaluate(model, val_ldr, crit, device)
        print(f"Val ROC-AUC: {m['roc_auc']:.4f}")
        results.append({"fraction": frac, "n_samples": n, "val_roc_auc": m["roc_auc"]})

    return pd.DataFrame(results)


# ── Artifact saving ───────────────────────────────────────────────────────────

def save_artifacts(
    audio_encoder: MelspecCNNEncoder,
    projection: CLIPToMoodProjection,
    test_metrics: dict,
    models_dir: str = "./models",
) -> None:
    """
    Save the final audio encoder and CLIP projection as a combined checkpoint.

    Files written:
        models/audio_encoder_final.pt  – full checkpoint with config + metrics
        models/clip_projection.pt      – projection MLP state dict
        models/mood_tags.json          – ordered list of mood tag strings

    Args:
        audio_encoder: Trained MelspecCNNEncoder.
        projection:    Trained CLIPToMoodProjection.
        test_metrics:  Dict from evaluate() on the test set.
        models_dir:    Target directory (created if absent).
    """
    os.makedirs(models_dir, exist_ok=True)

    checkpoint = {
        "model_state_dict":      audio_encoder.state_dict(),
        "projection_state_dict": projection.state_dict(),
        "mood_tags":             MOOD_TAGS,
        "n_moods":               N_MOODS,
        "test_roc_auc":          test_metrics["roc_auc"],
        "test_pr_auc":           test_metrics["pr_auc"],
    }
    torch.save(checkpoint,
               os.path.join(models_dir, "audio_encoder_final.pt"))
    torch.save(projection.state_dict(),
               os.path.join(models_dir, "clip_projection.pt"))
    with open(os.path.join(models_dir, "mood_tags.json"), "w") as fh:
        json.dump(MOOD_TAGS, fh)

    print(f"\n✅  Artifacts saved to {models_dir}/")
    print(f"  audio_encoder_best.pt    – best val checkpoint")
    print(f"  audio_encoder_final.pt   – full checkpoint + metrics")
    print(f"  clip_projection.pt       – CLIP → mood projection")
    print(f"  mood_tags.json           – ordered tag list")
    print(f"\n  Test ROC-AUC : {test_metrics['roc_auc']:.4f}")
    print(f"  Test PR-AUC  : {test_metrics['pr_auc']:.4f}")
    print(f"  Test BCE loss: {test_metrics['bce']:.4f}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the deep learning audio CNN for Visual Vibe.")
    parser.add_argument("--processed-dir", default="./data/processed",
                        help="Directory with processed parquets + mlb.pkl.")
    parser.add_argument("--melspec-dir",   default="./data/melspecs",
                        help="Root directory of mel-spectrogram .npy files.")
    parser.add_argument("--models-dir",    default="./models",
                        help="Output directory for model checkpoints.")
    parser.add_argument("--embed-dir",     default="./data/embeddings",
                        help="Output directory for song embedding index.")
    parser.add_argument("--epochs",  type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr",      type=float, default=1e-4)
    parser.add_argument("--embed-dim", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--alpha",   type=float, default=0.0,
                        help="Contrastive loss weight (0=BCE only).")
    parser.add_argument("--quick",   action="store_true",
                        help="5 epochs + no size experiment (smoke-test).")
    parser.add_argument("--skip-experiment", action="store_true",
                        help="Skip the training set size experiment.")
    args = parser.parse_args()

    set_seeds(SEED)
    device = get_device()

    # ── Load processed data ───────────────────────────────────────────────────
    import pickle
    print("Loading processed data …")
    df_train = pd.read_parquet(f"{args.processed_dir}/df_train.parquet")
    df_val   = pd.read_parquet(f"{args.processed_dir}/df_val.parquet")
    df_test  = pd.read_parquet(f"{args.processed_dir}/df_test.parquet")
    df_all   = pd.read_parquet(f"{args.processed_dir}/df_all.parquet")

    with open(f"{args.processed_dir}/mlb.pkl", "rb") as fh:
        mlb = pickle.load(fh)

    # Add label_vector column required by the Dataset
    for df_split in [df_train, df_val, df_test, df_all]:
        df_split["label_vector"] = list(
            mlb.transform(df_split["mood_tags"].tolist()))

    print(f"Train: {len(df_train):,}  Val: {len(df_val):,}  Test: {len(df_test):,}")

    # ── DataLoaders ───────────────────────────────────────────────────────────
    train_ds = JamendoMelspecDataset(df_train, args.melspec_dir, augment=True)
    val_ds   = JamendoMelspecDataset(df_val,   args.melspec_dir, augment=False)
    test_ds  = JamendoMelspecDataset(df_test,  args.melspec_dir, augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)

    # ── Load CLIP ─────────────────────────────────────────────────────────────
    clip_model, clip_preprocess = load_clip_model(device=device)

    # ── Train audio encoder ───────────────────────────────────────────────────
    audio_encoder = MelspecCNNEncoder(
        embed_dim=args.embed_dim).to(device)
    criterion     = CombinedLoss(
        temperature=args.temperature, alpha=args.alpha)

    epochs = 5 if args.quick else args.epochs
    audio_encoder, history = train_audio_encoder(
        audio_encoder, train_loader, val_loader, criterion, device,
        epochs=epochs, lr=args.lr, models_dir=args.models_dir)

    # ── Test set evaluation ───────────────────────────────────────────────────
    print("\n=== Audio CNN — Test Set ===")
    test_metrics = evaluate(audio_encoder, test_loader, criterion, device)
    print(f"  ROC-AUC  = {test_metrics['roc_auc']:.4f}")
    print(f"  PR-AUC   = {test_metrics['pr_auc']:.4f}")
    print(f"  BCE loss = {test_metrics['bce']:.4f}")

    # ── Per-tag breakdown ─────────────────────────────────────────────────────
    print("\n=== Per-tag Test Metrics ===")
    tag_df = per_tag_metrics(audio_encoder, test_loader, device)
    print(tag_df.to_string(index=False))

    # ── Error analysis ────────────────────────────────────────────────────────
    print("\n=== Top-5 Mispredictions ===")
    misses = find_mispredictions(audio_encoder, test_loader, device)
    for i, m in enumerate(misses, 1):
        print(f"\n#{i}  Track {m['track_id']}  (error={m['error']:.3f})")
        print(f"     True: {m['true_tags']}")
        print(f"     Pred: {m['pred_tags']}")

    # ── Training set size experiment ──────────────────────────────────────────
    if not args.skip_experiment and not args.quick:
        print("\n=== Training Set Size Experiment ===")
        exp_df = run_size_experiment(
            df_train, df_val, args.melspec_dir, device)
        print(exp_df.to_string(index=False))
        exp_df.to_csv(f"{args.processed_dir}/size_experiment.csv", index=False)

    # ── Train CLIP projection ─────────────────────────────────────────────────
    clip_embs, mood_lbls = build_clip_mood_pairs(
        df_train, audio_encoder, clip_model, clip_preprocess,
        args.melspec_dir, device)

    projection = train_clip_projection(
        clip_embs, mood_lbls, device, models_dir=args.models_dir)

    # ── Build song index ──────────────────────────────────────────────────────
    build_song_index(
        audio_encoder, df_all, args.melspec_dir,
        device, embed_dir=args.embed_dir)

    # ── Save final artifacts ──────────────────────────────────────────────────
    save_artifacts(audio_encoder, projection, test_metrics,
                   models_dir=args.models_dir)
