"""
VibeTrack — Flask Backend
Mood-based Song Recommendation using ResNet-18 CNN on MTG-Jamendo
"""

import os
import numpy as np
from pathlib import Path

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)

# ── Config ────────────────────────────────────────────────────────────────────
MOOD_TAGS = [
    'happy', 'sad', 'energetic', 'calm', 'dark', 'epic',
    'romantic', 'aggressive', 'relaxing', 'melancholic',
    'uplifting', 'dramatic', 'peaceful', 'tense', 'fun'
]

MODEL_DIR     = os.environ.get('MODEL_DIR', './models')
MELSPEC_DIR   = os.environ.get('MELSPEC_DIR', './data/melspecs')
METADATA_PATH = os.environ.get('METADATA_PATH', './data/autotagging_moodtheme.tsv')
PORT          = int(os.environ.get('PORT', 5000))
DEVICE        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TOP_K         = 10

print(f'Using device: {DEVICE}')


# ── Model Definition ──────────────────────────────────────────────────────────
class MelspecCNNEncoder(nn.Module):
    """ResNet-18 CNN encoder for mel-spectrograms."""

    def __init__(self, embed_dim=512, n_moods=15):
        super().__init__()
        backbone = models.resnet18(pretrained=False)
        backbone.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(feature_dim, embed_dim)
        )
        self.mood_classifier = nn.Linear(embed_dim, n_moods)

    def forward(self, x):
        features       = self.backbone(x)
        embedding      = self.projection(features)
        embedding_norm = F.normalize(embedding, dim=-1)
        logits         = self.mood_classifier(embedding_norm)
        return {'embedding': embedding_norm, 'logits': logits}


# ── Load Model ────────────────────────────────────────────────────────────────
print('Loading audio encoder...')
audio_encoder = MelspecCNNEncoder(embed_dim=512, n_moods=15).to(DEVICE)
audio_encoder.load_state_dict(
    torch.load(
        os.path.join(MODEL_DIR, 'audio_encoder_best.pt'),
        map_location=DEVICE
    )
)
audio_encoder.eval()
print('Model loaded.')


# ── Load Metadata ─────────────────────────────────────────────────────────────
def load_metadata():
    """Load MTG-Jamendo TSV metadata."""
    rows = []
    with open(METADATA_PATH, 'r') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            parts = line.strip().split('\t')
            if len(parts) < 5:
                continue
            tags = [
                t.replace('mood/theme---', '').strip()
                for t in parts[5:]
                if t.startswith('mood/theme---') and
                t.replace('mood/theme---', '').strip() in MOOD_TAGS
            ]
            if tags:
                rows.append({
                    'track_id':  parts[0],
                    'path':      parts[3],
                    'mood_tags': tags
                })
    return rows


def has_melspec(song: dict) -> bool:
    """Check if a mel-spectrogram file exists for this song."""
    path       = song['path']
    subdir     = path.split('/')[0]
    numeric_id = path.split('/')[1].replace('.mp3', '')
    npy_path   = Path(MELSPEC_DIR) / subdir / f'{numeric_id}.npy'
    return npy_path.exists()


print('Loading metadata...')
ALL_SONGS = load_metadata()
print(f'Total songs in metadata: {len(ALL_SONGS):,}')

# ── Filter to only songs with available melspec files ─────────────────────────
SONGS = [s for s in ALL_SONGS if has_melspec(s)]
print(f'Songs with available melspecs: {len(SONGS):,} / {len(ALL_SONGS):,}')

if len(SONGS) == 0:
    print('WARNING: No melspec files found!')
    print(f'  MELSPEC_DIR = {MELSPEC_DIR}')
    print(f'  Exists: {Path(MELSPEC_DIR).exists()}')


# ── Mel-spec Loading ──────────────────────────────────────────────────────────
def load_melspec(path: str, n_mels=128, time_frames=1300) -> np.ndarray:
    """Load and preprocess a mel-spectrogram .npy file."""
    subdir     = path.split('/')[0]
    numeric_id = path.split('/')[1].replace('.mp3', '')
    npy_path   = Path(MELSPEC_DIR) / subdir / f'{numeric_id}.npy'

    if not npy_path.exists():
        return np.zeros((n_mels, time_frames), dtype=np.float32)

    mel = np.load(str(npy_path)).astype(np.float32)
    t = mel.shape[1]
    if t < time_frames:
        mel = np.pad(mel, ((0, 0), (0, time_frames - t)))
    else:
        mel = mel[:, :time_frames]

    mel_min, mel_max = mel.min(), mel.max()
    if mel_max > mel_min:
        mel = 2 * (mel - mel_min) / (mel_max - mel_min) - 1
    return mel


# ── Pre-compute song mood scores at startup ───────────────────────────────────
print('Pre-computing song mood scores...')
SONG_MOOD_PROBS = []

with torch.no_grad():
    for i, song in enumerate(SONGS):
        mel   = load_melspec(song['path'])
        mel_t = torch.from_numpy(mel).unsqueeze(0).unsqueeze(0).to(DEVICE)
        out   = audio_encoder(mel_t)
        probs = torch.sigmoid(out['logits']).squeeze().cpu().numpy()
        SONG_MOOD_PROBS.append(probs)

        if (i + 1) % 100 == 0:
            print(f'  Scored {i+1}/{len(SONGS)} songs...')

SONG_MOOD_PROBS = np.vstack(SONG_MOOD_PROBS) if SONG_MOOD_PROBS else np.zeros((0, 15))
print(f'Pre-computed {len(SONG_MOOD_PROBS):,} song mood vectors.')


# ── Recommendation Logic ──────────────────────────────────────────────────────
def recommend_by_mood(mood: str, top_k: int = TOP_K) -> list:
    """
    Recommend top-K songs for a given mood using cosine similarity
    between a one-hot mood query and pre-computed song mood vectors.
    """
    if len(SONG_MOOD_PROBS) == 0:
        return []

    mood_vec = np.zeros(len(MOOD_TAGS), dtype=np.float32)
    mood_vec[MOOD_TAGS.index(mood)] = 1.0

    norms        = np.linalg.norm(SONG_MOOD_PROBS, axis=1, keepdims=True) + 1e-8
    normed_probs = SONG_MOOD_PROBS / norms
    similarities = normed_probs @ mood_vec

    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []
    for rank, idx in enumerate(top_indices, 1):
        song = SONGS[idx]
        tid  = song['track_id']
        num  = tid.replace('track_', '').lstrip('0') or '0'
        results.append({
            'rank':        rank,
            'track_id':   tid,
            'mood_tags':  song['mood_tags'],
            'similarity': float(similarities[idx]),
            'jamendo_url': f'https://www.jamendo.com/track/{num}'
        })

    return results


# ── API Routes ────────────────────────────────────────────────────────────────
@app.route('/api/moods', methods=['GET'])
def get_moods():
    return jsonify({'moods': MOOD_TAGS})


@app.route('/api/recommend', methods=['POST'])
def recommend():
    """
    POST /api/recommend
    Body: { "mood": "happy", "top_k": 10 }
    Returns: { "mood": "happy", "results": [...] }
    """
    data  = request.get_json()
    mood  = data.get('mood', '').lower().strip()
    top_k = int(data.get('top_k', TOP_K))

    if mood not in MOOD_TAGS:
        return jsonify({
            'error': f'Invalid mood. Choose from: {MOOD_TAGS}'
        }), 400

    results = recommend_by_mood(mood, top_k=top_k)
    return jsonify({'mood': mood, 'results': results})


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status':       'ok',
        'songs_loaded': len(SONGS),
        'songs_total':  len(ALL_SONGS),
        'device':       str(DEVICE),
        'moods':        MOOD_TAGS
    })


# ── Serve frontend ────────────────────────────────────────────────────────────
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path and (Path('static') / path).exists():
        return send_from_directory('static', path)
    return send_from_directory('static', 'index.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT, debug=False)
