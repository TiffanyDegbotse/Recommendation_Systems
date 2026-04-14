# VibeTrack 🎵

> A deep learning music recommendation system that maps mood to music — trained on 8,500 real songs from the MTG-Jamendo dataset.

**Live App:** [web-production-470c3.up.railway.app](https://web-production-470c3.up.railway.app)

---

## What It Does

VibeTrack recommends songs based on mood. Select a feeling from 15 mood categories or describe your vibe in plain text — the system finds songs whose acoustic properties best match that emotional space.

Under the hood, a ResNet-18 CNN was trained to listen to music and predict its mood from mel-spectrograms. Every song in the catalog has been scored by the model. At query time, your mood is matched against those scores using cosine similarity to surface the best matches.

---

## Model Performance

| Metric | Score |
|--------|-------|
| ROC-AUC (test) | 0.7197 |
| PR-AUC (test) | 0.2230 |
| Best tag — fun | ROC-AUC 0.89 |
| Best tag — happy | ROC-AUC 0.81 |
| Best tag — epic | ROC-AUC 0.76 |

---

## Architecture

    Mood Query (button or free text)
            ↓
    One-hot or blended mood vector (12-d)
            ↓ cosine similarity
    Song → Mel-spectrogram → ResNet-18 CNN → 12-d mood probability vector
            ↓
    Top-10 Songs Returned with Jamendo listen links

**Training details:**
- Dataset: MTG-Jamendo mood/theme subset (~8,506 tracks, Creative Commons licensed)
- Input: 128 × 1300 mel-spectrogram (30s audio clip)
- Backbone: ResNet-18 modified for single-channel input
- Loss: Multi-label Binary Cross-Entropy
- Optimizer: AdamW with cosine annealing LR schedule
- Augmentation: SpecAugment (time + frequency masking)
- Best checkpoint: epoch 2, early stopping on val ROC-AUC

**Inference:**
- Mood scores pre-computed offline for all songs and stored in song_index.json
- No GPU or audio processing required at runtime
- Recommendations served in milliseconds

---

## Dataset

MTG-Jamendo is an open dataset for music auto-tagging built from Jamendo, a platform for Creative Commons licensed music.

- 55,000+ full audio tracks total
- 18,486 tracks in the mood/theme subset
- 56 mood/theme tags available (12 used after filtering rare classes)
- Audio encoded as 320 kbps MP3, pre-computed mel-spectrograms provided
- Published: Bogdanov et al., ICML 2019

---

## Three Modeling Approaches

As required by the course rubric, three approaches were implemented and evaluated:

**1. Naive Baseline**
Three scalar features (energy, brightness, variance) extracted directly from each mel spectrogram. A rule based decision tree maps these to one of 15 mood buckets using percentile thresholds from the training split, and songs are scored by affinity map overlap. No learning involved.

**2. Classical ML**
A 261 dimensional feature vector per track (128 mel band means, 128 mel band standard deviations, plus global energy and spectral statistics) fed to a OneVsRest Logistic Regression classifier. Retrieval via KNN over the resulting 15d mood probability vectors using cosine similarity.

**3. Deep Learning**
ResNet-18 CNN trained end-to-end on mel-spectrograms with multi-label BCE loss. Achieves ROC-AUC 0.72 on the test set — significantly above both baselines.

---

## Repo Structure

    Recommendation_Systems/
    ├── app.py                          Flask backend API
    ├── requirements.txt                Runtime dependencies
    ├── Procfile                        Railway deployment config
    ├── railway.toml                    Railway build config
    ├── static/
    │   └── index.html                  Frontend UI
    ├── models/
    │   └── audio_encoder_best.pt       Trained ResNet-18 checkpoint
    ├── data/
    │   ├── autotagging_moodtheme.tsv   MTG-Jamendo metadata
    │   └── song_index.json             Pre-computed CNN mood scores
    └── notebooks/
        └── deep_learning_image_to_song.ipynb   Training notebook

---

## Running Locally

    # 1. Clone the repo
    git clone https://github.com/TiffanyDegbotse/Recommendation_Systems.git
    cd Recommendation_Systems

    # 2. Install dependencies
    pip install -r requirements.txt

    # 3. Run
    python app.py

    # 4. Open browser
    # http://localhost:5000

---

## API

**POST /api/recommend**

    Request:  { "mood": "happy", "top_k": 10 }
    Response: { "mood": "happy", "results": [ { "rank": 1, "track_id": "track_1195933", "mood_tags": ["happy"], "similarity": 0.2854, "jamendo_url": "https://www.jamendo.com/track/1195933" }, ... ] }

**POST /api/recommend/text**

    Request:  { "query": "dark mysterious night drive", "top_k": 10 }
    Response: { "query": "...", "matched_moods": ["dark", "dramatic"], "results": [...] }

**GET /api/health**

    Response: { "status": "ok", "songs_loaded": 8506, "moods": [...] }

---

## Deployment

Deployed on Railway via GitHub integration. Auto-deploys on push to main.

Environment variables required:

    DATA_DIR = ./data

---