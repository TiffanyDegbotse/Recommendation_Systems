"""
VibeTrack — Flask Backend
Mood-based Song Recommendation using ResNet-18 CNN on MTG-Jamendo
Pre-computed mood scores loaded from song_index.json (no melspecs needed at runtime)
"""

import os
import json
import numpy as np
from pathlib import Path

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)

# ── Config ────────────────────────────────────────────────────────────────────
MOOD_TAGS = [
    'happy', 'sad', 'energetic', 'calm', 'dark', 'epic',
    'romantic', 'aggressive', 'relaxing', 'melancholic',
    'uplifting', 'dramatic', 'peaceful', 'tense', 'fun'
]

KEYWORD_MAP = {
    'happy':       ['joy', 'joyful', 'cheerful', 'bright', 'sunny', 'upbeat', 'positive', 'good vibes'],
    'sad':         ['cry', 'crying', 'grief', 'sorrow', 'blue', 'depressed', 'heartbreak', 'lonely'],
    'energetic':   ['workout', 'gym', 'pump', 'hype', 'running', 'intense', 'power', 'adrenaline'],
    'calm':        ['chill', 'study', 'focus', 'quiet', 'gentle', 'soft', 'mellow', 'lofi'],
    'dark':        ['night', 'mysterious', 'shadow', 'gloomy', 'moody', 'noir', 'sinister'],
    'epic':        ['cinematic', 'grand', 'powerful', 'adventure', 'heroic', 'battle', 'movie'],
    'romantic':    ['love', 'date', 'crush', 'tender', 'sweet', 'intimate', 'passionate', 'wedding'],
    'aggressive':  ['angry', 'rage', 'fury', 'metal', 'hard', 'fierce', 'mad'],
    'relaxing':    ['sleep', 'relax', 'spa', 'meditation', 'ambient', 'rest', 'wind down'],
    'melancholic': ['nostalgic', 'longing', 'wistful', 'bittersweet', 'yearning', 'memories'],
    'uplifting':   ['inspire', 'motivate', 'hope', 'optimistic', 'rise', 'overcome', 'morning'],
    'dramatic':    ['tension', 'suspense', 'emotional', 'climax', 'film', 'scene'],
    'fun':         ['party', 'dance', 'playful', 'silly', 'bounce', 'festive', 'celebration'],
    'tense':       ['anxious', 'nervous', 'thriller', 'scary', 'suspense', 'horror', 'stress'],
    'peaceful':    ['nature', 'forest', 'ocean', 'zen', 'tranquil', 'serene', 'still', 'breeze'],
}

DATA_DIR = os.environ.get('DATA_DIR', './data')
PORT     = int(os.environ.get('PORT', 5000))
TOP_K    = 10


# ── Load Pre-computed Song Index ──────────────────────────────────────────────
print('Loading song index...')
song_index_path = os.path.join(DATA_DIR, 'song_index.json')

with open(song_index_path, 'r') as f:
    SONGS = json.load(f)

SONG_MOOD_PROBS = np.array(
    [s['mood_probs'] for s in SONGS], dtype=np.float32
)  # (N, 15)

print(f'Loaded {len(SONGS):,} songs with pre-computed mood scores.')
print(f'Mood matrix shape: {SONG_MOOD_PROBS.shape}')


# ── Core Retrieval ────────────────────────────────────────────────────────────
def get_top_k(mood_vec: np.ndarray, top_k: int) -> list:
    """
    Cosine similarity between mood query vector and all song mood vectors.

    Args:
        mood_vec: Query vector of shape (15,)
        top_k: Number of results to return

    Returns:
        List of result dicts sorted by similarity descending
    """
    norms        = np.linalg.norm(SONG_MOOD_PROBS, axis=1, keepdims=True) + 1e-8
    normed_probs = SONG_MOOD_PROBS / norms
    similarities = normed_probs @ mood_vec
    top_indices  = np.argsort(similarities)[::-1][:top_k]

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


def recommend_by_mood(mood: str, top_k: int = TOP_K) -> list:
    """Recommend songs for a single mood tag via one-hot query vector."""
    mood_vec = np.zeros(len(MOOD_TAGS), dtype=np.float32)
    mood_vec[MOOD_TAGS.index(mood)] = 1.0
    return get_top_k(mood_vec, top_k)


def recommend_by_text(query: str, top_k: int = TOP_K) -> dict:
    """
    Recommend songs from a free-text mood description.
    Detects mood keywords in the query, blends them into a
    weighted query vector, and retrieves top-K songs.

    Args:
        query: Free-text string e.g. 'dark mysterious night drive'
        top_k: Number of songs to return

    Returns:
        dict with matched_moods list and results list
    """
    query_lower = query.lower()

    # Direct mood tag matches
    matched_moods = [m for m in MOOD_TAGS if m in query_lower]

    # Keyword-based fuzzy matches
    for mood, keywords in KEYWORD_MAP.items():
        if any(kw in query_lower for kw in keywords):
            if mood not in matched_moods:
                matched_moods.append(mood)

    if not matched_moods:
        return {'matched_moods': [], 'results': []}

    # Blend matched moods equally into one query vector
    mood_vec = np.zeros(len(MOOD_TAGS), dtype=np.float32)
    for mood in matched_moods:
        mood_vec[MOOD_TAGS.index(mood)] += 1.0
    mood_vec = mood_vec / (mood_vec.sum() + 1e-8)

    results = get_top_k(mood_vec, top_k)
    return {'matched_moods': matched_moods, 'results': results}


# ── API Routes ────────────────────────────────────────────────────────────────
@app.route('/api/moods', methods=['GET'])
def get_moods():
    """Return available mood tags."""
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


@app.route('/api/recommend/text', methods=['POST'])
def recommend_text():
    """
    POST /api/recommend/text
    Body: { "query": "dark mysterious night drive", "top_k": 10 }
    Returns: { "query": "...", "matched_moods": [...], "results": [...] }
    """
    data  = request.get_json()
    query = data.get('query', '').strip()
    top_k = int(data.get('top_k', TOP_K))

    if not query:
        return jsonify({'error': 'Query cannot be empty'}), 400

    output = recommend_by_text(query, top_k=top_k)

    if not output['matched_moods']:
        return jsonify({
            'error': 'No moods detected. Try words like: happy, dark, chill, workout, romantic...',
            'matched_moods': [],
            'suggestions': MOOD_TAGS
        }), 400

    return jsonify({
        'query':         query,
        'matched_moods': output['matched_moods'],
        'results':       output['results']
    })


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status':       'ok',
        'songs_loaded': len(SONGS),
        'device':       'cpu',
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
