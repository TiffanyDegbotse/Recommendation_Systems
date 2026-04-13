# VibeTrack — Mood-Based Music Recommendation

Deep learning music recommendation system built on ResNet-18 CNN trained on MTG-Jamendo.
Select a mood → get top-10 matching songs with direct Jamendo listen links.

**Model Performance:** ROC-AUC 0.72 | PR-AUC 0.22 | Best tag (fun): ROC-AUC 0.89

---

## Folder Structure

```
vibetrack/
├── app.py               ← Flask backend
├── requirements.txt
├── Procfile             ← Railway deployment
├── railway.toml         ← Railway config
├── static/
│   └── index.html       ← Frontend
├── models/
│   └── audio_encoder_best.pt   ← trained model (download from Colab)
└── data/
    ├── autotagging_moodtheme.tsv
    └── melspecs/
        ├── 00/
        ├── 01/
        └── ...
```

---

## Run Locally

### 1. Download model from Colab
```python
# Run in your Colab notebook
from google.colab import files
files.download('models/audio_encoder_best.pt')
```

### 2. Set up folder structure
```bash
mkdir -p models data/melspecs static
# Put audio_encoder_best.pt in models/
# Put autotagging_moodtheme.tsv in data/
# Put your melspecs folders in data/melspecs/
# Put index.html in static/
```

### 3. Install & run
```bash
pip install -r requirements.txt
python app.py
```

Open: **http://localhost:5000**

---

## Deploy to Railway

### 1. Push to GitHub
```bash
git init
git add .
git commit -m "VibeTrack app"
git remote add origin https://github.com/TiffanyDegbotse/Recommendation_Systems.git
git push
```

### 2. Deploy on Railway
1. Go to [railway.app](https://railway.app)
2. New Project → Deploy from GitHub repo
3. Select your repo
4. Add environment variables:
   - `MODEL_DIR` = `/app/models`
   - `MELSPEC_DIR` = `/app/data/melspecs`
   - `METADATA_PATH` = `/app/data/autotagging_moodtheme.tsv`
5. Railway auto-detects Procfile and deploys

> **Note on model files:** GitHub has a 100MB file limit.
> Use Git LFS for `audio_encoder_best.pt`:
> ```bash
> git lfs install
> git lfs track "*.pt"
> git add .gitattributes
> git add models/audio_encoder_best.pt
> git commit -m "Add model via LFS"
> git push
> ```

### 3. Data on Railway
The melspecs (~69GB) are too large for Railway's disk.
Two options:
- **Option A:** Use only the metadata TSV + serve a subset of songs
  (app still works, just with fewer songs in the index)
- **Option B:** Mount a Railway volume and upload melspecs via SSH

For the course demo, Option A is fine — the model still demonstrates
the full recommendation pipeline.

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Frontend app |
| `/api/health` | GET | Health check |
| `/api/moods` | GET | List available moods |
| `/api/recommend` | POST | Get song recommendations |

### POST /api/recommend
```json
// Request
{ "mood": "happy", "top_k": 10 }

// Response
{
  "mood": "happy",
  "results": [
    {
      "rank": 1,
      "track_id": "track_1195933",
      "mood_tags": ["happy"],
      "similarity": 0.2854,
      "jamendo_url": "https://www.jamendo.com/track/1195933"
    },
    ...
  ]
}
```
