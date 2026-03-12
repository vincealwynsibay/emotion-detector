# Combined Emotion Detection (Keras + MediaPipe)

Real-time facial expression detection combining:
- **MediaPipe** for tongue, glare, and shock (priority)
- **Keras CNN** for 7 emotions: anger, disgust, fear, happiness, sadness, surprise, neutral

Displays reaction images with variety from both emotion and cat image sets.

---

## Requirements

- Python 3.9 – 3.12 (MediaPipe does not support 3.13+)

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Add model and images

Place these files in the project root or `assets/` folder:

**Required:**
- `model.keras` — your trained Keras emotion model (in project root)

**Emotion images** (PNG, for Keras emotions):
- `anger.png`, `disgust.png`, `fear.png`, `happiness.png`, `sadness.png`, `surprise.png`, `neutral.png`

**Cat images** (from [MeowCV](https://github.com/reinesana/MeowCV), optional for variety):
- `cat-shock.jpeg`, `cat-tongue.jpeg`, `cat-glare.jpeg`, `larry.jpeg`

You can put all images in the `assets/` folder, or emotion images in the project root (backward compatible).

### 3. Run

```bash
python main.py
```

Press **q** to quit.

---

## Detection priority

1. **Tongue** (MediaPipe) → happiness / cat-tongue
2. **Glare** (MediaPipe) → anger, disgust / cat-glare
3. **Shock** (MediaPipe) → surprise / cat-shock
4. **Otherwise** → Keras emotion (anger, disgust, fear, happiness, sadness, surprise, neutral)

Each state randomly picks from its image pool for variety.
# emotion-detector
