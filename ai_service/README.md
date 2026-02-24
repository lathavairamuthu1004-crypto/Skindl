# SkinDL AI Service

Runs the prediction API and optionally a trained **9-class skin disease model** (Acne, Eczema, Psoriasis, Melanoma, BCC, SCC, Melasma, Rosacea, Healthy).

## Runtime

- **No model**: OpenCV heuristics (limited accuracy).
- **9-class model** (`skin_full.keras`): Full ML predictions for all conditions.
- **HAM-only model** (`ham_efficientnetb0.keras`): Cancer detection only.

## Training (all 9 diseases, fast)

### 1) Install

```bash
pip install -r requirements-train.txt
```

### 2) Get data

**HAM10000** (melanoma, BCC, SCC, benign):

```bash
python download_datasets.py --ham --out-dir data
# Or: kaggle datasets download -d kmader/skin-cancer-mnist-ham10000
```

**Multi-disease** (acne, eczema, psoriasis, etc.): Create folders and add images:

```
data/skin_diseases/
  acne/      # acne images
  eczema/
  psoriasis/
  melasma/
  rosacea/
  healthy/
```

**Mendeley** (acne, psoriasis, hyperpigmentation): Download from [Mendeley Data](https://data.mendeley.com/datasets/3hckgznc67/1), extract to `data/mendeley_skin/`.

### 3) Train (fast mode)

```bash
# Fast: 3 epochs, no fine-tune (~5–10 min on GPU)
python train_full.py --ham-dir data/HAM10000 --multi-dir data/skin_diseases --fast

# Full: 8+3 epochs
python train_full.py --ham-dir data/HAM10000 --multi-dir data/skin_diseases
```

Outputs: `models/skin_full.keras`, `models/skin_label_map.json`

### 4) Run

```bash
python app.py
```

