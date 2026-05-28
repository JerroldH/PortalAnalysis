# PortalAnalysis — Hand Movement Training & Inference

Shareable package for **training**, **evaluating**, and **running inference** on three MDS-UPDRS hand motor tasks from the UBC Booth system.

| Task | MDS item | Data column |
|------|----------|-------------|
| Finger Tapping | 3.4 | `Finger Normalized Distance` |
| Hand Open/Close | 3.5 | `Normalized Hand Sum Finger Distances` |
| Hand Pronation-Supination | 3.6 | `yaw_rad` |

Each task predicts **severity** (0–3). Symptom column names are listed in `configs/*.json` for future per-symptom models.

---

## Project structure

```
PortalAnalysis/
├── portal_analysis/
│   ├── cli.py                     # train | evaluate | predict
│   ├── config.py                  # Default N:/ paths
│   ├── data/data_loader.py        # Labels + time series loading
│   ├── training/                  # Pipeline, artifacts, metrics
│   ├── classification/            # MiniRocket + augmentation
│   ├── preprocessing/             # Video → pose → distances
│   ├── models/                    # Load/save + path resolution
│   └── inference/                 # Per-task + batch inference
├── configs/                       # Task JSON (committed to git)
│   ├── finger_tapping.json
│   ├── hand_open_close.json
│   └── hand_up_down.json
├── models/                        # Trained artifacts (gitignored)
├── scripts/
│   ├── train_models.py
│   └── run_inference.py
└── config.example.py              # Optional local path override
```

---

## Installation

```bash
conda env create -f environment.yml
conda activate booth_inference
pip install -e .
```

**Data path** (pick one):

- Default: `portal_analysis/config.py` → `N:/Booth_Processed`
- Override: `set PORTAL_DATA_DIR=N:\Booth_Processed`
- Or copy `config.example.py` → `config.py`

---

## Training

Train all three tasks (saves to `models/<task>/latest/`):

```bash
python -m portal_analysis.cli train --tasks all
# or
python scripts/train_models.py
```

Train one task with a version tag:

```bash
python -m portal_analysis.cli train --tasks hand_open_close --version v1.0.0
```

Evaluate on the held-out test set:

```bash
python -m portal_analysis.cli evaluate --model models/hand_open_close/latest
```

**Artifact layout** (share this folder):

```
models/hand_open_close/v1.0.0/
├── classifier.joblib
├── rocket.joblib
└── metadata.json      # versions, metrics, full task config
```

---

## Inference

### From pre-computed distances CSVs

```bash
python scripts/run_inference.py \
    --mode csv \
    --patient-ids SUBJECT_DATE_001 \
    --processed-dir N:/Booth_Processed \
    --model-version latest \
    --output results/inference.csv
```

### From raw videos (full pipeline)

```bash
python scripts/run_inference.py \
    --mode video \
    --patient-ids SUBJECT_DATE_001 \
    --raw-dir "N:/CAMERA Booth Data/Booth" \
    --processed-dir N:/Booth_Processed
```

### Python API

```python
from pathlib import Path
from portal_analysis.inference import FingerTappingPipeline, BatchInferencePipeline

pipe = FingerTappingPipeline(model_version="latest")
result = pipe.run_from_csv("P001_right", Path(".../P001_finger_distances.csv"))
print(result.severity)

batch = BatchInferencePipeline(model_version="v1.0.0")
df = batch.run_from_csvs(["P001"], Path("N:/Booth_Processed"))
```

---

## Tests

```bash
pip install pytest
pytest tests/test_pipeline_smoke.py -v
```

