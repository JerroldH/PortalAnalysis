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

## Output structure

End-to-end processing writes intermediate files under the processed data root (`N:/Booth_Processed` by default, or `PORTAL_DATA_DIR`) and final UPDRS severity scores to a results CSV.

### Processed data layout

When running **video mode**, each task and hand side gets its own subtree:

```
Booth_Processed/
├── finger_tapping/
│   ├── right/
│   │   ├── videos/          # optional; source MP4s if copied here
│   │   ├── pose/            # per-video landmark CSVs
│   │   │   └── SUBJECT_DATE_right_finger_tapping.csv
│   │   └── distances/       # kinematic time series for inference
│   │       └── SUBJECT_DATE_right_finger_tapping_distances.csv
│   ├── left/
│   │   └── …
│   └── docs/                # training labels & test split (not produced by inference)
│       ├── weak_supervision_final.csv
│       └── test-set-balanced.csv
├── hand_open_close/
│   ├── right/ … left/
│   └── docs/ …
├── hand_up_down/
│   ├── right/ … left/
│   └── docs/ …
└── results/
    └── inference.csv        # default batch inference output
```


### Pose CSV (`pose/<patient_id>_<side>_<task>.csv`)

One row per detected hand per frame (MediaPipe, 21 landmarks):

| Column | Description |
|--------|-------------|
| `frame_number` | Frame index in the source video |
| `hand_id`, `hand_label` | Hand index and corrected left/right label |
| `hand_width`, `hand_height` | bounding-box size |
| `x_0`…`x_20`, `y_0`…`y_20`, `z_0`…`z_20` | Landmark coordinates (0–1) |

### Distances CSV (`distances/<patient_id>_<side>_<task>_distances.csv`)

Time-series features derived from pose. **Finger tapping** files are written by `DistanceCalculator` with:

| Column | Description |
|--------|-------------|
| `Frame` | Frame number |
| `Finger Distance` | Thumb–index 3-D distance (pixels) |
| `Finger Normalized Distance` | Distance normalized by hand scale (**used for finger tapping inference**) |
| `Angular Distance` | Wrist angle (degrees) |
| `Wrist Coordinate` | Scaled wrist position |
| `Hand BBox Width`, `Hand BBox Height` | Hand bounding box (pixels) |

**Hand open/close** and **pronation–supination** use task-specific columns already present in `Booth_Processed`:

| Task | Inference column |
|------|------------------|
| Hand open/close | `Normalized Hand Sum Finger Distances` |
| Pronation–supination | `yaw_rad` |

Matching pose files omit the `_distances` suffix (e.g. `SUBJECT_DATE_right_finger_tapping.csv`).

### UPDRS severity scores (inference)

Each recording yields an **MDS-UPDRS Part III severity** class **0–3** (0 = normal, 1 = slight, 2 = mild, 3 = moderate/severe).

Optional per-symptom labels (`symptom_*` columns) are supported in the API when symptom models are loaded; symptom column names are listed in `configs/*.json`.

**Batch inference CSV** (default: `Booth_Processed/results/inference.csv`):

| Column | Description |
|--------|-------------|
| `patient_id` | Patient ID + side, e.g. `SUBJECT_DATE_001_right` |
| `task` | `finger_tapping`, `hand_open_close`, or `hand_up_down` |
| `subtask` | `right` or `left` |
| `severity` | Predicted UPDRS severity (0–3), or empty if the distances file was missing/invalid |
| `raw_sequence_length` | Number of frames/rows in the distances CSV |

Six rows per patient when all three tasks and both hands run successfully.

### Training artifacts & evaluation metrics

Training does **not** write pose or distances; it reads existing `distances/` CSVs and saves models under `models/`:

```
models/<task>/<version>/
├── classifier.joblib
├── rocket.joblib
└── metadata.json      # task config, accuracy/MAE/MSE on held-out test set
```

`evaluate` prints a classification report and confusion matrix to the terminal; metrics are also stored in `metadata.json`.

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

Three entry points: **pose** (landmark CSVs), **csv** (feature time series), or **video** (full pipeline).

| Mode | Input | Best for |
|------|--------|----------|
| `pose` | `…/pose/<id>_<side>_<task>.csv` | You already ran MediaPipe (or have Booth pose exports) |
| `csv` | `…/distances/<id>_<side>_<task>_distances.csv` | Feature CSVs ready for all tasks |
| `video` | `--video-path` and/or MP4s under `--raw-dir` | End-to-end from recordings |

Pose mode writes distances under `distances/` and then predicts severity. **Finger tapping** is fully supported from pose; hand open/close and pronation need distances CSVs with their own columns (use `csv` mode).

Use `--hands left`, `--hands right`, or `--hands both` (default) to run one or both sides per task.

### From pre-computed pose CSVs (recommended starting point)

Place pose files in the standard layout, then:

```bash
python scripts/run_inference.py \
    --mode pose \
    --patient-ids 06238_20240813_right \
    --processed-dir N:/Booth_Processed \
    --tasks finger_tapping \
    --hands right \
    --video-width 1920 \
    --video-height 1080 \
    --model-version latest \
    --output results/inference.csv
```

`--video-width` / `--video-height` must match the resolution used when the pose was extracted (MediaPipe coords are normalized 0–1 and scaled back to pixels).

Expected paths (right hand, finger tapping example):

```
Booth_Processed/finger_tapping/right/pose/SUBJECT_DATE_001_right_finger_tapping.csv
→ Booth_Processed/finger_tapping/right/distances/SUBJECT_DATE_001_right_finger_tapping_distances.csv  (written automatically)
```

### From pre-computed distances CSVs

```bash
python scripts/run_inference.py \
    --mode csv \
    --patient-ids SUBJECT_DATE_001 \
    --processed-dir N:/Booth_Processed \
    --model-version latest \
    --output results/inference.csv
```

Use this for **all three tasks** when distances already exist (required for hand open/close and pronation).

### From raw videos (full pipeline)

Explicit video file(s):

```bash
python scripts/run_inference.py \
    --mode video \
    --patient-ids SUBJECT_DATE_001 \
    --processed-dir N:/Booth_Processed \
    --video-path "N:/path/to/right_finger_tapping.mp4" \
    --hands right
```

Task and side are inferred from the filename (e.g. `right_finger_tapping.mp4`). Use one `--patient-ids` value per run. With `--hands left` or `--hands right`, only matching videos are processed.

Booth directory layout (all videos for listed patients):

```bash
python scripts/run_inference.py \
    --mode video \
    --patient-ids SUBJECT_DATE_001 \
    --raw-dir "N:/CAMERA Booth Data/Booth" \
    --processed-dir N:/Booth_Processed
```


---

## Tests

```bash
pip install pytest
pytest tests/test_pipeline_smoke.py -v
```

