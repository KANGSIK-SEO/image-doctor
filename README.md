# image-doctor 🩺

**Real-time object detection, tracking, and feature matching — classical computer vision only.**

> Assignment: AI/SW 심화 — 컴퓨터 비전  
> Constraint: No YOLO / SSD / R-CNN / dnn / MediaPipe

---

## Project Structure

```
image-doctor/
├── main.py              # Entry point: video I/O, BG subtraction, draw loop
├── tracker.py           # Tracker class — greedy nearest-neighbour ID manager
├── feature_matcher.py   # FeatureMatcher class — ORB/SIFT single-target match
├── requirements.txt     # Pinned dependencies
├── report.md            # Limit analysis report (Korean)
└── README.md            # This file
```

### File responsibilities

| File | Responsibility |
|---|---|
| `main.py` | CLI args, `cv2.VideoCapture`, MOG2 background subtractor, morphology, detection, draw overlay, keyboard control, stats CSV |
| `tracker.py` | `Tracker` class only — ID lifecycle, greedy matching, trail storage, ID-switch / failure counters |
| `feature_matcher.py` | `FeatureMatcher` class only — ORB or SIFT descriptor extraction, Lowe ratio test, `detected` flag |
| `report.md` | Quantitative performance tables, failure analysis, classical-vs-DL conclusion |

---

## Quick Start

```bash
# 1. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3a. Run with webcam (default)
python main.py

# 3b. Run with a video file
python main.py --source path/to/video.mp4

# 3c. Run with feature matching target
python main.py --source path/to/video.mp4 --target path/to/target.jpg

# 3d. Use SIFT instead of ORB
python main.py --algo SIFT --target path/to/target.jpg
```

### Keyboard Controls

| Key | Action |
|---|---|
| `q` | Quit |
| `s` | Save snapshot (`snapshot_NNNN.png`) |
| `p` | Pause / unpause |

---

## CLI Options

| Option | Default | Description |
|---|---|---|
| `--source` | `0` | Webcam index or video file path |
| `--target` | `None` | Reference image for feature matching |
| `--lr` | `0.005` | MOG2 learning rate (0–1) |
| `--algo` | `ORB` | Feature matching: `ORB` or `SIFT` |
| `--save` | `stats.csv` | Output path for session statistics CSV |

---

## How It Works

### 1. Background Subtraction (MOG2)

```
Frame → MOG2.apply(lr) → fg_mask (0/127/255)
         ↓
  threshold(>200) → binary mask
         ↓
  MORPH_OPEN  → remove noise pixels
         ↓
  MORPH_CLOSE → fill holes inside objects
```

`learning_rate` controls how fast the background model adapts:
- **0.001** — slow; misses lighting changes but stable
- **0.005** — balanced default
- **0.05** — fast; absorbs slow-moving objects into background

### 2. Detection

`findContours` on the cleaned mask → filter by area (`MIN_CONTOUR_AREA = 800 px²`) → list of `(x, y, w, h)` bounding boxes.

### 3. Tracking (`tracker.py`)

```
detections → compute centroids
           → build distance matrix [N_objects × N_detections]
           → greedy nearest-neighbour assignment (threshold: 100 px)
           → update ID positions & trails
           → retire IDs missing > max_disappeared frames
           → register new detections as new IDs
```

**Known limitation**: greedy matching fails when two objects cross paths — the tracker has no appearance information to disambiguate.

### 4. Feature Matching (`feature_matcher.py`)

```
Reference image → ORB/SIFT keypoints + descriptors (once at init)
Each frame      → ORB/SIFT keypoints + descriptors
               → knnMatch(k=2) → Lowe ratio test (θ=0.75)
               → n_good >= min_matches → TARGET DETECTED
```

---

## Constraints Respected

- ✅ No YOLO / SSD / R-CNN
- ✅ No `cv2.dnn` module
- ✅ No MediaPipe
- ✅ `Tracker` class in separate `tracker.py`
- ✅ Quantitative performance measured and reported in `report.md`
- ✅ ID Switch and per-condition failure rates included

---

## Requirements

- Python 3.8+  (recommended: 3.10+)
- `opencv-contrib-python >= 4.5` (includes SIFT)
- `numpy`

SIFT is patent-free since 2020 and ships in `opencv-contrib-python`.  
ORB requires only plain `opencv-python`.

---

## Bonus Features (implemented)

| Feature | Location | Notes |
|---|---|---|
| IoU-based matching | `tracker.py` → switchable via `use_iou` param | Compare vs greedy centroid distance |
| Learning-rate sweep | `main.py --lr` | Try 0.001 / 0.005 / 0.05 to observe tradeoff |
| Session stats CSV | `main.py --save` | Auto-saved on quit |

---

## Report

See [`report.md`](report.md) for:
1. Tracking performance tables (ID Switch, failure rates by condition)
2. Feature matching performance by rotation / occlusion
3. Background modelling learning-rate tradeoff
4. Why distance-only tracking cannot survive crossing paths
5. Why hand-crafted features fail under unseen transforms
6. Why deep learning is necessary (answered from hands-on experience)
