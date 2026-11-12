"""
main.py — Real-time video analysis application.

Usage
-----
  python main.py [--source SOURCE] [--target TARGET] [--lr LR]

Arguments
---------
  --source  : int (webcam index) or path to video file. Default: 0 (webcam).
  --target  : path to reference image for feature matching. Default: none.
  --lr      : background subtractor learning rate (0–1). Default: 0.005.
  --algo    : feature matching algorithm, 'ORB' or 'SIFT'. Default: 'ORB'.
  --save    : path to save session stats CSV on exit. Default: stats.csv.

Keyboard controls
-----------------
  q   quit
  s   save snapshot of current frame
  p   pause / unpause

Constraints respected
---------------------
- No YOLO / SSD / R-CNN / dnn module / MediaPipe
- OpenCV + NumPy only
- Tracker logic fully in tracker.py
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from tracker import Tracker
from feature_matcher import FeatureMatcher


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Object detection & tracking demo")
    p.add_argument("--source", default="0",
                   help="Webcam index (int) or video file path")
    p.add_argument("--target", default=None,
                   help="Reference image path for feature matching")
    p.add_argument("--lr", type=float, default=0.005,
                   help="Background subtractor learning rate (0–1)")
    p.add_argument("--algo", default="ORB", choices=["ORB", "SIFT"],
                   help="Feature matching algorithm")
    p.add_argument("--save", default="stats.csv",
                   help="Path to save session statistics CSV")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Background subtraction
# ---------------------------------------------------------------------------

def build_background_subtractor(lr: float) -> cv2.BackgroundSubtractorMOG2:
    """
    MOG2 is the standard adaptive GMM background model in OpenCV.
    learning_rate fed here is the *default* rate; it can be overridden per
    apply() call.  Three learning-rate experiments are logged in the report.
    """
    bs = cv2.createBackgroundSubtractorMOG2(
        history=500,
        varThreshold=25,
        detectShadows=True,   # shadows labelled 127, foreground 255
    )
    return bs


def process_fg_mask(fg_mask: np.ndarray) -> np.ndarray:
    """
    Post-process raw foreground mask:
      1. Threshold: keep only definite foreground (255), discard shadows (127).
      2. Morphological open  → remove isolated noise pixels.
      3. Morphological close → fill holes inside objects.
    """
    _, thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)
    return closed


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

MIN_CONTOUR_AREA = 800   # px² — ignore tiny blobs (leaves, sensor noise)


def detect_objects(
    clean_mask: np.ndarray,
) -> list[tuple[int, int, int, int]]:
    """
    Extract bounding boxes from the cleaned foreground mask.

    Returns list of (x, y, w, h).
    """
    contours, _ = cv2.findContours(
        clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_CONTOUR_AREA:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        boxes.append((x, y, w, h))
    return boxes


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

COLORS = [
    (0, 255, 0),    # green
    (0, 165, 255),  # orange
    (255, 0, 255),  # magenta
    (255, 255, 0),  # cyan
    (0, 0, 255),    # red
    (255, 128, 0),  # blue-ish
]


def color_for(oid: int) -> tuple[int, int, int]:
    return COLORS[oid % len(COLORS)]


def draw_overlay(
    frame: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    objects: dict[int, tuple[int, int]],
    tracker: "Tracker",
    fps: float,
    target_detected: bool,
    paused: bool,
) -> None:
    """Draw all visual overlays onto *frame* in-place."""

    # --- trails ---
    for oid, centroid in objects.items():
        trail = tracker.get_trail(oid)
        c = color_for(oid)
        for i in range(1, len(trail)):
            cv2.line(frame, trail[i - 1], trail[i], c, 2)

    # --- bounding boxes & IDs ---
    # Map centroid → box for annotation (best-effort: nearest centroid wins)
    centroid_to_box: dict[tuple[int, int], tuple[int, int, int, int]] = {}
    for bx, by, bw, bh in boxes:
        cx, cy = bx + bw // 2, by + bh // 2
        centroid_to_box[(cx, cy)] = (bx, by, bw, bh)

    for oid, (cx, cy) in objects.items():
        c = color_for(oid)
        # Find the closest box to this centroid
        best_box = None
        best_d = float("inf")
        for (bcx, bcy), box in centroid_to_box.items():
            d = (cx - bcx) ** 2 + (cy - bcy) ** 2
            if d < best_d:
                best_d = d
                best_box = box
        if best_box and best_d < 50 ** 2:
            bx, by, bw, bh = best_box
            cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), c, 2)
        cv2.putText(
            frame, f"ID:{oid}",
            (cx - 20, cy - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2,
        )
        cv2.circle(frame, (cx, cy), 4, c, -1)

    # --- FPS ---
    cv2.putText(
        frame, f"FPS: {fps:.1f}",
        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2,
    )

    # --- stats ---
    stats = tracker.stats()
    cv2.putText(
        frame,
        f"ID Switches: {stats['id_switches']}  Failures: {stats['tracking_failures']}",
        (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 255), 2,
    )

    # --- target detected banner ---
    if target_detected:
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (w // 2 - 180, h - 60), (w // 2 + 180, h - 10),
                      (0, 0, 180), -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
        cv2.putText(
            frame, "TARGET DETECTED",
            (w // 2 - 155, h - 22),
            cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 255, 255), 2,
        )

    # --- paused indicator ---
    if paused:
        cv2.putText(
            frame, "PAUSED",
            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2,
        )

    # --- controls reminder ---
    cv2.putText(
        frame, "q: quit  s: snapshot  p: pause",
        (10, frame.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1,
    )


# ---------------------------------------------------------------------------
# Stats persistence
# ---------------------------------------------------------------------------

def save_stats(path: str, tracker: "Tracker") -> None:
    s = tracker.stats()
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k, v in s.items():
            w.writerow([k, v])
    print(f"[info] Stats saved to {path}")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # --- open video source ---
    try:
        source = int(args.source)
    except ValueError:
        source = args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[error] Cannot open source: {args.source}", file=sys.stderr)
        sys.exit(1)

    # --- build subsystems ---
    bs = build_background_subtractor(args.lr)
    tracker = Tracker(max_disappeared=30, max_distance=100, max_trail_len=60)

    matcher: FeatureMatcher | None = None
    if args.target:
        try:
            matcher = FeatureMatcher(args.target, algorithm=args.algo)
            print(f"[info] Feature matcher loaded: {args.target} ({args.algo})")
        except (FileNotFoundError, ValueError) as e:
            print(f"[warn] Feature matcher disabled: {e}", file=sys.stderr)

    paused = False
    snapshot_idx = 0
    fps = 0.0
    t_prev = time.perf_counter()
    target_detected = False
    frame_count = 0

    print("[info] Starting. Press q=quit, s=snapshot, p=pause")

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("[info] End of stream.")
                break
            frame_count += 1

            # --- FPS ---
            t_now = time.perf_counter()
            fps = 1.0 / max(t_now - t_prev, 1e-6)
            t_prev = t_now

            # --- background subtraction ---
            fg_mask = bs.apply(frame, learningRate=args.lr)
            clean_mask = process_fg_mask(fg_mask)

            # --- detection ---
            boxes = detect_objects(clean_mask)

            # --- tracking ---
            objects = tracker.update(boxes)

            # --- feature matching (every 3rd frame for performance) ---
            if matcher is not None and frame_count % 3 == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                target_detected, _, _ = matcher.match(gray)

            # --- draw ---
            draw_overlay(
                frame, boxes, objects, tracker, fps,
                target_detected, paused,
            )

        # --- show ---
        cv2.imshow("image-doctor | object tracker", frame)

        # --- keyboard ---
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            path = f"snapshot_{snapshot_idx:04d}.png"
            cv2.imwrite(path, frame)
            print(f"[info] Snapshot saved: {path}")
            snapshot_idx += 1
        elif key == ord("p"):
            paused = not paused
            print("[info] Paused" if paused else "[info] Resumed")

    cap.release()
    cv2.destroyAllWindows()
    save_stats(args.save, tracker)
    print(f"[info] Final stats: {tracker.stats()}")


if __name__ == "__main__":
    main()
