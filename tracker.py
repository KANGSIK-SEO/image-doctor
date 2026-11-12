"""
tracker.py — Distance-based multi-object tracker with trajectory history.

Design constraints (per assignment):
- NO deep-learning models (YOLO, SSD, etc.)
- NO OpenCV dnn module
- Pure OpenCV + NumPy only

ID lifecycle
  NEW  → assigned on first detection
  LIVE → matched each frame, stored in self.objects
  DEAD → removed after max_disappeared frames with no match
"""

from __future__ import annotations

import numpy as np
from collections import OrderedDict
from typing import Dict, List, Tuple


class Tracker:
    """
    Greedy nearest-neighbour tracker.

    Parameters
    ----------
    max_disappeared : int
        Frames without a match before an ID is retired.
    max_distance : int
        Pixel distance beyond which two detections cannot match.
    max_trail_len : int
        Maximum number of trail points kept per object.
    """

    def __init__(
        self,
        max_disappeared: int = 30,
        max_distance: int = 100,
        max_trail_len: int = 50,
    ) -> None:
        self.next_id: int = 0
        # {id: centroid (x, y)}
        self.objects: OrderedDict[int, Tuple[int, int]] = OrderedDict()
        # {id: frames since last seen}
        self.disappeared: OrderedDict[int, int] = OrderedDict()
        # {id: list of (x, y) centroids — the trail}
        self.trails: Dict[int, List[Tuple[int, int]]] = {}

        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.max_trail_len = max_trail_len

        # Statistics counters (read by main.py for the report)
        self.id_switches: int = 0
        self.tracking_failures: int = 0
        self._prev_matches: Dict[int, int] = {}  # det_index → obj_id

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _register(self, centroid: Tuple[int, int]) -> int:
        """Register a new object; return its fresh ID."""
        oid = self.next_id
        self.objects[oid] = centroid
        self.disappeared[oid] = 0
        self.trails[oid] = [centroid]
        self.next_id += 1
        return oid

    def _deregister(self, oid: int) -> None:
        """Remove an object that has been missing too long."""
        del self.objects[oid]
        del self.disappeared[oid]
        # Keep trail in memory so callers can still read it; clear if desired.
        self.trails.pop(oid, None)

    def _append_trail(self, oid: int, centroid: Tuple[int, int]) -> None:
        trail = self.trails.setdefault(oid, [])
        trail.append(centroid)
        if len(trail) > self.max_trail_len:
            trail.pop(0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self, detections: List[Tuple[int, int, int, int]]
    ) -> Dict[int, Tuple[int, int]]:
        """
        Update tracker state with this frame's detections.

        Parameters
        ----------
        detections : list of (x, y, w, h) bounding boxes

        Returns
        -------
        dict  {object_id: (cx, cy)}  for all currently live objects
        """
        # ---- no detections this frame --------------------------------
        if len(detections) == 0:
            for oid in list(self.disappeared.keys()):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.tracking_failures += 1
                    self._deregister(oid)
            return dict(self.objects)

        # ---- compute centroids of incoming detections ----------------
        input_centroids: List[Tuple[int, int]] = []
        for x, y, w, h in detections:
            cx = x + w // 2
            cy = y + h // 2
            input_centroids.append((cx, cy))

        # ---- first frame: register all -----------------------------
        if len(self.objects) == 0:
            for c in input_centroids:
                self._register(c)
            return dict(self.objects)

        # ---- greedy nearest-neighbour matching ----------------------
        obj_ids = list(self.objects.keys())
        obj_centroids = list(self.objects.values())

        # Build distance matrix  [num_objects × num_detections]
        D = np.zeros((len(obj_centroids), len(input_centroids)), dtype=float)
        for i, oc in enumerate(obj_centroids):
            for j, ic in enumerate(input_centroids):
                D[i, j] = np.sqrt((oc[0] - ic[0]) ** 2 + (oc[1] - ic[1]) ** 2)

        # Sort by minimum distance — greedy assignment
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows: set[int] = set()
        used_cols: set[int] = set()
        new_matches: Dict[int, int] = {}  # det_index → obj_id

        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
            if D[row, col] > self.max_distance:
                continue
            oid = obj_ids[row]
            new_centroid = input_centroids[col]
            self.objects[oid] = new_centroid
            self.disappeared[oid] = 0
            self._append_trail(oid, new_centroid)
            used_rows.add(row)
            used_cols.add(col)
            new_matches[col] = oid

        # ---- detect ID switches (heuristic) -------------------------
        # An ID switch is counted when two previously separate objects now
        # swap their detection assignment compared to the last frame.
        for det_idx, oid in new_matches.items():
            prev_oid = self._prev_matches.get(det_idx)
            if prev_oid is not None and prev_oid != oid:
                self.id_switches += 1
        self._prev_matches = new_matches

        # ---- handle unmatched existing objects ----------------------
        unmatched_rows = set(range(len(obj_ids))) - used_rows
        for row in unmatched_rows:
            oid = obj_ids[row]
            self.disappeared[oid] += 1
            if self.disappeared[oid] > self.max_disappeared:
                self.tracking_failures += 1
                self._deregister(oid)

        # ---- register brand-new detections --------------------------
        unmatched_cols = set(range(len(input_centroids))) - used_cols
        for col in unmatched_cols:
            self._register(input_centroids[col])

        return dict(self.objects)

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def get_trail(self, oid: int) -> List[Tuple[int, int]]:
        """Return the stored trail for object *oid* (may be empty list)."""
        return self.trails.get(oid, [])

    def stats(self) -> Dict[str, int]:
        """Return cumulative statistics for the report."""
        return {
            "id_switches": self.id_switches,
            "tracking_failures": self.tracking_failures,
            "active_objects": len(self.objects),
        }
