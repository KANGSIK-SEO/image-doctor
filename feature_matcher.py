"""
feature_matcher.py — ORB-based single-target feature matcher.

Loads one reference image; each call to match() checks whether the
target object appears in the current frame.

Constraints:
- Uses ORB (available in plain opencv-python; no contrib needed for ORB)
- NO deep-learning; NO dnn module
- SIFT is also supported (requires opencv-contrib-python) and can be
  selected via algorithm='SIFT'.
"""

from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple


class FeatureMatcher:
    """
    Parameters
    ----------
    reference_path : str | Path
        Path to the reference image file.
    algorithm : str
        'ORB' (default) or 'SIFT'.
    min_matches : int
        Minimum good matches required to declare TARGET DETECTED.
    ratio_thresh : float
        Lowe's ratio-test threshold (SIFT only).
    """

    def __init__(
        self,
        reference_path: str | Path,
        algorithm: str = "ORB",
        min_matches: int = 15,
        ratio_thresh: float = 0.75,
    ) -> None:
        self.algorithm = algorithm.upper()
        self.min_matches = min_matches
        self.ratio_thresh = ratio_thresh
        self._detected = False

        # ----- build detector & matcher -----
        if self.algorithm == "SIFT":
            self._detector = cv2.SIFT_create()
            # FLANN parameters for SIFT (float descriptors)
            index_params = {"algorithm": 1, "trees": 5}  # FLANN_INDEX_KDTREE
            search_params = {"checks": 50}
            self._matcher = cv2.FlannBasedMatcher(index_params, search_params)
        else:
            self._detector = cv2.ORB_create(nfeatures=1000)
            # Brute-force Hamming for ORB (binary descriptors)
            self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        # ----- load reference image -----
        ref_img = cv2.imread(str(reference_path), cv2.IMREAD_GRAYSCALE)
        if ref_img is None:
            raise FileNotFoundError(
                f"Reference image not found: {reference_path}"
            )
        self._ref_kp, self._ref_des = self._detector.detectAndCompute(
            ref_img, None
        )
        if self._ref_des is None or len(self._ref_des) == 0:
            raise ValueError(
                "No features detected in reference image. "
                "Try a more textured image."
            )

        # Statistics (for the report)
        self.last_ref_kp_count: int = len(self._ref_kp)
        self.last_matched: int = 0
        self.last_good: int = 0

    # ------------------------------------------------------------------

    @property
    def detected(self) -> bool:
        """True if the last match() call found the target."""
        return self._detected

    def match(self, frame_gray: np.ndarray) -> Tuple[bool, int, int]:
        """
        Match target against *frame_gray*.

        Returns
        -------
        (detected, n_matched, n_good)
            detected   – whether target is considered found
            n_matched  – raw descriptor matches
            n_good     – matches that pass the quality filter
        """
        frame_kp, frame_des = self._detector.detectAndCompute(frame_gray, None)

        if frame_des is None or len(frame_des) < 2:
            self._detected = False
            self.last_matched = 0
            self.last_good = 0
            return False, 0, 0

        # ----- match -----
        if self.algorithm == "SIFT":
            # knnMatch → Lowe ratio test
            raw = self._matcher.knnMatch(self._ref_des, frame_des, k=2)
            good = []
            for pair in raw:
                if len(pair) == 2:
                    m, n = pair
                    if m.distance < self.ratio_thresh * n.distance:
                        good.append(m)
        else:
            # BFMatcher for ORB: kNN + ratio test on Hamming distances
            raw = self._matcher.knnMatch(self._ref_des, frame_des, k=2)
            good = []
            for pair in raw:
                if len(pair) == 2:
                    m, n = pair
                    if m.distance < self.ratio_thresh * n.distance:
                        good.append(m)

        n_matched = len(raw)
        n_good = len(good)
        self.last_matched = n_matched
        self.last_good = n_good
        self._detected = n_good >= self.min_matches
        return self._detected, n_matched, n_good
