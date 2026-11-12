"""
feature_matcher.py — ORB 기반 단일 대상 특징점 매칭기.

레퍼런스 이미지 1장을 로드하고, match() 호출마다 현재 프레임에
해당 대상이 등장하는지 확인한다.

제약:
- ORB 사용 (기본 opencv-python으로 가능; contrib 불필요)
- SIFT도 지원 (opencv-contrib-python 필요, algorithm='SIFT' 선택)
- 딥러닝 금지; dnn 모듈 금지
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
        레퍼런스 이미지 파일 경로.
    algorithm : str
        'ORB' (기본값) 또는 'SIFT'.
    min_matches : int
        "TARGET DETECTED" 판정에 필요한 최소 양호 매칭 수.
    ratio_thresh : float
        Lowe's ratio test 임계값 (SIFT 및 ORB 모두 적용).
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

        # ----- 검출기 & 매칭기 생성 -----
        if self.algorithm == "SIFT":
            self._detector = cv2.SIFT_create()
            # SIFT(실수 디스크립터) → FLANN 기반 매칭
            index_params = {"algorithm": 1, "trees": 5}  # FLANN_INDEX_KDTREE
            search_params = {"checks": 50}
            self._matcher = cv2.FlannBasedMatcher(index_params, search_params)
        else:
            # ORB (이진 디스크립터) → Brute-Force Hamming 거리
            self._detector = cv2.ORB_create(nfeatures=1000)
            self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        # ----- 레퍼런스 이미지 로드 및 특징점 추출 -----
        ref_img = cv2.imread(str(reference_path), cv2.IMREAD_GRAYSCALE)
        if ref_img is None:
            raise FileNotFoundError(
                f"레퍼런스 이미지를 찾을 수 없습니다: {reference_path}"
            )
        self._ref_kp, self._ref_des = self._detector.detectAndCompute(
            ref_img, None
        )
        if self._ref_des is None or len(self._ref_des) == 0:
            raise ValueError(
                "레퍼런스 이미지에서 특징점을 검출하지 못했습니다. "
                "텍스처가 풍부한 이미지를 사용하세요."
            )

        # 보고서용 통계
        self.last_ref_kp_count: int = len(self._ref_kp)
        self.last_matched: int = 0   # 원시 매칭 수
        self.last_good: int = 0      # ratio test 통과 수

    # ------------------------------------------------------------------

    @property
    def detected(self) -> bool:
        """마지막 match() 호출에서 대상이 감지되었으면 True."""
        return self._detected

    def match(self, frame_gray: np.ndarray) -> Tuple[bool, int, int]:
        """
        frame_gray에 대해 대상 매칭을 수행한다.

        Returns
        -------
        (detected, n_matched, n_good)
            detected   – 대상 감지 여부
            n_matched  – 원시 매칭 수
            n_good     – ratio test 통과 매칭 수
        """
        frame_kp, frame_des = self._detector.detectAndCompute(frame_gray, None)

        if frame_des is None or len(frame_des) < 2:
            self._detected = False
            self.last_matched = 0
            self.last_good = 0
            return False, 0, 0

        # ----- kNN 매칭 + Lowe ratio test -----
        raw = self._matcher.knnMatch(self._ref_des, frame_des, k=2)
        good = []
        for pair in raw:
            if len(pair) == 2:
                m, n = pair
                # 가장 가까운 매칭이 두 번째보다 충분히 가까워야 양호한 매칭
                if m.distance < self.ratio_thresh * n.distance:
                    good.append(m)

        n_matched = len(raw)
        n_good = len(good)
        self.last_matched = n_matched
        self.last_good = n_good
        self._detected = n_good >= self.min_matches
        return self._detected, n_matched, n_good
