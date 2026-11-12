"""
tracker.py — 거리 기반 다중 객체 추적기 (궤적 기록 포함).

설계 제약 (과제 기준):
- 딥러닝 모델(YOLO, SSD 등) 사용 금지
- OpenCV dnn 모듈 사용 금지
- OpenCV + NumPy만 사용

ID 생명주기
  NEW  → 최초 감지 시 새 ID 부여
  LIVE → 매 프레임 매칭 성공, self.objects에 저장
  DEAD → max_disappeared 프레임 연속 미감지 시 삭제
"""

from __future__ import annotations

import numpy as np
from collections import OrderedDict
from typing import Dict, List, Tuple


class Tracker:
    """
    그리디 최근접 이웃(Greedy Nearest-Neighbour) 추적기.

    Parameters
    ----------
    max_disappeared : int
        매칭 실패 허용 최대 프레임 수. 초과 시 ID 삭제.
    max_distance : int
        두 감지 결과가 같은 객체로 매칭될 수 있는 최대 픽셀 거리.
    max_trail_len : int
        객체당 저장할 최대 궤적 점 수.
    """

    def __init__(
        self,
        max_disappeared: int = 30,
        max_distance: int = 100,
        max_trail_len: int = 50,
    ) -> None:
        self.next_id: int = 0
        # {id: 중심점 (x, y)}
        self.objects: OrderedDict[int, Tuple[int, int]] = OrderedDict()
        # {id: 마지막으로 감지된 이후 경과 프레임 수}
        self.disappeared: OrderedDict[int, int] = OrderedDict()
        # {id: (x, y) 중심점 목록 — 궤적}
        self.trails: Dict[int, List[Tuple[int, int]]] = {}

        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.max_trail_len = max_trail_len

        # 성능 통계 카운터 (보고서용, main.py에서 읽음)
        self.id_switches: int = 0       # ID 교환 발생 횟수
        self.tracking_failures: int = 0 # 추적 실패(ID 삭제) 횟수
        self._prev_matches: Dict[int, int] = {}  # 감지 인덱스 → 객체 ID

    # ------------------------------------------------------------------
    # 내부 헬퍼
    # ------------------------------------------------------------------

    def _register(self, centroid: Tuple[int, int]) -> int:
        """새 객체를 등록하고 새 ID를 반환한다."""
        oid = self.next_id
        self.objects[oid] = centroid
        self.disappeared[oid] = 0
        self.trails[oid] = [centroid]
        self.next_id += 1
        return oid

    def _deregister(self, oid: int) -> None:
        """오랫동안 사라진 객체를 추적 목록에서 제거한다."""
        del self.objects[oid]
        del self.disappeared[oid]
        self.trails.pop(oid, None)

    def _append_trail(self, oid: int, centroid: Tuple[int, int]) -> None:
        """궤적 점을 추가하고 최대 길이를 초과하면 오래된 점을 삭제한다."""
        trail = self.trails.setdefault(oid, [])
        trail.append(centroid)
        if len(trail) > self.max_trail_len:
            trail.pop(0)

    # ------------------------------------------------------------------
    # 공개 API
    # ------------------------------------------------------------------

    def update(
        self, detections: List[Tuple[int, int, int, int]]
    ) -> Dict[int, Tuple[int, int]]:
        """
        현재 프레임의 감지 결과로 추적기 상태를 갱신한다.

        Parameters
        ----------
        detections : (x, y, w, h) 바운딩 박스 목록

        Returns
        -------
        dict  {객체_ID: (cx, cy)}  현재 살아있는 모든 객체
        """
        # ---- 이번 프레임에 감지 없음 --------------------------------
        if len(detections) == 0:
            for oid in list(self.disappeared.keys()):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.tracking_failures += 1
                    self._deregister(oid)
            return dict(self.objects)

        # ---- 입력 감지 결과의 중심점 계산 ---------------------------
        input_centroids: List[Tuple[int, int]] = []
        for x, y, w, h in detections:
            cx = x + w // 2
            cy = y + h // 2
            input_centroids.append((cx, cy))

        # ---- 첫 프레임: 모두 신규 등록 ----------------------------
        if len(self.objects) == 0:
            for c in input_centroids:
                self._register(c)
            return dict(self.objects)

        # ---- 그리디 최근접 이웃 매칭 --------------------------------
        obj_ids = list(self.objects.keys())
        obj_centroids = list(self.objects.values())

        # 거리 행렬 [기존 객체 수 × 신규 감지 수]
        D = np.zeros((len(obj_centroids), len(input_centroids)), dtype=float)
        for i, oc in enumerate(obj_centroids):
            for j, ic in enumerate(input_centroids):
                D[i, j] = np.sqrt((oc[0] - ic[0]) ** 2 + (oc[1] - ic[1]) ** 2)

        # 최소 거리 기준 정렬 — 그리디 할당
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows: set[int] = set()
        used_cols: set[int] = set()
        new_matches: Dict[int, int] = {}  # 감지 인덱스 → 객체 ID

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

        # ---- ID 교환 감지 (휴리스틱) --------------------------------
        # 이전 프레임 대비 같은 감지 인덱스에 다른 ID가 매칭되면 교환으로 집계
        for det_idx, oid in new_matches.items():
            prev_oid = self._prev_matches.get(det_idx)
            if prev_oid is not None and prev_oid != oid:
                self.id_switches += 1
        self._prev_matches = new_matches

        # ---- 매칭 실패한 기존 객체 처리 -----------------------------
        unmatched_rows = set(range(len(obj_ids))) - used_rows
        for row in unmatched_rows:
            oid = obj_ids[row]
            self.disappeared[oid] += 1
            if self.disappeared[oid] > self.max_disappeared:
                self.tracking_failures += 1
                self._deregister(oid)

        # ---- 새로 등장한 감지 결과 등록 -----------------------------
        unmatched_cols = set(range(len(input_centroids))) - used_cols
        for col in unmatched_cols:
            self._register(input_centroids[col])

        return dict(self.objects)

    # ------------------------------------------------------------------
    # 편의 메서드
    # ------------------------------------------------------------------

    def get_trail(self, oid: int) -> List[Tuple[int, int]]:
        """객체 oid의 궤적 점 목록을 반환한다 (없으면 빈 리스트)."""
        return self.trails.get(oid, [])

    def stats(self) -> Dict[str, int]:
        """보고서용 누적 통계를 반환한다."""
        return {
            "id_switches": self.id_switches,
            "tracking_failures": self.tracking_failures,
            "active_objects": len(self.objects),
        }
