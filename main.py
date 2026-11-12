"""
main.py — 실시간 영상 분석 애플리케이션 (메인 진입점).

사용법
-----
  python main.py [--source SOURCE] [--target TARGET] [--lr LR]

인수
---------
  --source  : 웹캠 인덱스(int) 또는 비디오 파일 경로. 기본값: 0 (웹캠).
  --target  : 특징점 매칭용 레퍼런스 이미지 경로. 기본값: 없음.
  --lr      : 배경 차분 학습률 (0–1). 기본값: 0.005.
  --algo    : 특징점 알고리즘, 'ORB' 또는 'SIFT'. 기본값: 'ORB'.
  --save    : 종료 시 세션 통계를 저장할 CSV 경로. 기본값: stats.csv.

키보드 조작
-----------------
  q   종료
  s   현재 프레임 스냅샷 저장
  p   일시정지 / 재개

제약사항
---------------------
- YOLO / SSD / R-CNN / dnn 모듈 / MediaPipe 사용 금지
- OpenCV + NumPy만 사용
- 추적 로직은 tracker.py의 Tracker 클래스로 분리
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
# CLI 인수 파싱
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="객체 감지 및 추적 데모")
    p.add_argument("--source", default="0",
                   help="웹캠 인덱스(int) 또는 비디오 파일 경로")
    p.add_argument("--target", default=None,
                   help="특징점 매칭용 레퍼런스 이미지 경로")
    p.add_argument("--lr", type=float, default=0.005,
                   help="배경 차분 학습률 (0–1)")
    p.add_argument("--algo", default="ORB", choices=["ORB", "SIFT"],
                   help="특징점 매칭 알고리즘")
    p.add_argument("--save", default="stats.csv",
                   help="세션 통계 저장 CSV 경로")
    return p.parse_args()


# ---------------------------------------------------------------------------
# 배경 차분 (Background Subtraction)
# ---------------------------------------------------------------------------

def build_background_subtractor(lr: float) -> cv2.BackgroundSubtractorMOG2:
    """
    MOG2(Mixture of Gaussians v2) 배경 모델을 생성한다.

    MOG2는 각 픽셀을 여러 개의 가우시안 분포 혼합으로 모델링하여
    조명 변화에 적응한다. learning_rate가 클수록 빠르게 변화에 적응하지만
    정지 객체를 배경으로 흡수하는 부작용이 있다.
    """
    bs = cv2.createBackgroundSubtractorMOG2(
        history=500,       # 배경 모델 학습에 사용할 이전 프레임 수
        varThreshold=25,   # 전경/배경 판단 임계값 (높을수록 엄격)
        detectShadows=True,  # 그림자 감지: 127(그림자), 255(전경)
    )
    return bs


def process_fg_mask(fg_mask: np.ndarray) -> np.ndarray:
    """
    원시 전경 마스크를 후처리한다.

    처리 순서:
      1. 임계값 처리 — 완전한 전경(255)만 남기고 그림자(127) 제거
      2. 모폴로지 열림(Opening) — 고립된 노이즈 픽셀 제거
      3. 모폴로지 닫힘(Closing) — 객체 내부 구멍 채움
    """
    # 그림자(127) 제거 — 255(전경)만 통과
    _, thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

    # 타원형 커널: 사람 형태의 불규칙한 경계에 적합
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # 열림: 작은 노이즈 제거 (침식 후 팽창)
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # 닫힘: 내부 구멍 채움 (팽창 후 침식)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)

    return closed


# ---------------------------------------------------------------------------
# 객체 감지
# ---------------------------------------------------------------------------

MIN_CONTOUR_AREA = 800   # px² — 이보다 작은 움직임은 노이즈로 무시


def detect_objects(
    clean_mask: np.ndarray,
) -> list[tuple[int, int, int, int]]:
    """
    정제된 전경 마스크에서 바운딩 박스를 추출한다.

    윤곽선(contour)을 찾고, 면적 필터링으로 너무 작은 블롭을 제거한다.

    Returns
    -------
    (x, y, w, h) 형식의 바운딩 박스 목록
    """
    contours, _ = cv2.findContours(
        clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_CONTOUR_AREA:
            continue  # 너무 작은 블롭 무시
        x, y, w, h = cv2.boundingRect(cnt)
        boxes.append((x, y, w, h))
    return boxes


# ---------------------------------------------------------------------------
# 화면 오버레이 그리기
# ---------------------------------------------------------------------------

# ID별 색상 팔레트 (BGR 형식)
COLORS = [
    (0, 255, 0),    # 초록
    (0, 165, 255),  # 주황
    (255, 0, 255),  # 자홍
    (255, 255, 0),  # 청록
    (0, 0, 255),    # 빨강
    (255, 128, 0),  # 파랑 계열
]


def color_for(oid: int) -> tuple[int, int, int]:
    """ID 번호로 색상을 선택한다 (팔레트 순환)."""
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
    """frame에 모든 시각화 요소를 그린다 (in-place)."""

    # --- 궤적 선 그리기 ---
    for oid, centroid in objects.items():
        trail = tracker.get_trail(oid)
        c = color_for(oid)
        for i in range(1, len(trail)):
            cv2.line(frame, trail[i - 1], trail[i], c, 2)

    # --- 바운딩 박스 & ID 표시 ---
    # 중심점 → 가장 가까운 박스 매핑 (최선 근사)
    centroid_to_box: dict[tuple[int, int], tuple[int, int, int, int]] = {}
    for bx, by, bw, bh in boxes:
        cx, cy = bx + bw // 2, by + bh // 2
        centroid_to_box[(cx, cy)] = (bx, by, bw, bh)

    for oid, (cx, cy) in objects.items():
        c = color_for(oid)
        # 이 객체 중심점과 가장 가까운 박스 탐색
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
        # ID 텍스트
        cv2.putText(
            frame, f"ID:{oid}",
            (cx - 20, cy - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2,
        )
        # 중심점 점
        cv2.circle(frame, (cx, cy), 4, c, -1)

    # --- FPS 표시 ---
    cv2.putText(
        frame, f"FPS: {fps:.1f}",
        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2,
    )

    # --- ID 교환 / 추적 실패 카운터 ---
    stats = tracker.stats()
    cv2.putText(
        frame,
        f"ID Switch: {stats['id_switches']}  추적 실패: {stats['tracking_failures']}",
        (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 255), 2,
    )

    # --- TARGET DETECTED 배너 ---
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

    # --- 일시정지 표시 ---
    if paused:
        cv2.putText(
            frame, "일시정지",
            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2,
        )

    # --- 조작 안내 ---
    cv2.putText(
        frame, "q: 종료  s: 스냅샷  p: 일시정지",
        (10, frame.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1,
    )


# ---------------------------------------------------------------------------
# 통계 저장
# ---------------------------------------------------------------------------

def save_stats(path: str, tracker: "Tracker") -> None:
    """추적 통계를 CSV 파일로 저장한다."""
    s = tracker.stats()
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["항목", "값"])
        for k, v in s.items():
            w.writerow([k, v])
    print(f"[정보] 통계 저장 완료: {path}")


# ---------------------------------------------------------------------------
# 메인 루프
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # --- 영상 소스 열기 ---
    try:
        source = int(args.source)   # 정수면 웹캠 인덱스
    except ValueError:
        source = args.source        # 문자열이면 파일 경로
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[오류] 소스를 열 수 없습니다: {args.source}", file=sys.stderr)
        sys.exit(1)

    # --- 서브시스템 초기화 ---
    bs = build_background_subtractor(args.lr)
    tracker = Tracker(max_disappeared=30, max_distance=100, max_trail_len=60)

    matcher: FeatureMatcher | None = None
    if args.target:
        try:
            matcher = FeatureMatcher(args.target, algorithm=args.algo)
            print(f"[정보] 특징점 매칭기 로드 완료: {args.target} ({args.algo})")
        except (FileNotFoundError, ValueError) as e:
            print(f"[경고] 특징점 매칭기 비활성화: {e}", file=sys.stderr)

    paused = False
    snapshot_idx = 0
    fps = 0.0
    t_prev = time.perf_counter()
    target_detected = False
    frame_count = 0

    print("[정보] 시작. q=종료, s=스냅샷, p=일시정지")

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("[정보] 영상 스트림 종료.")
                break
            frame_count += 1

            # --- FPS 계산 ---
            t_now = time.perf_counter()
            fps = 1.0 / max(t_now - t_prev, 1e-6)
            t_prev = t_now

            # --- 배경 차분 ---
            fg_mask = bs.apply(frame, learningRate=args.lr)
            clean_mask = process_fg_mask(fg_mask)

            # --- 객체 감지 ---
            boxes = detect_objects(clean_mask)

            # --- 객체 추적 ---
            objects = tracker.update(boxes)

            # --- 특징점 매칭 (3프레임마다 1회 — 성능 최적화) ---
            if matcher is not None and frame_count % 3 == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                target_detected, _, _ = matcher.match(gray)

            # --- 화면 그리기 ---
            draw_overlay(
                frame, boxes, objects, tracker, fps,
                target_detected, paused,
            )

        # --- 화면 출력 ---
        cv2.imshow("image-doctor | 객체 추적기", frame)

        # --- 키보드 처리 ---
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            path = f"snapshot_{snapshot_idx:04d}.png"
            cv2.imwrite(path, frame)
            print(f"[정보] 스냅샷 저장: {path}")
            snapshot_idx += 1
        elif key == ord("p"):
            paused = not paused
            print("[정보] 일시정지" if paused else "[정보] 재개")

    cap.release()
    cv2.destroyAllWindows()
    save_stats(args.save, tracker)
    print(f"[정보] 최종 통계: {tracker.stats()}")


if __name__ == "__main__":
    main()
