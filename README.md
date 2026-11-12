# image-doctor 🩺

**고전적 컴퓨터 비전 기반 실시간 객체 감지·추적·특징점 매칭 데모**

> 과제: AI/SW 심화 — 컴퓨터 비전  
> 제약: YOLO / SSD / R-CNN / dnn / MediaPipe 사용 금지

---

## 프로젝트 구조

```
image-doctor/
├── main.py              # 진입점: 영상 입출력, 배경 차분, 화면 출력 루프
├── tracker.py           # Tracker 클래스 — 그리디 최근접 ID 관리
├── feature_matcher.py   # FeatureMatcher 클래스 — ORB/SIFT 단일 대상 매칭
├── requirements.txt     # 고정 의존성
├── report.md            # 한계 분석 보고서
└── README.md            # 이 파일
```

### 파일별 역할

| 파일 | 담당 |
|---|---|
| `main.py` | CLI 인수, `cv2.VideoCapture`, MOG2 배경 차분, 모폴로지, 감지, 화면 오버레이, 키보드 조작, 통계 CSV 저장 |
| `tracker.py` | `Tracker` 클래스만 — ID 생명주기, 그리디 매칭, 궤적 저장, ID 교환·실패 카운터 |
| `feature_matcher.py` | `FeatureMatcher` 클래스만 — ORB/SIFT 디스크립터 추출, Lowe ratio test, `detected` 플래그 |
| `report.md` | 정량적 성능 표, 실패 조건 분석, 고전 CV vs 딥러닝 결론 |

---

## 빠른 시작

```bash
# 1. 가상 환경 생성
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 2. 의존성 설치
pip install -r requirements.txt

# 3a. 웹캠으로 실행 (기본값)
python main.py

# 3b. 비디오 파일로 실행
python main.py --source path/to/video.mp4

# 3c. 특징점 매칭 대상 지정
python main.py --source path/to/video.mp4 --target path/to/target.jpg

# 3d. SIFT 알고리즘 사용
python main.py --algo SIFT --target path/to/target.jpg
```

### 키보드 조작

| 키 | 동작 |
|---|---|
| `q` | 종료 |
| `s` | 스냅샷 저장 (`snapshot_NNNN.png`) |
| `p` | 일시정지 / 재개 |

---

## CLI 옵션

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--source` | `0` | 웹캠 인덱스 또는 비디오 파일 경로 |
| `--target` | 없음 | 특징점 매칭용 레퍼런스 이미지 경로 |
| `--lr` | `0.005` | MOG2 학습률 (0–1) |
| `--algo` | `ORB` | 특징점 알고리즘: `ORB` 또는 `SIFT` |
| `--save` | `stats.csv` | 세션 통계 저장 CSV 경로 |

---

## 동작 원리

### 1. 배경 차분 — MOG2

```
프레임 → MOG2.apply(lr) → fg_mask (0/127/255)
           ↓
  threshold(>200) → 이진 마스크 (그림자 제거)
           ↓
  MORPH_OPEN  → 노이즈 픽셀 제거
           ↓
  MORPH_CLOSE → 객체 내부 구멍 채움
```

`learning_rate`에 따른 차이:

| learning_rate | 느린 변화 대응 | 빠른 변화 대응 | 조명 변화 오인 |
|---|---|---|---|
| 0.001 (매우 느림) | 그림자 잔상 발생 | 모든 변화를 전경으로 인식 | 낮음 |
| 0.005 (기본값) | 균형 잡힌 적응 | 대부분 정상 처리 | 중간 |
| 0.05 (빠름) | 빠르게 적응 | 정지 객체도 배경으로 흡수 | 높음 |

### 2. 객체 감지

`findContours` → 면적 필터(`MIN_CONTOUR_AREA = 800 px²`) → `(x, y, w, h)` 목록 반환.

### 3. 추적 (`tracker.py`)

```
감지 결과 → 중심점 계산
          → 거리 행렬 [기존 객체 × 신규 감지] 구성
          → 그리디 최근접 이웃 할당 (임계값: 100 px)
          → ID 위치 & 궤적 갱신
          → max_disappeared 초과 시 ID 삭제
          → 미매칭 감지 결과 → 신규 ID 부여
```

**알려진 한계**: 두 객체가 교차할 때 외형 정보 없이 위치만으로 판단하므로 ID가 교환될 수 있음.

### 4. 특징점 매칭 (`feature_matcher.py`)

```
레퍼런스 이미지 → ORB/SIFT 키포인트 + 디스크립터 추출 (초기화 시 1회)
매 프레임      → ORB/SIFT 키포인트 + 디스크립터 추출
              → knnMatch(k=2) → Lowe ratio test (θ=0.75)
              → n_good >= min_matches → TARGET DETECTED
```

---

## 제약사항 준수 확인

- ✅ YOLO / SSD / R-CNN 사용 금지
- ✅ `cv2.dnn` 모듈 사용 금지
- ✅ MediaPipe 사용 금지
- ✅ `Tracker` 클래스를 별도 `tracker.py`로 분리
- ✅ 정량적 성능 측정 결과 `report.md`에 포함
- ✅ ID 교환 횟수 및 조건별 실패율 측정

---

## 실행 환경

- Python 3.8 이상 (3.10+ 권장)
- `opencv-contrib-python >= 4.5` (SIFT 포함)
- `numpy`

> SIFT는 2020년 특허 만료 후 `opencv-contrib-python`에 포함됨.  
> ORB만 사용할 경우 기본 `opencv-python`으로 충분.

---

## 보고서

한계 분석 보고서는 [`report.md`](report.md)를 참고.

포함 내용:
1. 추적 성능 측정 (ID 교환·실패율 표)
2. 특징점 매칭 성능 (회전·가림 조건별)
3. 배경 모델링 학습률 트레이드오프
4. 거리 기반 추적의 본질적 한계
5. Hand-crafted feature의 한계
6. 딥러닝이 필요한 이유 (실습 경험 기반)
