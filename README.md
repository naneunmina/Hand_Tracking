# Hand_Tracking (Python + OpenCV)

Python과 OpenCV, MediaPipe를 사용한 핸드 트래킹 및 손 제스처 인식 프로젝트입니다.  
하드코딩 기반 제스처 인식과 머신러닝 기반 제스처 인식을 모두 포함합니다.

---

## Environment

- Python 3.11
- OpenCV
- MediaPipe 0.10.14
- scikit-learn (ML 버전 사용 시)

---

## 1. Basic Hand Tracking

### `track01.py`

- MediaPipe 핸드 트래킹만 연결된 가장 기본적인 예제
- 손 랜드마크를 실시간으로 확인 가능

**실행 방법**

`python track01.py`

## 2. Rule-Based Gesture Recognition (Hard Coding)
### `hand_gesture_image/main.py`
- 손 랜드마크 좌표를 기준으로 손가락 상태를 하드코딩하여 제스처 판별
- 인식된 손동작에 따라 오른쪽 화면에 이미지 표시

**특징**

- 학습 데이터 없이 규칙 기반으로 동작
- 제스처 추가 시 조건문 직접 수정 필요

**실행 방법**

`python main.py`


## 3. Machine Learning Gesture Recognition
### 3-1. Data Collection
### `hand_gesture_ML/collect/collect_data.py`
- MediaPipe 랜드마크 좌표를 사용해 제스처 데이터 수집

- 8번째 줄에서 제스처 라벨 수정 후 실행
`GESTURE_LABEL = "{제스처 이름}"`

**실행 방법**

`python collect_data.py`

### 3-2. Model Training
### `hand_gesture_ML/train/`
폴더에서 원하는 모델을 학습합니다.

`python train_knn.py`
`python train_mlp.py`
`python train_svm.py`

학습 완료 시 `gesture_모델이름.pkl` 파일이 생성됩니다.

### 3-3. Realtime Inference
### `hand_gesture_ML/infer/realtime.py`
12번째 줄에서 사용할 모델 경로 수정

`MODEL_PATH = "../train/gesture_knn.pkl"`
(예: gesture_mlp.pkl, gesture_svm.pkl 등)

**실행 방법**

`python realtime.py`
