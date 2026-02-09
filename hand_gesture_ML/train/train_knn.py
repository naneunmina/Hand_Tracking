import os
import csv
import pickle
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_DIR = os.path.join(BASE_DIR, "collect/dataset")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "gesture_knn.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)

def load_dataset():
    X = []
    y = []

    for file in os.listdir(DATASET_DIR):
        if not file.endswith(".csv"):
            continue

        label = file.replace(".csv", "")
        file_path = os.path.join(DATASET_DIR, file)

        with open(file_path, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                features = list(map(float, row[:-1]))
                X.append(features)
                y.append(label)

    return np.array(X), np.array(y)

def main():
    print("📥 데이터 로딩 중...")
    X, y = load_dataset()

    print(f"총 샘플 수: {len(X)}")
    print(f"특징 차원: {X.shape[1]}")

    # train / test 분리
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # KNN 모델
    knn = KNeighborsClassifier(
        n_neighbors=5,
        weights="distance",
        metric="euclidean"
    )

    print("🧠 학습 중...")
    knn.fit(X_train, y_train)

    # 평가
    y_pred = knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"✅ 테스트 정확도: {acc * 100:.2f}%")

    # 모델 저장
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(knn, f)

    print(f"💾 모델 저장 완료: {MODEL_PATH}")

if __name__ == "__main__":
    main()
