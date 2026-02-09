import cv2
import mediapipe as mp
import csv
import os
import time

# ===== 설정 =====
GESTURE_LABEL = "PALM"   # ← 지금 수집할 제스처 이름
SAVE_PATH = f"dataset/{GESTURE_LABEL}.csv"
SAMPLES = 200            # 수집할 샘플 수
DELAY = 0.05             # 저장 간 딜레이 (초)

# =================

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

os.makedirs("dataset", exist_ok=True)

def main():
    count = 0

    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    ) as hands, open(SAVE_PATH, "a", newline="") as f:

        writer = csv.writer(f)

        while cap.isOpened() and count < SAMPLES:
            success, frame = cap.read()
            if not success:
                continue

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]

                row = []
                for lm in hand_landmarks.landmark:
                    row.extend([lm.x, lm.y])  # 정규화 좌표

                row.append(GESTURE_LABEL)
                writer.writerow(row)
                count += 1

                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                time.sleep(DELAY)

            cv2.putText(
                frame,
                f"{GESTURE_LABEL} : {count}/{SAMPLES}",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            cv2.imshow("Collecting Data", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("✅ 데이터 수집 완료")

if __name__ == "__main__":
    main()
