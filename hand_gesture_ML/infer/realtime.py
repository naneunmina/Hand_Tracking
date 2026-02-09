import cv2
import mediapipe as mp
import pickle
import numpy as np
import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# ===== 경로 설정 =====
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "gesture_mlp.pkl")

# ===== MediaPipe =====
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# ===== 모델 로드 =====
with open(MODEL_PATH, "rb") as f:
  knn = pickle.load(f)

cap = cv2.VideoCapture(0)

gestureImages = {
  "FIST": cv2.imread("images/fist.png", cv2.IMREAD_UNCHANGED),
  "THUMB": cv2.imread("images/thumb.png", cv2.IMREAD_UNCHANGED),
  "V": cv2.imread("images/v.png", cv2.IMREAD_UNCHANGED),
  "PALM": cv2.imread("images/palm.png", cv2.IMREAD_UNCHANGED)
}

def extract_features(hand_landmarks):
  features = []
  for lm in hand_landmarks.landmark:
      features.extend([lm.x, lm.y])
  return np.array(features).reshape(1, -1)

def resizeToFit(img, targetW, targetH):
  h, w = img.shape[:2]
  scale = min(targetW / w, targetH / h)
  newW = int(w * scale)
  newH = int(h * scale)
  return cv2.resize(img, (newW, newH))

def overlayImage(bg, overlay, x, y):
  h, w = overlay.shape[:2]

  if x < 0 or y < 0:
    return
  if y + h > bg.shape[0] or x + w > bg.shape[1]:
    return

  if overlay.shape[2] == 4:
    overlayImg = overlay[:, :, :3]
    alpha = overlay[:, :, 3:] / 255.0

    bg[y:y+h, x:x+w] = (
      (1 - alpha) * bg[y:y+h, x:x+w] +
      alpha * overlayImg
    ).astype("uint8")

  else:
      bg[y:y+h, x:x+w] = overlay

def main():
  with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
  ) as hands:
    while cap.isOpened():
      success, img = cap.read()
      if not success:
        continue

      img = cv2.flip(img, 1)
      h, w, _ = img.shape

      rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      results = hands.process(rgb)

      gesture = None

      canvas = np.zeros((h, w * 2, 3), dtype=np.uint8)

      if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]

        features = extract_features(hand_landmarks)
        gesture = knn.predict(features)[0]

        mp_draw.draw_landmarks(
          img,
          hand_landmarks,
          mp_hands.HAND_CONNECTIONS
        )

        cv2.putText(
          img,
          f"Gesture: {gesture}",
          (10, 40),
          cv2.FONT_HERSHEY_SIMPLEX,
          1.2,
          (0, 255, 0),
          3
        )

      ## 카메라
      canvas[:, :w] = img
      ## 그림
      canvas[:, w:] = (40, 40, 40)

      if gesture and gestureImages[gesture] is not None:
        overlayImg = resizeToFit(gestureImages[gesture], w, h)
        oh, ow = overlayImg.shape[:2]
        y = (h - oh) // 2
        x = w + (w - ow) // 2
        overlayImage(canvas, overlayImg, x, y)
          
      cv2.imshow("Gesture Recognition (ML)", canvas)

      if cv2.waitKey(1) & 0xFF == ord("q"):
        break

  cap.release()
  cv2.destroyAllWindows()

if __name__ == "__main__":
  main()
