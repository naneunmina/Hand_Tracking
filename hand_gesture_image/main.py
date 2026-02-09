import cv2
import mediapipe as mp
import time
import numpy as np

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

gestureImages = {
    "FIST": cv2.imread("images/fist.png", cv2.IMREAD_UNCHANGED),
    "THUMB": cv2.imread("images/thumb.png", cv2.IMREAD_UNCHANGED),
    "V": cv2.imread("images/v.png", cv2.IMREAD_UNCHANGED)
}

def getGesture(lmList):
    fingers = []

    # 엄지 (x 좌표 기준)
    fingers.append(1 if lmList[4][0] > lmList[3][0] else 0)

    # 나머지 손가락 (y 좌표 기준)
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]

    for tip, pip in zip(tips, pips):
        fingers.append(1 if lmList[tip][1] < lmList[pip][1] else 0)

    # 제스처 분기
    if fingers == [0, 0, 0, 0, 0]:
        return "FIST"
    elif fingers == [1, 0, 0, 0, 0]:
        return "THUMB"
    elif fingers == [0, 1, 1, 0, 0]:
        return "V"

    return None

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
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
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
        for hand_landmarks in results.multi_hand_landmarks:
          lmList = []
          h, w, _ = img.shape

          for lm in hand_landmarks.landmark:
                lmList.append((int(lm.x * w), int(lm.y * h)))

          gesture = getGesture(lmList)

          mp_draw.draw_landmarks(
            img,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS
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
          
      cv2.imshow("Image", canvas)

      if cv2.waitKey(1) & 0xFF == ord('q'):
        break

  cap.release()
  cv2.destroyAllWindows()

if __name__ == "__main__":
  main()