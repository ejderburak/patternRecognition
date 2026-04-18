import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import joblib
import time
import warnings
warnings.filterwarnings("ignore")

# === AYARLAR ===
MODEL_PKL = "svm_gesture_model.pkl"
TASK_PATH = "hand_landmarker.task"
NUM_HANDS = 1

WARMUP_SEC = 2
PRED_EVERY_N_FRAMES = 15
CONF_THRESHOLD = 0.0 

#uploading model
model = joblib.load(MODEL_PKL)

# mediapipe task landmarker
base_options = python.BaseOptions(model_asset_path=TASK_PATH)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=NUM_HANDS)
detector = vision.HandLandmarker.create_from_options(options)

#cam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Webcam failed")

time.sleep(WARMUP_SEC)

frame_i = 0
last_print = ""

print("Press q to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_i += 1

    # predict every 15 frames
    if frame_i % PRED_EVERY_N_FRAMES != 0:
        # show demo 
        cv2.imshow("Demo", frame)
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break
        continue

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    result = detector.detect(mp_image)

    if result.hand_landmarks:
        landmarks = result.hand_landmarks[0]

        # normalize according to wrist
        wrist_x = landmarks[0].x
        wrist_y = landmarks[0].y

        row = []
        for lm in landmarks:
            row.append(lm.x - wrist_x)
            row.append(lm.y - wrist_y)

        X = np.array(row, dtype=np.float32).reshape(1, -1)

        pred = model.predict(X)[0]

        # confidince
        conf_str = ""
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[0]
            conf = float(np.max(proba))
            conf_str = f" | Conf: %{conf*100:.1f}"

            if conf < CONF_THRESHOLD:
                pred = "Unknown"

        msg = f"Gesture {pred}{conf_str}"
        if msg != last_print:
            print(msg)
            last_print = msg

    else:
        msg = "Hand cannot be recognized"
        if msg != last_print:
            print(msg)
            last_print = msg

    cv2.imshow("Demo", frame)
    if (cv2.waitKey(1) & 0xFF) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()