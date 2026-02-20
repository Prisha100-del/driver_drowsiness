import cv2
import mediapipe as mp
import numpy as np
import time
import threading
from tensorflow.keras.models import load_model
import pyttsx3

# =========================================================
# BULLETPROOF VOICE SYSTEM
# =========================================================
engine = pyttsx3.init()
engine.setProperty('rate', 170)

voice_lock = threading.Lock()
is_speaking = False

def speak(text):
    global is_speaking

    def run():
        global is_speaking
        with voice_lock:
            if not is_speaking:
                is_speaking = True
                engine.stop()
                engine.say(text)
                engine.runAndWait()
                is_speaking = False

    threading.Thread(target=run, daemon=True).start()

# =========================================================
# LOAD MODELS
# =========================================================
eye_model = load_model("eye_cnn_model.h5")
yawn_model = load_model("yawn_cnn_model.h5")

# =========================================================
# PREPROCESS
# =========================================================
def preprocess(img):
    img = cv2.resize(img, (64, 64))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img / 255.0
    img = img.reshape(1, 64, 64, 1)
    return img

# =========================================================
# PREDICTIONS
# =========================================================
def predict_eye(eye_img):
    if eye_img is None or eye_img.size == 0:
        return "OPEN"

    pred = eye_model.predict(preprocess(eye_img), verbose=0)[0][0]
    return "CLOSED" if pred > 0.6 else "OPEN"

def predict_yawn(mouth_img):
    if mouth_img is None or mouth_img.size == 0:
        return "NO_YAWN"

    pred = yawn_model.predict(preprocess(mouth_img), verbose=0)[0][0]
    return "YAWN" if pred > 0.6 else "NO_YAWN"

# =========================================================
# MEDIAPIPE
# =========================================================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH_POINTS = [13, 14, 78, 308]

# =========================================================
# PARAMETERS
# =========================================================
CLOSED_FRAMES_THRESHOLD = 6
YAWN_FRAMES_THRESHOLD = 8
DISTRACTION_TIME = 1.2
SPEECH_COOLDOWN = 4   # seconds

# =========================================================
# STATE VARIABLES
# =========================================================
closed_frame_count = 0
yawn_frame_count = 0
distraction_start = None

last_spoken_time = 0
last_spoken_status = ""

# =========================================================
# CAMERA
# =========================================================
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    status = "ALERT"
    color = (0, 255, 0)

    if results.multi_face_landmarks:
        face = results.multi_face_landmarks[0]

        # ---------------- EYE ----------------
        left_eye = []
        for idx in LEFT_EYE:
            x = int(face.landmark[idx].x * w)
            y = int(face.landmark[idx].y * h)
            left_eye.append([x, y])

        x_min = max(min(p[0] for p in left_eye) - 5, 0)
        x_max = min(max(p[0] for p in left_eye) + 5, w)
        y_min = max(min(p[1] for p in left_eye) - 5, 0)
        y_max = min(max(p[1] for p in left_eye) + 5, h)

        eye_roi = frame[y_min:y_max, x_min:x_max]
        eye_state = predict_eye(eye_roi)

        if eye_state == "CLOSED":
            closed_frame_count += 1
        else:
            closed_frame_count = 0

        drowsy = closed_frame_count >= CLOSED_FRAMES_THRESHOLD

        # ---------------- YAWN ----------------
        xs, ys = [], []
        for idx in MOUTH_POINTS:
            x = int(face.landmark[idx].x * w)
            y = int(face.landmark[idx].y * h)
            xs.append(x)
            ys.append(y)

        x_min = max(min(xs) - 25, 0)
        x_max = min(max(xs) + 25, w)
        y_min = max(min(ys) - 25, 0)
        y_max = min(max(ys) + 25, h)

        mouth_roi = frame[y_min:y_max, x_min:x_max]
        yawn_state = predict_yawn(mouth_roi)

        if yawn_state == "YAWN":
            yawn_frame_count += 1
        else:
            yawn_frame_count = 0

        yawning = yawn_frame_count >= YAWN_FRAMES_THRESHOLD

        # ---------------- DISTRACTION ----------------
        nose_x = int(face.landmark[1].x * w)
        left_eye_x = int(face.landmark[33].x * w)
        right_eye_x = int(face.landmark[263].x * w)
        center_eyes = (left_eye_x + right_eye_x) // 2

        offset = abs(nose_x - center_eyes)

        distracted = False
        if offset > 40:
            if distraction_start is None:
                distraction_start = time.time()
            elif time.time() - distraction_start > DISTRACTION_TIME:
                distracted = True
        else:
            distraction_start = None

        # ---------------- FINAL STATUS ----------------
        if drowsy:
            status = "DROWSY"
            color = (0, 0, 255)
        elif yawning:
            status = "YAWNING"
            color = (0, 165, 255)
        elif distracted:
            status = "DISTRACTED"
            color = (255, 0, 0)
        else:
            status = "ALERT"
            color = (0, 255, 0)

        # =========================================================
        # VOICE ALERT SYSTEM (WORKS MULTIPLE TIMES)
        # =========================================================
        current_time = time.time()

        if status in ["DROWSY", "YAWNING", "DISTRACTED"]:

            if (status != last_spoken_status) or \
               (current_time - last_spoken_time > SPEECH_COOLDOWN):

                if status == "DROWSY":
                    speak("Warning. You are feeling drowsy. Please take a break immediately.")
                elif status == "YAWNING":
                    speak("You seem tired. Please stay alert.")
                elif status == "DISTRACTED":
                    speak("Please focus on the road.")

                last_spoken_status = status
                last_spoken_time = current_time

        else:
            last_spoken_status = ""

        # ---------------- DISPLAY ----------------
        cv2.putText(frame, f"Eye: {eye_state}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.putText(frame, f"Yawn: {yawn_state}", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.putText(frame, f"STATUS: {status}", (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)

    cv2.imshow("Driver Monitoring System", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
