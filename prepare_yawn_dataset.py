import cv2
import mediapipe as mp
import os

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

INPUT_DIR = "dataset_yawn/train"
OUTPUT_DIR = "dataset_yawn_cropped/train"

MOUTH_POINTS = [13, 14, 78, 308]

def process_folder(label):
    input_path = os.path.join(INPUT_DIR, label)
    output_path = os.path.join(OUTPUT_DIR, label)

    os.makedirs(output_path, exist_ok=True)

    for img_name in os.listdir(input_path):
        img_path = os.path.join(input_path, img_name)
        img = cv2.imread(img_path)

        if img is None:
            continue

        h, w = img.shape[:2]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            face = results.multi_face_landmarks[0]

            xs = []
            ys = []

            for idx in MOUTH_POINTS:
                x = int(face.landmark[idx].x * w)
                y = int(face.landmark[idx].y * h)
                xs.append(x)
                ys.append(y)

            padding = 30

            x_min = max(min(xs) - padding, 0)
            x_max = min(max(xs) + padding, w)
            y_min = max(min(ys) - padding, 0)
            y_max = min(max(ys) + padding, h)

            mouth_roi = img[y_min:y_max, x_min:x_max]

            save_path = os.path.join(output_path, img_name)
            cv2.imwrite(save_path, mouth_roi)

    print(f"{label} done!")

process_folder("yawn")
process_folder("no_yawn")

print("Dataset preparation complete!")
