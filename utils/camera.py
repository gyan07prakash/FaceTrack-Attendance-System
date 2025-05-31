import cv2
import os
import numpy as np
from PIL import Image
from datetime import datetime

DATASET_DIR = 'dataset'
TRAINER_PATH = 'trainer.yml'
NAME_MAP_PATH = 'names.txt'
ATTENDANCE_CSV = 'attendance.csv'

os.makedirs(DATASET_DIR, exist_ok=True)


def create_user(user_id, user_name):
    path = os.path.join(DATASET_DIR, user_name)
    os.makedirs(path, exist_ok=True)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    count = 0
    while True:
        ret, img = cap.read()
        img = cv2.flip(img, 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            count += 1
            face_img = gray[y:y + h, x:x + w]
            cv2.imwrite(f"{path}/{user_name}.{user_id}.{count}.jpg", face_img)
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.imshow('Capturing Faces', img)
        k = cv2.waitKey(100) & 0xff
        if k == 27 or count >= 40:
            break

    cap.release()
    cv2.destroyAllWindows()


def train_model():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    image_paths = []
    label_ids = []
    label_dict = {}
    current_id = 0

    for root, dirs, files in os.walk(DATASET_DIR):
        for file in files:
            if file.endswith(".jpg"):
                path = os.path.join(root, file)
                label = os.path.basename(root)

                if label not in label_dict:
                    label_dict[label] = current_id
                    current_id += 1

                image_paths.append((path, label_dict[label]))

    x_train = []
    y_labels = []

    for path, label in image_paths:
        img = Image.open(path).convert("L")
        img_np = np.array(img, "uint8")
        faces = face_cascade.detectMultiScale(img_np)
        for (x, y, w, h) in faces:
            x_train.append(img_np[y:y + h, x:x + w])
            y_labels.append(label)

    recognizer.train(x_train, np.array(y_labels))
    recognizer.write(TRAINER_PATH)

    with open(NAME_MAP_PATH, 'w') as f:
        for name, idx in label_dict.items():
            f.write(f"{idx}:{name}\n")


def load_name_dict():
    if not os.path.exists(NAME_MAP_PATH):
        return {}
    with open(NAME_MAP_PATH, 'r') as f:
        return {int(k): v for line in f for k, v in [line.strip().split(":")]}


def recognize_and_mark():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(TRAINER_PATH)

    name_dict = load_name_dict()
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    seen = set()
    while True:
        ret, img = cap.read()
        img = cv2.flip(img, 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 5)

        for (x, y, w, h) in faces:
            id_, confidence = recognizer.predict(gray[y:y + h, x:x + w])
            if confidence < 70:
                name = name_dict.get(id_, 'Unknown')
                if name not in seen:
                    seen.add(name)
                    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    with open(ATTENDANCE_CSV, 'a') as f:
                        f.write(f"{name},{id_},{now}\n")
                cv2.putText(img, name, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(img, "Unknown", (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.imshow('Attendance', img)
        if cv2.waitKey(100) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

