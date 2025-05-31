import cv2
import numpy as np
import os
import csv
from datetime import datetime
from PIL import Image
import time

def create_user(f_id, name):
    web = cv2.VideoCapture(0)
    web.set(3, 640)
    web.set(4, 480)

    faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    f_dir = 'dataset'
    path = os.path.join(f_dir, name)
    os.makedirs(path, exist_ok=True)

    counter = 0

    while True:
        ret, img = web.read()
        if not ret:
            break
        img = cv2.flip(img, 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        multi_face = faces.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in multi_face:
            counter += 1
            cv2.imwrite(f"{path}/{name}.{f_id}.{counter}.jpg", gray[y:y + h, x:x + w])
            print(f"Captured face {counter}/40 for {name}")

        if counter >= 40:
            break

    web.release()

def train_model():
    database = 'dataset'
    img_dir = [x[0] for x in os.walk(database)][1:]
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faceSamples = []
    ids = []
    name_dict = {}

    for idx, path in enumerate(img_dir):
        name = os.path.basename(path)
        name_dict[idx] = name
        imagePaths = [os.path.join(path, f) for f in os.listdir(path)]

        for imagePath in imagePaths:
            PIL_img = Image.open(imagePath).convert('L')
            img_numpy = np.array(PIL_img, 'uint8')
            faces = detector.detectMultiScale(img_numpy)
            for (x, y, w, h) in faces:
                faceSamples.append(img_numpy[y:y + h, x:x + w])
                ids.append(idx)

    recognizer.train(faceSamples, np.array(ids))
    recognizer.write('trainer.yml')

    with open("names.txt", "w") as f:
        for k, v in name_dict.items():
            f.write(f"{k}:{v}\n")

def load_name_dict():
    name_dict = {}
    if os.path.exists("names.txt"):
        with open("names.txt", "r") as f:
            for line in f:
                key, val = line.strip().split(":")
                name_dict[int(key)] = val
    return name_dict

def log_checkin(name):
    now = datetime.now()
    with open('attendance.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([name, now.strftime("%Y-%m-%d %H:%M:%S"), "", ""])

def log_checkout(name):
    rows = []
    updated = False
    now = datetime.now()

    if not os.path.exists("attendance.csv"):
        return

    with open('attendance.csv', 'r') as f:
        reader = csv.reader(f)
        rows = list(reader)

    for row in reversed(rows[1:]):  # Skip header and check last entries first
        if row[0] == name and row[2] == "":
            checkin_time = datetime.strptime(row[1], "%Y-%m-%d %H:%M:%S")
            duration = now - checkin_time
            row[2] = now.strftime("%Y-%m-%d %H:%M:%S")
            row[3] = str(duration).split('.')[0]  # Remove microseconds
            updated = True
            break

    if updated:
        with open('attendance.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Name", "Check-in", "Check-out", "Duration"])
            writer.writerows(rows[1:])


def recognize(names, status="checkin"):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('trainer.yml')
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    cam = cv2.VideoCapture(0)
    cam.set(3, 640)
    cam.set(4, 480)

    minW = 0.1 * cam.get(3)
    minH = 0.1 * cam.get(4)

    logged = set()
    start_time = time.time()
    duration = 5  # capture for 5 seconds

    try:
        while True:
            ret, img = cam.read()
            if not ret:
                break
            img = cv2.flip(img, 1)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            faces = faceCascade.detectMultiScale(gray, 1.2, 5, minSize=(int(minW), int(minH)))

            for (x, y, w, h) in faces:
                id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
                if confidence < 70:
                    name = names.get(id, "Unknown")
                    if name not in logged:
                        if status == "checkin":
                            log_checkin(name)
                        elif status == "checkout":
                            log_checkout(name)
                        logged.add(name)
                        print(f"{name} marked {status}.")

                    label = f"{name} ({status})"
                else:
                    label = "Unknown"

                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

            # Countdown message
            elapsed = time.time() - start_time
            if elapsed < duration:
                cv2.putText(img, f"Capturing for {duration - int(elapsed)}s...", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 100, 100), 2)
            else:
                cv2.putText(img, "Captured. Press 'Q' to exit.", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 255, 100), 2)

            cv2.imshow('Recognizing Face', img)

            if elapsed >= duration and cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cam.release()
        cv2.destroyAllWindows()
