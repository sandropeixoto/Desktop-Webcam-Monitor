import os

import cv2
import face_recognition
import mediapipe as mp
import numpy as np
from ultralytics import YOLO


class MotionDetector:
    def __init__(self, threshold=500):
        self.threshold = threshold
        self.prev_frame = None

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if self.prev_frame is None:
            self.prev_frame = gray
            return False, []

        frame_delta = cv2.absdiff(self.prev_frame, gray)
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)

        contours, _ = cv2.findContours(
            thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        motion_detected = False
        detected_contours = []

        for contour in contours:
            if cv2.contourArea(contour) < self.threshold:
                continue
            motion_detected = True
            detected_contours.append(contour)

        self.prev_frame = gray
        return motion_detected, detected_contours


class AIDetector:
    def __init__(self, model_name="yolov8n.pt", confidence=0.5, target_classes=None):
        self.confidence = confidence
        self.target_classes = target_classes or []

        # Initialize YOLOv8
        try:
            self.model = YOLO(model_name)
            self.has_yolo = True
            print(f"✅ YOLOv8 Model {model_name} loaded.")
        except Exception as e:
            print(f"⚠️ Error loading YOLO: {e}. AI features disabled.")
            self.has_yolo = False

        # Fallback to Face Detection (MediaPipe) if YOLO fails or for specific face tracking
        try:
            self.mp_face_detection = mp.solutions.face_detection
            self.face_detection = self.mp_face_detection.FaceDetection(
                model_selection=0, min_detection_confidence=confidence
            )
            self.has_mp = True
        except AttributeError:
            self.has_mp = False

    def detect_objects(self, frame):
        if not self.has_yolo:
            return []

        results = self.model.predict(frame, conf=self.confidence, verbose=False)
        detected_objects = []

        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                label = r.names[cls_id]

                # Filter by target classes if specified
                if not self.target_classes or label in self.target_classes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    detected_objects.append(
                        {
                            "box": [int(x1), int(y1), int(x2), int(y2)],
                            "label": label,
                            "conf": float(box.conf[0]),
                        }
                    )
        return detected_objects

    def detect_faces(self, frame):
        if not self.has_mp:
            return []
        results = self.face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return results.detections if results.detections else []


class FaceRecognizer:
    def __init__(self, known_faces_dir="known_faces"):
        self.known_faces_dir = known_faces_dir
        self.known_face_encodings = []
        self.known_face_names = []
        self.load_known_faces()

    def load_known_faces(self):
        self.known_face_encodings = []
        self.known_face_names = []

        if not os.path.exists(self.known_faces_dir):
            os.makedirs(self.known_faces_dir)
            return

        for filename in os.listdir(self.known_faces_dir):
            if filename.endswith((".jpg", ".png", ".jpeg")):
                path = os.path.join(self.known_faces_dir, filename)
                try:
                    image = face_recognition.load_image_file(path)
                    encodings = face_recognition.face_encodings(image)
                    if encodings:
                        self.known_face_encodings.append(encodings[0])
                        # Use filename without extension as name
                        name = os.path.splitext(filename)[0]
                        self.known_face_names.append(name)
                except Exception as e:
                    print(f"Error loading face {filename}: {e}")
        print(
            f"Loaded {len(self.known_face_names)} known faces: {self.known_face_names}"
        )

    def identify_faces(self, frame):
        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Find all faces and their encodings in the current frame
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(
            rgb_small_frame, face_locations
        )

        face_results = []
        for (top, right, bottom, left), face_encoding in zip(
            face_locations, face_encodings
        ):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            matches = face_recognition.compare_faces(
                self.known_face_encodings, face_encoding, tolerance=0.6
            )
            name = "Unknown"

            if matches:
                face_distances = face_recognition.face_distance(
                    self.known_face_encodings, face_encoding
                )
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = self.known_face_names[best_match_index]

            face_results.append(
                {
                    "box": (top, right, bottom, left),
                    "name": name,
                    "encoding": face_encoding,
                }
            )

        return face_results
