import cv2
import numpy as np
import mediapipe as mp

class MotionDetector:
    def __init__(self, threshold=500):
        self.threshold = threshold
        self.prev_frame = None

    def detect(self, frame):
        # Grayscale and Blur to reduce noise
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if self.prev_frame is None:
            self.prev_frame = gray
            return False, []

        # Difference between current and previous frame
        frame_delta = cv2.absdiff(self.prev_frame, gray)
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
        
        # Dilate for filling gaps
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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
    def __init__(self, confidence=0.5):
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=confidence)
        
        # Person detection (simplified via face for now)
        # For full person detection, we could use MediaPipe Pose or Object Detection
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=confidence, min_tracking_confidence=confidence)

    def detect_faces(self, frame):
        results = self.face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return results.detections if results.detections else []

    def detect_pose(self, frame):
        results = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return results.pose_landmarks if results.pose_landmarks else None
