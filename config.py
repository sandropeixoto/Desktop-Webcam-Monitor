import os

# Camera Settings
CAMERA_ID = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 20.0

# Detection Settings
MOTION_THRESHOLD = 500  # Contours area threshold
AI_DETECTION_CONFIDENCE = 0.5
YOLO_MODEL = "yolov8n.pt"  # nano model (fastest)
TARGET_CLASSES = [
    "person",
    "dog",
    "cat",
    "car",
    "motorcycle",
    "bird",
]  # Classes to track

# Recording Settings
RECORDINGS_PATH = "recordings"
VIDEO_CODEC = "mp4v"  # codec for .mp4
RECORDING_FPS = 20.0

# Create output dir if not exists
if not os.path.exists(RECORDINGS_PATH):
    os.makedirs(RECORDINGS_PATH)
