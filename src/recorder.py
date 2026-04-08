import os
from datetime import datetime

import cv2


class VideoRecorder:
    def __init__(
        self, output_path="recordings", fps=20.0, resolution=(640, 480), codec="mp4v"
    ):
        self.output_path = output_path
        self.fps = fps
        self.resolution = resolution
        self.fourcc = cv2.VideoWriter_fourcc(*codec)
        self.out = None
        self.is_recording = False
        self.filename = ""

        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

    def start(self):
        if not self.is_recording:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.filename = os.path.join(self.output_path, f"record_{timestamp}.mp4")
            self.out = cv2.VideoWriter(
                self.filename, self.fourcc, self.fps, self.resolution
            )
            self.is_recording = True
            print(f"Recording started: {self.filename}")

    def write(self, frame):
        if self.is_recording and self.out:
            self.out.write(frame)

    def stop(self):
        if self.is_recording and self.out:
            self.out.release()
            self.is_recording = False
            print(f"Recording stopped: {self.filename}")

    def save_snapshot(self, frame):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = os.path.join(self.output_path, f"snap_{timestamp}.jpg")
        cv2.imwrite(filename, frame)
        print(f"Snapshot saved: {filename}")
