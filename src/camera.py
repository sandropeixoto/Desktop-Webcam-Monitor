import logging

import cv2


class Camera:
    def __init__(self, camera_id=0, width=640, height=480):
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.cap = cv2.VideoCapture(self.camera_id)

        if not self.cap.isOpened():
            logging.error(f"Cannot open camera {self.camera_id}")
            raise Exception(f"Camera {self.camera_id} not available")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            logging.warning("Can't receive frame (stream end?). Exiting ...")
            return False, None
        return True, frame

    def release(self):
        self.cap.release()
        logging.info("Camera released")

    @property
    def is_opened(self):
        return self.cap.isOpened()
