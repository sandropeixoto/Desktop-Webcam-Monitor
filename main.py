import cv2
import time
import config
from src.camera import Camera
from src.detector import MotionDetector, AIDetector
from src.recorder import VideoRecorder

def main():
    print("--- Desktop Webcam Monitor ---")
    print("Press 'q' to quit, 's' for manual snapshot.")
    
    # Initialize components
    cam = Camera(camera_id=config.CAMERA_ID, 
                 width=config.FRAME_WIDTH, 
                 height=config.FRAME_HEIGHT)
    
    motion_detector = MotionDetector(threshold=config.MOTION_THRESHOLD)
    ai_detector = AIDetector(confidence=config.AI_DETECTION_CONFIDENCE)
    
    recorder = VideoRecorder(output_path=config.RECORDINGS_PATH, 
                             fps=config.FPS, 
                             resolution=(config.FRAME_WIDTH, config.FRAME_HEIGHT),
                             codec=config.VIDEO_CODEC)

    recording_timer = 0
    RECORDING_DURATION = 10 # seconds to record after motion/person detected

    try:
        while cam.is_opened:
            frame = cam.get_frame()
            if frame is None:
                break

            display_frame = frame.copy()
            
            # 1. Motion Detection
            motion_detected, contours = motion_detector.detect(frame)
            
            # 2. AI Detection (Faces)
            faces = ai_detector.detect_faces(frame)
            person_detected = len(faces) > 0

            # Logic: If motion or person detected, start/continue recording
            if motion_detected or person_detected:
                if not recorder.is_recording:
                    recorder.start()
                recording_timer = time.time() + RECORDING_DURATION

            # If recording and timer expired, stop
            if recorder.is_recording and time.time() > recording_timer:
                recorder.stop()

            # Write to video if recording
            if recorder.is_recording:
                recorder.write(frame)
                cv2.putText(display_frame, "RECORDING", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Draw visual cues on display frame
            if motion_detected:
                cv2.putText(display_frame, "Motion Detected", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                for contour in contours:
                    (x, y, w, h) = cv2.boundingRect(contour)
                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            for face in faces:
                bbox = face.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x, y, w, h = int(bbox.xmin * iw), int(bbox.ymin * ih), int(bbox.width * iw), int(bbox.height * ih)
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(display_frame, "Face", (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Show the frame
            cv2.imshow('Webcam Monitor', display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                recorder.save_snapshot(frame)

    except KeyboardInterrupt:
        print("\nStopping monitor...")
    finally:
        recorder.stop()
        cam.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
