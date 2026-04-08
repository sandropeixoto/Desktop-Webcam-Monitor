import cv2
import time
import config
from src.camera import Camera
from src.detector import MotionDetector, AIDetector
from src.recorder import VideoRecorder

def main():
    print("--- Desktop Webcam Monitor ---")
    print(f"Tracking: {config.TARGET_CLASSES}")
    print("Press 'q' to quit, 's' for manual snapshot.")
    
    cam = Camera(camera_id=config.CAMERA_ID, 
                 width=config.FRAME_WIDTH, 
                 height=config.FRAME_HEIGHT)
    
    motion_detector = MotionDetector(threshold=config.MOTION_THRESHOLD)
    ai_detector = AIDetector(model_name=config.YOLO_MODEL, 
                            confidence=config.AI_DETECTION_CONFIDENCE,
                            target_classes=config.TARGET_CLASSES)
    
    recorder = VideoRecorder(output_path=config.RECORDINGS_PATH, 
                             fps=config.FPS, 
                             resolution=(config.FRAME_WIDTH, config.FRAME_HEIGHT),
                             codec=config.VIDEO_CODEC)

    recording_timer = 0
    RECORDING_DURATION = 10 

    try:
        while cam.is_opened:
            frame = cam.get_frame()
            if frame is None:
                break

            display_frame = frame.copy()
            
            # 1. Motion Detection
            motion_detected, contours = motion_detector.detect(frame)
            
            # 2. AI Object Detection (YOLO)
            objects = ai_detector.detect_objects(frame)
            object_detected = len(objects) > 0

            # Logic: If motion or target object detected, record
            if motion_detected or object_detected:
                if not recorder.is_recording:
                    recorder.start()
                recording_timer = time.time() + RECORDING_DURATION

            if recorder.is_recording and time.time() > recording_timer:
                recorder.stop()

            if recorder.is_recording:
                recorder.write(frame)
                cv2.putText(display_frame, "RECORDING", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # Draw Objects
            for obj in objects:
                x1, y1, x2, y2 = obj['box']
                label = f"{obj['label']} {obj['conf']:.2f}"
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(display_frame, label, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Draw Motion (only if no objects to keep UI clean)
            if motion_detected and not object_detected:
                for contour in contours:
                    (x, y, w, h) = cv2.boundingRect(contour)
                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 1)

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
