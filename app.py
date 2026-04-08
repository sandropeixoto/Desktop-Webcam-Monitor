import os
import time

import cv2
import streamlit as st

import config
from src.camera import Camera
from src.detector import AIDetector, FaceRecognizer, MotionDetector
from src.recorder import VideoRecorder

st.set_page_config(page_title="Desktop Webcam Monitor", layout="wide")

st.title("📹 Desktop Webcam Monitor")

# Sidebar settings
st.sidebar.header("Settings")
enable_ai = st.sidebar.checkbox("Enable Object Detection (YOLO)", value=True)
enable_face_rec = st.sidebar.checkbox("Enable Face Recognition", value=True)
record_on_motion = st.sidebar.checkbox("Record on Motion", value=True)

# Tabs
tab1, tab2, tab3 = st.tabs(["🔴 Live Feed", "📁 Recordings", "👤 Manage Faces"])


# Helper for saving new face
def save_new_face(frame, name):
    if not name:
        st.error("Please enter a name.")
        return False

    face_dir = "known_faces"
    if not os.path.exists(face_dir):
        os.makedirs(face_dir)

    filename = os.path.join(face_dir, f"{name}.jpg")
    cv2.imwrite(filename, frame)
    st.success(f"Face saved for {name}!")
    # Clear session state after saving
    if "last_unknown_frame" in st.session_state:
        del st.session_state["last_unknown_frame"]
    return True


with tab1:
    st.subheader("Live Stream")

    # Placeholder for the video feed
    frame_placeholder = st.empty()
    status_placeholder = st.empty()

    # Controls
    col1, col2 = st.columns(2)
    start_button = col1.button("Start Monitor")
    stop_button = col2.button("Stop Monitor")

    if stop_button:
        st.session_state["run_monitor"] = False

    # Registration Section (Outside the loop)
    if st.session_state.get("last_unknown_frame") is not None:
        with st.expander("🕵️ Register Last Detected Unknown Person", expanded=True):
            col_img, col_form = st.columns([1, 2])
            col_img.image(
                cv2.cvtColor(st.session_state["last_unknown_frame"], cv2.COLOR_BGR2RGB),
                caption="Unknown Face",
            )
            new_name = col_form.text_input("Name this person:", key="reg_name")
            if col_form.button("Save Identity"):
                if save_new_face(st.session_state["last_unknown_frame"], new_name):
                    st.rerun()

    if start_button:
        cam = Camera(
            camera_id=config.CAMERA_ID,
            width=config.FRAME_WIDTH,
            height=config.FRAME_HEIGHT,
        )

        motion_detector = MotionDetector(threshold=config.MOTION_THRESHOLD)
        ai_detector = AIDetector(
            model_name=config.YOLO_MODEL,
            confidence=config.AI_DETECTION_CONFIDENCE,
            target_classes=config.TARGET_CLASSES,
        )

        face_recognizer = FaceRecognizer(known_faces_dir="known_faces")

        recorder = VideoRecorder(
            output_path=config.RECORDINGS_PATH,
            fps=config.RECORDING_FPS,
            resolution=(config.FRAME_WIDTH, config.FRAME_HEIGHT),
        )

        st.session_state["run_monitor"] = True

        while st.session_state.get("run_monitor", False):
            ret, frame = cam.get_frame()
            if not ret:
                st.error("Failed to access camera.")
                break

            display_frame = frame.copy()
            status_items = []

            # 1. Motion Detection
            motion_detected, contours = motion_detector.detect(frame)

            # 2. AI Object Detection
            if enable_ai:
                detections = ai_detector.detect_objects(frame)
                for det in detections:
                    box = det["box"]
                    label = det["label"]
                    conf = det["conf"]
                    cv2.rectangle(
                        display_frame,
                        (box[0], box[1]),
                        (box[2], box[3]),
                        (0, 255, 0),
                        2,
                    )
                    cv2.putText(
                        display_frame,
                        f"{label} {conf:.2f}",
                        (box[0], box[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )
                    status_items.append(f"AI: {label}")

            # 3. Face Recognition
            if enable_face_rec:
                face_results = face_recognizer.identify_faces(frame)
                for res in face_results:
                    top, right, bottom, left = res["box"]
                    name = res["name"]
                    cv2.rectangle(
                        display_frame, (left, top), (right, bottom), (255, 0, 0), 2
                    )
                    cv2.putText(
                        display_frame,
                        name,
                        (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 0, 0),
                        2,
                    )
                    status_items.append(f"Face: {name}")

                    # If unknown, store frame for registration outside loop
                    if name == "Unknown":
                        st.session_state["last_unknown_frame"] = frame.copy()

            # 4. Recording Logic
            is_recording = False
            if motion_detected:
                if record_on_motion:
                    recorder.start()
                    is_recording = True

                if not status_items:
                    for contour in contours:
                        x, y, w, h = cv2.boundingRect(contour)
                        cv2.rectangle(
                            display_frame, (x, y), (x + w, y + h), (0, 0, 255), 1
                        )

            if is_recording:
                recorder.write(frame)
                cv2.circle(display_frame, (30, 30), 10, (0, 0, 255), -1)
                cv2.putText(
                    display_frame,
                    "REC",
                    (50, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )
            else:
                recorder.stop()

            # Update Feed
            display_frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(
                display_frame_rgb, channels="RGB", use_container_width=True
            )

            status_text = f"Status: {'🔴 Recording' if is_recording else '⚪ IDLE'} | Detected: {', '.join(status_items) if status_items else 'Nothing'}"
            status_placeholder.info(status_text)

            time.sleep(0.01)

        cam.release()
        recorder.stop()
        st.rerun()

with tab2:
    st.subheader("Saved Recordings")
    recording_path = config.RECORDINGS_PATH
    if not os.path.exists(recording_path):
        os.makedirs(recording_path)

    files = [
        f for f in os.listdir(recording_path) if f.endswith((".mp4", ".avi", ".jpg"))
    ]
    files.sort(reverse=True)

    if not files:
        st.write("No recordings found.")
    else:
        for file in files:
            file_path = os.path.join(recording_path, file)
            with st.expander(f"📁 {file}"):
                if file.endswith(".jpg"):
                    st.image(file_path)
                else:
                    try:
                        video_file = open(file_path, "rb")
                        video_bytes = video_file.read()
                        st.video(video_bytes)
                    except Exception as e:
                        st.error(f"Could not play video: {e}")

                if st.button(f"Delete {file}", key=f"del_{file}"):
                    os.remove(file_path)
                    st.rerun()

with tab3:
    st.subheader("Known People")
    face_dir = "known_faces"
    if not os.path.exists(face_dir):
        os.makedirs(face_dir)

    known_files = [
        f for f in os.listdir(face_dir) if f.endswith((".jpg", ".png", ".jpeg"))
    ]

    if not known_files:
        st.write("No one registered yet.")
    else:
        cols = st.columns(4)
        for idx, file in enumerate(known_files):
            with cols[idx % 4]:
                name = os.path.splitext(file)[0]
                st.image(os.path.join(face_dir, file), caption=name)
                if st.button(f"Remove {name}", key=f"rem_{name}"):
                    os.remove(os.path.join(face_dir, file))
                    st.rerun()
