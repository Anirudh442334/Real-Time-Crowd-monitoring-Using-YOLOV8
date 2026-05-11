import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import pandas as pd
from collections import deque
from scipy.spatial import distance
import time
import os
import requests

# ---------------- AUTO SNAPSHOT PATH ----------------
AUTO_SNAPSHOT_DIR = r"C:\Users\betha\Downloads\CrowdSnapshots"
os.makedirs(AUTO_SNAPSHOT_DIR, exist_ok=True)

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Crowd Monitoring App", layout="wide")
st.title("📷 Real-Time Crowd Monitoring with YOLOv8")

# ---------------- SIDEBAR ----------------
st.sidebar.header("Camera Source")

camera_type = st.sidebar.radio(
    "Select Camera Type",
    ["Laptop Camera", "External Camera", "CCTV/RTSP", "Mobile Camera"]
)

if camera_type == "Laptop Camera":
    camera_source = 0
elif camera_type == "External Camera":
    camera_source = st.sidebar.number_input(
        "External Camera Index", min_value=1, max_value=10, value=1
    )
else:
    camera_source = st.sidebar.text_input(
        "Camera URL", "http://192.168.1.5:8080/video"
    )

# -------- SMS SETTINGS --------
st.sidebar.header("📱 SMS Alert Settings")

to_number = st.sidebar.text_input(
    "Alert Mobile Number (without +91)",
    "9876543210"
)

# -------- DETECTION SETTINGS --------
st.sidebar.header("Detection Settings")

confidence = st.sidebar.slider("Detection Confidence", 0.1, 1.0, 0.5)
max_people = st.sidebar.number_input("People Alert Threshold", 0, 100, 5)
social_dist_thresh = st.sidebar.slider("Social Distance (pixels)", 50, 300, 100)

start_btn = st.sidebar.button("▶ Start Camera")
stop_btn = st.sidebar.button("⏹ Stop Camera")
snapshot_btn = st.sidebar.button("📸 Take Snapshot")

# ---------------- SMS FUNCTION ----------------
def send_sms_alert(people_count, to_number):
    url = "https://www.fast2sms.com/dev/bulkV2"

    payload = {
        "route": "q",
        "message": f" ALERT! {people_count} people detected. Limit exceeded.",
        "language": "english",
        "flash": 0,
        "numbers": to_number
    }

    headers = {
        "authorization": "mMkcIQUNBaHsA5f3yZbdFROj4zKXPiCq2V6wE7v9Yt0JuTroDSJAv6IRhbB7XLdNxt5EG2fKlwSPsHU0",
        "Content-Type": "application/json"
    }

    response = requests.post(url, json=payload, headers=headers)
    return response.json()

# ---------------- SESSION STATE ----------------
if "run" not in st.session_state:
    st.session_state.run = False

if "cap" not in st.session_state:
    st.session_state.cap = None

if "model" not in st.session_state:
    st.session_state.model = YOLO("yolov8n.pt")

if "counts" not in st.session_state:
    st.session_state.counts = deque(maxlen=50)
    st.session_state.times = deque(maxlen=50)

if "sms_sent" not in st.session_state:
    st.session_state.sms_sent = False

if "auto_snapshot_taken" not in st.session_state:
    st.session_state.auto_snapshot_taken = False

# ---------------- PLACEHOLDERS ----------------
frame_holder = st.empty()
count_holder = st.empty()
chart_holder = st.empty()

os.makedirs("snapshots", exist_ok=True)

# ---------------- BUTTON LOGIC ----------------
if start_btn and not st.session_state.run:
    st.session_state.cap = cv2.VideoCapture(camera_source)
    if st.session_state.cap.isOpened():
        st.session_state.run = True
        st.success("✅ Camera started")
    else:
        st.error("❌ Unable to access camera")

if stop_btn:
    st.session_state.run = False
    if st.session_state.cap:
        st.session_state.cap.release()
        st.session_state.cap = None
    st.warning("🛑 Camera stopped")

# ---------------- MAIN CAMERA LOOP ----------------
if st.session_state.run and st.session_state.cap:

    while st.session_state.run and st.session_state.cap.isOpened():
        ret, frame = st.session_state.cap.read()
        if not ret:
            st.error("❌ Failed to read frame")
            break

        # YOLO detection
        results = st.session_state.model(frame, conf=confidence, verbose=False)

        boxes = []
        for box in results[0].boxes:
            if int(box.cls) == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                boxes.append((x1, y1, x2, y2))

        person_count = len(boxes)

        # ---- SOCIAL DISTANCE ----
        violations = set()
        centers = [((x1 + x2)//2, (y1 + y2)//2) for x1, y1, x2, y2 in boxes]

        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                if distance.euclidean(centers[i], centers[j]) < social_dist_thresh:
                    violations.add(i)
                    violations.add(j)

        # Draw boxes
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            color = (0, 0, 255) if i in violations else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # ---- HEATMAP ----
        heatmap = np.zeros(frame.shape[:2], dtype=np.uint8)
        for (x1, y1, x2, y2) in boxes:
            heatmap[y1:y2, x1:x2] += 40

        heatmap = cv2.applyColorMap(np.clip(heatmap, 0, 255), cv2.COLORMAP_JET)
        output = cv2.addWeighted(frame, 0.7, heatmap, 0.3, 0)

        frame_holder.image(
            cv2.cvtColor(output, cv2.COLOR_BGR2RGB),
            channels="RGB"
        )

        # ---- ALERT + SMS + SNAPSHOT ----
        if person_count > max_people:
            count_holder.error(f"🚨 ALERT: {person_count} people detected!")

            if not st.session_state.sms_sent:
                res = send_sms_alert(person_count, to_number)

                if res.get("return") == True:
                    st.success("📩 SMS Sent Successfully!")
                else:
                    st.error("❌ SMS Failed!")

                st.session_state.sms_sent = True

            if not st.session_state.auto_snapshot_taken:
                filename = os.path.join(
                    AUTO_SNAPSHOT_DIR,
                    f"auto_{int(time.time())}.jpg"
                )
                cv2.imwrite(filename, output)
                st.success(f"📸 Snapshot saved: {filename}")
                st.session_state.auto_snapshot_taken = True

        else:
            count_holder.success(f"✅ People detected: {person_count}")
            st.session_state.sms_sent = False
            st.session_state.auto_snapshot_taken = False

        # ---- GRAPH ----
        st.session_state.counts.append(person_count)
        st.session_state.times.append(pd.Timestamp.now().strftime("%H:%M:%S"))

        df = pd.DataFrame({
            "Time": list(st.session_state.times),
            "Count": list(st.session_state.counts)
        })

        chart_holder.line_chart(df.set_index("Time"))

        # ---- MANUAL SNAPSHOT ----
        if snapshot_btn:
            filename = f"snapshots/manual_{int(time.time())}.jpg"
            cv2.imwrite(filename, output)
            st.success(f"📸 Snapshot saved: {filename}")

        time.sleep(0.03)