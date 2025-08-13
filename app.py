# app.py — Streamlit Real‑Time Traffic Violation Detector (with mobile IP camera, type detection, one-time line crossing, Excel download)
import os
import io
from datetime import datetime
from collections import defaultdict

import cv2 as cv
import numpy as np
import pandas as pd
import streamlit as st
from ultralytics import YOLO

st.set_page_config(page_title="Traffic Violation Detector", layout="wide")

# ------------------------------
# Model loading (cached)
# ------------------------------
@st.cache_resource(show_spinner=True)
def load_model(weights_path: str = "best.pt"):
    return YOLO(weights_path)

model = load_model("best.pt")  # put your weights alongside app.py
CLASS_NAMES = model.names  # YOLO class index -> label
VEHICLE_CLASS_IDS = {1, 2, 3, 5, 7}  # bicycle, car, motorcycle, bus, truck (COCO indices)

# ------------------------------
# Session state
# ------------------------------
if "log_df" not in st.session_state:
    st.session_state.log_df = pd.DataFrame(
        columns=["Timestamp", "Track ID", "Vehicle Type", "Action", "x1", "y1", "x2", "y2"]
    )
if "violations" not in st.session_state:
    st.session_state.violations = 0
if "passes" not in st.session_state:
    st.session_state.passes = 0
if "crossed_ids" not in st.session_state:
    st.session_state.crossed_ids = set()  # track IDs that already crossed this session
if "obj_y_hist" not in st.session_state:
    st.session_state.obj_y_hist = defaultdict(list)  # id -> [y positions]
if "running" not in st.session_state:
    st.session_state.running = False

# ------------------------------
# Sidebar controls
# ------------------------------
st.sidebar.header("Settings")
source_mode = st.sidebar.radio("Source", ["IP Camera (Phone)", "Webcam Index"], index=0)
ip_url = st.sidebar.text_input("IP Stream URL", value="http://192.168.1.5:8080/video")
webcam_index = st.sidebar.number_input("Webcam index", min_value=0, step=1, value=0)

conf = st.sidebar.slider("YOLO confidence", 0.1, 0.9, 0.35, 0.05)
iou = st.sidebar.slider("NMS IoU", 0.2, 0.9, 0.5, 0.05)
resize_w = st.sidebar.slider("Resize width (px)", 480, 1280, 854, 2)

st.sidebar.subheader("Red‑light logic")
red_light_mode = st.sidebar.radio("Mode", ["Manual Toggle", "Always Red", "Always Green"], index=0)
manual_red = st.sidebar.toggle("Red ON (manual)", value=False, disabled=(red_light_mode != "Manual Toggle"))

st.sidebar.subheader("Stop line")
line_pos_pct = st.sidebar.slider("Stop line Y (% of height)", 10, 90, 65)

with st.sidebar:
    c1, c2 = st.columns(2)
    if c1.button("Start", use_container_width=True):
        st.session_state.running = True
    if c2.button("Stop", use_container_width=True):
        st.session_state.running = False

# ------------------------------
# Utilities
# ------------------------------
def get_capture():
    if source_mode == "IP Camera (Phone)":
        return cv.VideoCapture(ip_url)
    return cv.VideoCapture(int(webcam_index))


def compute_line_y(h: int) -> int:
    return int(h * (line_pos_pct / 100.0))


def red_is_on() -> bool:
    if red_light_mode == "Always Red":
        return True
    if red_light_mode == "Always Green":
        return False
    return manual_red


def draw_traffic_light(frame, is_red: bool):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = w - 90, 10, w - 20, 120
    cv.rectangle(frame, (x1, y1), (x2, y2), (50, 50, 50), -1)
    cv.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
    cv.circle(frame, (w - 55, 40), 15, (0, 0, 255) if is_red else (0, 0, 60), -1)
    cv.circle(frame, (w - 55, 90), 15, (0, 255, 0) if not is_red else (0, 60, 0), -1)


def log_event(ts: str, track_id: int, vtype: str, action: str, x1: int, y1: int, x2: int, y2: int):
    st.session_state.log_df.loc[len(st.session_state.log_df)] = [ts, track_id, vtype, action, x1, y1, x2, y2]


def to_excel_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Violations")
    buf.seek(0)
    return buf.getvalue()

# ------------------------------
# UI placeholders
# ------------------------------
col_stream, col_stats = st.columns([3, 1])
frame_ph = col_stream.empty()
with col_stats:
    st.subheader("Session stats")
    stats_ph = st.empty()
    if not st.session_state.log_df.empty:
        st.download_button(
            label="📥 Download Excel log",
            data=to_excel_bytes(st.session_state.log_df),
            file_name="violation_log.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
    st.caption("Tip: use an Android IP camera app (e.g., 'IP Webcam'), same Wi‑Fi as this machine, copy the /video URL.")

# ------------------------------
# Main loop
# ------------------------------
if st.session_state.running:
    cap = get_capture()
    ok, frame0 = cap.read()
    if not ok:
        st.error("Could not read from the selected source. Check IP/port or webcam index.")
        st.stop()

    # compute geometry
    h0, w0 = frame0.shape[:2]
    new_w = int(resize_w)
    new_h = int(h0 * (new_w / w0))

    while st.session_state.running:
        ok, frame = cap.read()
        if not ok:
            st.warning("Stream ended or unavailable.")
            break

        # resize keeping aspect ratio
        frame_rs = cv.resize(frame, (new_w, new_h))
        line_y = compute_line_y(new_h)

        # Track only vehicle classes
        try:
            results = model.track(
                frame_rs,
                persist=True,
                conf=conf,
                iou=iou,
                classes=list(VEHICLE_CLASS_IDS),
            )
        except Exception as e:
            st.error(f"Model inference failed: {e}")
            break

        annotated = results[0].plot()

        # Draw stop line
        cv.line(annotated, (10, line_y), (annotated.shape[1] - 10, line_y), (0, 0, 255), 2)

        # Process tracks and detect first-time crossings
        if results[0].boxes.id is not None:
            for box in results[0].boxes:
                # Safe parsing
                try:
                    tid = int(box.id)
                except Exception:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0]) if hasattr(box, "cls") else -1
                vtype = CLASS_NAMES.get(cls_id, str(cls_id))

                bottom_y = y2  # bottom of bbox
                st.session_state.obj_y_hist[tid].append(bottom_y)
                hist = st.session_state.obj_y_hist[tid]
                if len(hist) >= 2:
                    prev_y, curr_y = hist[-2], hist[-1]
                    # crossing event = moved from above line to at/over the line
                    if (prev_y < line_y <= curr_y) and (tid not in st.session_state.crossed_ids):
                        action = "Violation" if red_is_on() else "Pass"
                        if action == "Violation":
                            st.session_state.violations += 1
                        else:
                            st.session_state.passes += 1
                        st.session_state.crossed_ids.add(tid)
                        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        log_event(ts, tid, vtype, action, x1, y1, x2, y2)
                        # flash highlight
                        cv.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255) if action == "Violation" else (0, 255, 0), 4)
                        cv.putText(annotated, f"{action}!", (x1, max(0, y1 - 10)), cv.FONT_HERSHEY_SIMPLEX, 0.7,
                                   (0, 0, 255) if action == "Violation" else (0, 128, 0), 2)

        # Overlays
        draw_traffic_light(annotated, red_is_on())
        cv.rectangle(annotated, (5, 4), (360, 58), (255, 255, 255), -1)
        cv.rectangle(annotated, (5, 4), (360, 58), (0, 0, 0), 2)
        cv.putText(annotated, f"Passes: {st.session_state.passes}", (15, 28), cv.FONT_HERSHEY_SIMPLEX, 0.65, (0, 128, 0), 2)
        cv.putText(annotated, f"Violations: {st.session_state.violations}", (15, 50), cv.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 200), 2)

        # Show frame in Streamlit (BGR->RGB)
        frame_ph.image(cv.cvtColor(annotated, cv.COLOR_BGR2RGB), channels="RGB")

        # Stats panel
        with col_stats:
            stats_ph.markdown(
                f"""
**Stop line Y:** {line_y}px  
**Red light:** {'ON' if red_is_on() else 'OFF'}  
**Crossed IDs:** {len(st.session_state.crossed_ids)}  
**Logged events:** {len(st.session_state.log_df)}
"""
            )

    cap.release()
else:
    st.info("Click **Start** in the sidebar to begin streaming.")

# EOF