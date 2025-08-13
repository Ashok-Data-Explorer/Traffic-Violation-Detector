import streamlit as st
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from datetime import datetime

st.set_page_config(page_title="Traffic Violation Detector", layout="wide")

# Load YOLO model
model = YOLO('best.pt')

# Initialize Session State
if 'log_df' not in st.session_state:
    st.session_state.log_df = pd.DataFrame(columns=["Timestamp", "Vehicle ID", "Action"])
if 'vehicle_count' not in st.session_state:
    st.session_state.vehicle_count = 0
if 'violation_count' not in st.session_state:
    st.session_state.violation_count = 0
if 'running' not in st.session_state:
    st.session_state.running = False
if 'cap' not in st.session_state:
    st.session_state.cap = None

# Sidebar controls
stop_line_y = st.sidebar.slider("Stop Line Y Position", 100, 600, 300)
red_light = st.sidebar.checkbox("Red Light ON")
cap_source = st.sidebar.text_input("Camera Source (0 for webcam or IP URL)", "0")

# Start/Stop buttons
col1, col2 = st.sidebar.columns(2)
if col1.button("Start"):
    st.session_state.running = True
    try:
        cap_source_val = int(cap_source)
    except ValueError:
        cap_source_val = cap_source
    st.session_state.cap = cv2.VideoCapture(cap_source_val)
if col2.button("Stop"):
    st.session_state.running = False
    if st.session_state.cap:
        st.session_state.cap.release()
        st.session_state.cap = None

# Download log
if st.sidebar.button("Download Log as Excel"):
    st.session_state.log_df.to_excel("violation_log.xlsx", index=False)
    st.sidebar.success("Excel file saved as violation_log.xlsx")

frame_window = st.image([])

def process_frame(frame):
    results = model(frame)
    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            vehicle_id = f"{x1}-{y1}-{x2}-{y2}"
            center_y = (y1 + y2) // 2

            if center_y < stop_line_y:
                color = (0, 255, 0)
                action = "Pass"
                st.session_state.vehicle_count += 1
            else:
                color = (0, 0, 255)
                action = "Crossed"
                if red_light:
                    st.session_state.violation_count += 1
                    action = "Violation"

            st.session_state.log_df.loc[len(st.session_state.log_df)] = [datetime.now(), vehicle_id, action]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Draw stop line
    cv2.line(frame, (0, stop_line_y), (frame.shape[1], stop_line_y), (0, 0, 255), 2)

    # Display counts
    cv2.putText(frame, f"Vehicles Passed: {st.session_state.vehicle_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.putText(frame, f"Violations: {st.session_state.violation_count}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    return frame

# Main loop
if st.session_state.running and st.session_state.cap:
    ret, frame = st.session_state.cap.read()
    if not ret:
        st.error("Camera feed not available.")
    else:
        frame = process_frame(frame)
        frame_window.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
else:
    st.info("Click Start to run detection.")
