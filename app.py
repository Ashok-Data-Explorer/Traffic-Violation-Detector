# app_streamlit_live_camera_stats.py
import io
from datetime import datetime
import time
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from ultralytics import YOLO

# -------------------- Configuration --------------------
MODEL_PATH = "best.pt"
OUTPUT_EXCEL = "violations.xlsx"

st.set_page_config(page_title="Traffic Violation Detector (YOLO Live Camera)", layout="wide")
st.title("🚦 Traffic Violation Detector (YOLO Live Camera)")

# -------------------- Sidebar --------------------
st.sidebar.header("Settings")
conf = st.sidebar.slider("YOLO confidence", 0.1, 0.9, 0.35, 0.05)
iou_thres = st.sidebar.slider("NMS IoU", 0.2, 0.9, 0.45, 0.05)
resize_w = st.sidebar.slider("Resize width (px)", 480, 1280, 640, 2)
line_pos_pct = st.sidebar.slider("Stop line Y (% of height)", 40, 80, 65)
fps_limit = st.sidebar.slider("Max FPS", 1, 30, 10, 1)

st.sidebar.subheader("Red-light (for violation flagging)")
red_light_mode = st.sidebar.radio("Mode", ["Manual Toggle", "Always Red", "Always Green"], index=0)
manual_red = st.sidebar.checkbox("Red ON (manual)", value=False)

def red_is_on() -> bool:
    if red_light_mode == "Always Red":
        return True
    if red_light_mode == "Always Green":
        return False
    return manual_red

# -------------------- Helper --------------------
def to_excel_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Violations")
    buf.seek(0)
    return buf.getvalue()

# -------------------- Load model --------------------
model = YOLO(MODEL_PATH)
class_names = model.names

# -------------------- Logs & stats --------------------
log = []
stats = {"passes": 0, "violations": 0}

# -------------------- Camera input --------------------
st.header("Live Camera Feed with FPS Limit")
st.caption("Press 'Stop' to end the stream.")
start_btn = st.button("▶ Start Live Stream")
stop_btn = st.button("⏹ Stop Live Stream")

frame_placeholder = st.empty()
stframe = frame_placeholder

if start_btn:
    cap = cv2.VideoCapture(0)  # default camera
    prev_time = 0
    while cap.isOpened():
        # Limit FPS
        elapsed = time.time() - prev_time
        if elapsed < 1.0 / fps_limit:
            time.sleep((1.0 / fps_limit) - elapsed)
        prev_time = time.time()

        ret, frame = cap.read()
        if not ret:
            st.warning("Cannot read from camera.")
            break

        h0, w0 = frame.shape[:2]
        new_w = int(resize_w)
        new_h = int(h0 * (new_w / w0))
        frame_rs = cv2.resize(frame, (new_w, new_h))

        # YOLO detection
        results = model.predict(frame_rs, conf=conf, iou=iou_thres, classes=None, verbose=False)
        annotated = frame_rs.copy()
        line_y = int(new_h * (line_pos_pct / 100.0))
        cv2.line(annotated, (0, line_y), (annotated.shape[1], line_y), (0,0,255), 2)

        # Process detections
        for res in results:
            for box in res.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                score = float(box.conf[0]) if hasattr(box, "conf") else 1.0
                cls_id = int(box.cls[0]) if hasattr(box, "cls") else -1
                if score < conf:
                    continue
                vtype = class_names.get(cls_id, str(cls_id))
                bottom_y = y2

                if bottom_y >= line_y:
                    action = "Violation" if red_is_on() else "Pass"
                    if action == "Violation":
                        stats["violations"] += 1
                        color = (0,0,255)
                    else:
                        stats["passes"] += 1
                        color = (0,255,0)
                    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    log.append({
                        "Timestamp": ts,
                        "VehicleType": vtype,
                        "Action": action,
                        "x1": x1, "y1": y1,
                        "x2": x2, "y2": y2
                    })
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
                    cv2.putText(annotated, f"{action}!", (x1, max(0, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                else:
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (255,200,0), 2)
                    cv2.putText(annotated, vtype, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,200,0), 2)

        # Overlay stats
        cv2.rectangle(annotated, (5,4), (300,60), (255,255,255), -1)
        cv2.rectangle(annotated, (5,4), (300,60), (0,0,0), 2)
        cv2.putText(annotated, f"Passes: {stats['passes']}", (15,28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,128,0), 2)
        cv2.putText(annotated, f"Violations: {stats['violations']}", (15,50), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,0,200), 2)

        # Show frame
        stframe.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB")

        if stop_btn:
            break

    cap.release()
    cv2.destroyAllWindows()

# -------------------- Save log --------------------
if log:
    st.markdown("### Logged Events")
    st.dataframe(pd.DataFrame(log).tail(8))
    if st.button("💾 Save Log to Excel"):
        df_out = pd.DataFrame(log, columns=["Timestamp","VehicleType","Action","x1","y1","x2","y2"])
        df_out.to_excel(OUTPUT_EXCEL, index=False)
        st.success(f"Saved {len(df_out)} events to {OUTPUT_EXCEL}")
        with open(OUTPUT_EXCEL, "rb") as f:
            st.download_button("📥 Download Excel", f, file_name=OUTPUT_EXCEL, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
