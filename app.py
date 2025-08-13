# app.py — Streamlit + streamlit-webrtc Traffic Violation Detector
import io
from datetime import datetime
from collections import defaultdict

import av
import cv2 as cv
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
from ultralytics import YOLO

# -------------------- Page config --------------------
st.set_page_config(page_title="Traffic Violation Detector (webrtc)", layout="wide")

# -------------------- Load model (cached) --------------------
@st.cache_resource(show_spinner=True)
def load_yolo(weights="best.pt"):
    return YOLO(weights)

# Do NOT call load_yolo here globally if you prefer lazy load inside transformer.
# We'll call it inside the transformer constructor so caching still applies.

# -------------------- Sidebar controls --------------------
st.sidebar.header("Settings")

conf = st.sidebar.slider("YOLO confidence", 0.1, 0.9, 0.35, 0.05)
iou = st.sidebar.slider("NMS IoU", 0.2, 0.9, 0.5, 0.05)
resize_w = st.sidebar.slider("Resize width (px)", 480, 1280, 640, 2)

st.sidebar.subheader("Red-light logic")
red_light_mode = st.sidebar.radio("Mode", ["Manual Toggle", "Always Red", "Always Green"], index=0)
manual_red = st.sidebar.checkbox("Red ON (manual)", value=False)

st.sidebar.subheader("Stop line")
line_pos_pct = st.sidebar.slider("Stop line Y (% of height)", 10, 90, 65)

# Start/Stop (webrtc streaming handles start when you press 'Start Streaming' in the widget)
st.sidebar.caption("Open Streamlit page on mobile and allow camera access to stream.")

# -------------------- Helper functions --------------------
def red_is_on() -> bool:
    if red_light_mode == "Always Red":
        return True
    if red_light_mode == "Always Green":
        return False
    return manual_red

def to_excel_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Violations")
    buf.seek(0)
    return buf.getvalue()

# -------------------- Video transformer (webrtc) --------------------
class TrafficTransformer(VideoTransformerBase):
    def __init__(self):
        # Load model (cached)
        self.model = load_yolo("best.pt")
        self.class_names = self.model.names
        # Vehicle classes to track (COCO indices). Adjust if your YOLO uses different mapping.
        self.vehicle_class_ids = {1, 2, 3, 5, 7}  # bicycle, car, motorcycle, bus, truck

        # Internal session state (transformer-local, thread-safe)
        self.log_df = pd.DataFrame(columns=[
            "Timestamp", "Track ID", "Vehicle Type", "Action", "x1", "y1", "x2", "y2"
        ])
        self.crossed_ids = set()
        self.obj_y_hist = defaultdict(list)
        self.passes = 0
        self.violations = 0

    def transform(self, frame: av.VideoFrame) -> av.VideoFrame:
        # Convert incoming frame to numpy BGR image
        img = frame.to_ndarray(format="bgr24")

        # Resize for performance while keeping aspect
        h0, w0 = img.shape[:2]
        new_w = int(resize_w)
        new_h = int(h0 * (new_w / w0))
        frame_rs = cv.resize(img, (new_w, new_h))

        # run tracking/detection (Ultralytics track)
        try:
            results = self.model.track(frame_rs, persist=True, conf=conf, iou=iou, classes=list(self.vehicle_class_ids))
        except Exception as e:
            # Return original frame if model fails
            cv.putText(frame_rs, f"Inference error: {e}", (10,30), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
            return av.VideoFrame.from_ndarray(frame_rs, format="bgr24")

        annotated = results[0].plot()  # annotated BGR image of frame_rs

        # compute stop line position in pixels
        line_y = int(new_h * (line_pos_pct / 100.0))
        cv.line(annotated, (10, line_y), (annotated.shape[1] - 10, line_y), (0, 0, 255), 2)

        # Process tracks and detect first-time crossings
        boxes = results[0].boxes
        if boxes.id is not None:
            for box in boxes:
                # safe extraction
                try:
                    tid = int(box.id)
                except Exception:
                    continue

                # bounding box coords
                xyxy = box.xyxy[0].cpu().numpy() if hasattr(box.xyxy, "cpu") else box.xyxy[0].numpy()
                x1, y1, x2, y2 = map(int, xyxy)
                # class id (if present)
                cls_id = int(box.cls[0]) if hasattr(box, "cls") else -1
                vtype = self.class_names.get(cls_id, str(cls_id))

                bottom_y = y2
                self.obj_y_hist[tid].append(bottom_y)
                hist = self.obj_y_hist[tid]
                if len(hist) >= 2:
                    prev_y, curr_y = hist[-2], hist[-1]
                    if (prev_y < line_y <= curr_y) and (tid not in self.crossed_ids):
                        action = "Violation" if red_is_on() else "Pass"
                        if action == "Violation":
                            self.violations += 1
                        else:
                            self.passes += 1
                        self.crossed_ids.add(tid)
                        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        # log event
                        self.log_df.loc[len(self.log_df)] = [ts, tid, vtype, action, x1, y1, x2, y2]
                        # flash highlight
                        color = (0, 0, 255) if action == "Violation" else (0, 255, 0)
                        cv.rectangle(annotated, (x1, y1), (x2, y2), color, 4)
                        cv.putText(annotated, f"{action}!", (x1, max(0, y1 - 10)),
                                   cv.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                # draw label & bbox for all boxes
                label = f"{vtype}"
                cv.rectangle(annotated, (x1, y1), (x2, y2), (255, 200, 0), 2)
                cv.putText(annotated, label, (x1, y1 - 6), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 2)

        # overlays: traffic light + counts
        # draw traffic light
        h, w = annotated.shape[:2]
        draw_x1, draw_y1, draw_x2, draw_y2 = w - 90, 10, w - 20, 120
        cv.rectangle(annotated, (draw_x1, draw_y1), (draw_x2, draw_y2), (50,50,50), -1)
        cv.circle(annotated, (w - 55, 40), 15, (0,0,255) if red_is_on() else (0,0,60), -1)
        cv.circle(annotated, (w - 55, 90), 15, (0,255,0) if not red_is_on() else (0,60,0), -1)

        cv.rectangle(annotated, (5, 4), (360, 58), (255, 255, 255), -1)
        cv.rectangle(annotated, (5, 4), (360, 58), (0, 0, 0), 2)
        cv.putText(annotated, f"Passes: {self.passes}", (15, 28), cv.FONT_HERSHEY_SIMPLEX, 0.65, (0,128,0), 2)
        cv.putText(annotated, f"Violations: {self.violations}", (15, 50), cv.FONT_HERSHEY_SIMPLEX, 0.65, (0,0,200), 2)

        # convert back to av.VideoFrame (bgr24)
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")

# -------------------- WebRTC configuration (optional STUN/TURN) --------------------
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# -------------------- Layout / placeholders --------------------
col_stream, col_stats = st.columns([3, 1])
with col_stream:
    st.header("Camera Stream")
with col_stats:
    st.header("Session Stats")

# -------------------- Start webrtc streamer --------------------
ctx = webrtc_streamer(
    key="traffic-violation",
    video_transformer_factory=TrafficTransformer,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    async_transform=True,
    client_settings={"userVisible": True},
)

# -------------------- Stats & download panel --------------------
with col_stats:
    # Access transformer instance to fetch logs/stats (only available after streaming starts)
    if ctx and ctx.video_transformer:
        tr = ctx.video_transformer
        st.markdown(f"**Passes:** {tr.passes}  ")
        st.markdown(f"**Violations:** {tr.violations}  ")
        st.markdown(f"**Crossed IDs:** {len(tr.crossed_ids)}  ")
        st.markdown(f"**Logged events:** {len(tr.log_df)}  ")

        # show a small table preview
        if not tr.log_df.empty:
            st.dataframe(tr.log_df.tail(10))

        # Excel download button (in-memory bytes)
        excel_bytes = to_excel_bytes(tr.log_df)
        st.download_button(
            label="📥 Download Violation Log (Excel)",
            data=excel_bytes,
            file_name="violation_log.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    else:
        st.info("Start the stream (allow camera access) to see live stats and logs.")

# -------------------- Footer tips --------------------
st.caption("Tip: open this Streamlit app on your phone browser, allow camera access when prompted, then start streaming. For better FPS reduce Resize width.")

# EOF
