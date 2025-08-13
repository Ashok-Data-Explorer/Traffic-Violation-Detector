import io
from datetime import datetime

import av
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
from ultralytics import YOLO

# -------------------- Configuration --------------------
MODEL_PATH = "best.pt"           
OUTPUT_EXCEL = "violations.xlsx"

st.set_page_config(page_title="Traffic Violation Detector (YOLO Only)", layout="wide")
st.title("🚦 Traffic Violation Detector (YOLO Only)")

# -------------------- Sidebar controls --------------------
st.sidebar.header("Settings")
conf = st.sidebar.slider("YOLO confidence", 0.1, 0.9, 0.35, 0.05)
iou_thres = st.sidebar.slider("NMS IoU", 0.2, 0.9, 0.45, 0.05)
resize_w = st.sidebar.slider("Resize width (px)", 480, 1280, 640, 2)
line_pos_pct = st.sidebar.slider("Stop line Y (% of height)", 40, 80, 65)

st.sidebar.subheader("Red-light (for violation flagging)")
red_light_mode = st.sidebar.radio("Mode", ["Manual Toggle", "Always Red", "Always Green"], index=0)
manual_red = st.sidebar.checkbox("Red ON (manual)", value=False)

def red_is_on() -> bool:
    if red_light_mode == "Always Red":
        return True
    if red_light_mode == "Always Green":
        return False
    return manual_red

# -------------------- Helper functions --------------------
def to_excel_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Violations")
    buf.seek(0)
    return buf.getvalue()

# -------------------- Transformer --------------------
class YoloTransformer(VideoTransformerBase):
    def __init__(self):
        self.model = YOLO(MODEL_PATH)
        self.class_names = self.model.names
        self.log = []
        self.passes = 0
        self.violations = 0

    def transform(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        h0, w0 = img.shape[:2]
        new_w = int(resize_w)
        new_h = int(h0 * (new_w / w0))
        frame_rs = cv2.resize(img, (new_w, new_h))

        # YOLO detection
        try:
            results = self.model.predict(frame_rs, conf=conf, iou=iou_thres, classes=None, verbose=False)
        except Exception as e:
            cv2.putText(frame_rs, f"Model error: {e}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
            return av.VideoFrame.from_ndarray(frame_rs, format="bgr24")

        annotated = frame_rs.copy()
        line_y = int(new_h * (line_pos_pct / 100.0))
        cv2.line(annotated, (0, line_y), (annotated.shape[1], line_y), (0,0,255), 2)

        for res in results:
            for box in res.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                score = float(box.conf[0]) if hasattr(box, "conf") else 1.0
                cls_id = int(box.cls[0]) if hasattr(box, "cls") else -1
                if score < conf:
                    continue
                vtype = self.class_names.get(cls_id, str(cls_id))
                bottom_y = y2

                # Check line crossing for violation
                if bottom_y >= line_y:
                    action = "Violation" if red_is_on() else "Pass"
                    if action == "Violation":
                        self.violations += 1
                        color = (0,0,255)
                    else:
                        self.passes += 1
                        color = (0,255,0)
                    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    self.log.append({
                        "Timestamp": ts,
                        "VehicleType": vtype,
                        "Action": action,
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2
                    })
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
                    cv2.putText(annotated, f"{action}!", (x1, max(0, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                else:
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (255,200,0), 2)
                    cv2.putText(annotated, vtype, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,200,0), 2)

        # Overlays: traffic light and counters
        h, w = annotated.shape[:2]
        cv2.rectangle(annotated, (5,4), (300,60), (255,255,255), -1)
        cv2.rectangle(annotated, (5,4), (300,60), (0,0,0), 2)
        cv2.putText(annotated, f"Passes: {self.passes}", (15,28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,128,0), 2)
        cv2.putText(annotated, f"Violations: {self.violations}", (15,50), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,0,200), 2)

        cv2.circle(annotated, (w-55,40), 15, (0,0,255) if red_is_on() else (0,0,60), -1)
        cv2.circle(annotated, (w-55,90), 15, (0,255,0) if not red_is_on() else (0,60,0), -1)

        return av.VideoFrame.from_ndarray(annotated, format="bgr24")

# -------------------- RTC config --------------------
RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

# -------------------- Layout --------------------
col_stream, col_stats = st.columns([3,1])
with col_stream:
    st.header("Camera Stream")
with col_stats:
    st.header("Session Stats")

ctx = webrtc_streamer(
    key="yolo-only",
    video_transformer_factory=YoloTransformer,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    async_transform=True,
)

# -------------------- Controls: Save & Stop --------------------
with col_stats:
    st.markdown("### Controls")
    if ctx and ctx.video_transformer:
        tr = ctx.video_transformer
        st.markdown(f"**Passes:** {tr.passes}  \n**Violations:** {tr.violations}  \n**Logged events:** {len(tr.log)}")
        if len(tr.log) > 0:
            st.dataframe(pd.DataFrame(tr.log).tail(8))

        if st.button("💾 Save & Stop (write Excel once)"):
            df_out = pd.DataFrame(tr.log, columns=["Timestamp","VehicleType","Action","x1","y1","x2","y2"])
            df_out.to_excel(OUTPUT_EXCEL, index=False)
            st.success(f"Saved {len(df_out)} events to {OUTPUT_EXCEL}")
            try:
                ctx.stop()
            except Exception:
                pass
            with open(OUTPUT_EXCEL, "rb") as f:
                st.download_button("📥 Download saved Excel", f, file_name=OUTPUT_EXCEL, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    else:
        st.info("Start the stream (allow camera access) to see live stats and logs.")

st.caption("Tip: open on your phone, allow camera permission, then start streaming.")
