# Traffic-Violation-Detector

````markdown
# 🚦 Traffic Violation Detector (YOLO + Streamlit)

A real-time traffic monitoring application built with **YOLO object detection** and **Streamlit**, capable of detecting vehicles, identifying their type, and logging **red-light violations** from a **live camera feed** (webcam or mobile IP camera).

---

## ✨ Features
- 📹 **Live Video Feed** from webcam or smartphone IP camera  
- 🚗 **Vehicle Detection & Classification** (Car, Bus, Bike, Truck, etc.)  
- 🆔 **Unique Vehicle Tracking** with `track_id` (no double counting)  
- 🚫 **Red-Light Violation Detection** with adjustable stop line  
- 📊 **Real-Time Counters** (vehicles passed, violations)  
- 📂 **Excel Report Download** of all violations with timestamp, vehicle ID, and type  
- 🎛 **Interactive Controls** via Streamlit sidebar  
- ✅ Works on desktop & mobile browsers  

---

## 🛠 Tech Stack
- **Python 3.8+**
- [Streamlit](https://streamlit.io/) – Web interface  
- [OpenCV](https://opencv.org/) – Video processing  
- [Ultralytics YOLO](https://docs.ultralytics.com/) – Object detection & tracking  
- **Pandas** – Data logging & export  
- **OpenPyXL** – Excel export support  

---

## 📦 Installation

1️⃣ Clone this repository:
```bash
git clone https://github.com/your-username/traffic-violation-detector.git
cd traffic-violation-detector
````

2️⃣ Install dependencies:

```bash
pip install -r requirements.txt
```

3️⃣ Download a YOLO model (e.g., YOLOv8n pre-trained weights or your custom trained model) and place it in the project folder:

```bash
# Example: YOLOv8n from Ultralytics
yolo download yolo8n.pt
```

Rename it to `best.pt` or update the code to your filename.

---

## ▶ Usage

Run the app:

```bash
streamlit run app.py
```

* **For Webcam:**
  In the sidebar, set **Camera Source** to `0`.

* **For Mobile Camera:**

  1. Install an IP camera app (e.g., *IP Webcam* on Android)
  2. Start streaming and copy the `/video` URL (e.g., `http://192.168.x.x:8080/video`)
  3. Paste this into the **IP Stream URL** field in the sidebar.

---

## 📁 Output

The app shows:

* **Real-time traffic feed** with bounding boxes and stop line
* **Excel log** of:

  * Timestamp
  * Vehicle ID
  * Vehicle Type
  * Action (Pass / Violation)

Example Excel export:

| Timestamp           | Vehicle ID | Vehicle Type | Action    |
| ------------------- | ---------- | ------------ | --------- |
| 2025-08-13 10:12:34 | 3          | Car          | Pass      |
| 2025-08-13 10:13:10 | 7          | Bike         | Violation |

---

## 📸 Screenshot

![Demo Screenshot](assets/demo.png)

---

## 📜 License

This project is licensed under the MIT License.

---

## 🙌 Acknowledgements

* [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) for the detection framework
* [Streamlit](https://streamlit.io/) for the easy web deployment
* [OpenCV](https://opencv.org/) for real-time image processing


