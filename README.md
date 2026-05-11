# 🚶 Real-Time Crowd Monitoring Using YOLOv8

A real-time crowd monitoring system built with YOLOv8 that detects and counts people in video streams or live camera feeds.

---

## 📌 Features

- Real-time people detection using YOLOv8
- Live crowd count display
- Works with webcam or video file input
- Lightweight and fast inference

---

## 🛠️ Tech Stack

- **Python**
- **YOLOv8** (Ultralytics)
- **OpenCV**
- **NumPy**

---

## ⚙️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/real-time-crowd-monitoring-yolov8.git
   cd real-time-crowd-monitoring-yolov8
   ```

2. **Install dependencies**
   ```bash
   pip install ultralytics opencv-python numpy
   ```

---

## 🚀 Usage

Run the app:
```bash
python app.py
```

- Press **Q** to quit the window

---

## 📁 Project Structure

```
├── app.py          # Main application file
└── README.md       # Project documentation
```

---

## 📷 How It Works

1. Captures video from webcam or a video file
2. Runs YOLOv8 object detection on each frame
3. Filters detections to count only people (class 0)
4. Displays the live count on the video feed

---

## 🙌 Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [OpenCV](https://opencv.org/)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
