# Age and Gender Detection System

A real-time computer vision application built with **Python** and **OpenCV Deep Neural Networks (DNN)** to detect faces and predict **gender** (`Male` / `Female`) and **age brackets** from images or live webcam video.

---

## Tech Stack

- **Language:** Python 3
- **Computer Vision & Inference:** OpenCV (`cv2.dnn`)
- **Pre-trained Models:**
  - Face Detection: TensorFlow SSD (`opencv_face_detector_uint8.pb`, `opencv_face_detector.pbtxt`)
  - Age Classification: Caffe CNN (`age_net.caffemodel`, `age_deploy.prototxt`) trained on the Adience dataset (Tal Hassner & Gil Levi)
  - Gender Classification: Caffe CNN (`gender_net.caffemodel`, `gender_deploy.prototxt`)

---

## Predicted Classes

- **Gender:** `Male`, `Female`
- **Age Ranges:** `(0-2)`, `(4-6)`, `(8-12)`, `(15-20)`, `(25-32)`, `(38-43)`, `(48-53)`, `(60-100)`

---

## Project Structure

```text
age and gender detection/
├── detect.py                          # Main face, age & gender detection script
├── opencv_face_detector.pbtxt         # Face detector network config
├── opencv_face_detector_uint8.pb      # Face detector weights
├── age_deploy.prototxt                # Age CNN network config
├── age_net.caffemodel                 # Age CNN pre-trained weights
├── gender_deploy.prototxt             # Gender CNN network config
├── gender_net.caffemodel              # Gender CNN pre-trained weights
├── example/                           # Sample output screenshots
├── girl1.jpg, girl2.jpg, kid1.jpg, ...# Sample test images
└── README.md
```

---

## Setup & How to Run

### 1. Install Dependencies
```bash
pip install opencv-python
```

### 2. Run on an Image
```bash
python detect.py --image girl1.jpg
```

### 3. Run with Live Webcam
```bash
python detect.py
```

Press **`Esc`** or **`Ctrl + C`** (or close the preview window) to stop.
