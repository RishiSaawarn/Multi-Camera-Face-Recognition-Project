# Multi-Camera Face Recognition Project

A distributed real-time computer vision pipeline for **face detection**, **face recognition**, and **anti-spoofing** using live video streams. The system is designed to process multiple camera feeds concurrently and supports both local visualization and remote MJPEG streaming.

---

## Project Overview

The primary experiment behind this project was to **simultaneously process live video feeds from two laptop cameras**, effectively creating a wider field of view while performing real-time:

- 🎯 Face Detection
- 👤 Face Recognition
- 🛡️ Face Anti-Spoofing

The system performs inference on both streams in parallel and merges the processed output for local display or browser-based remote viewing.

---

## Features

- 📹 Multi-camera live video acquisition
- ⚡ Parallel processing of two live camera streams
- 🤖 YOLOv8 face detection accelerated with ONNX Runtime
- 🧠 Real-time face recognition using a gallery of known identities
- 🛡️ Integrated anti-spoofing pipeline
- 🌐 MJPEG streaming for remote monitoring
- 🔌 Modular architecture for adding future vision modules

---

## Tech Stack

- Python
- OpenCV
- ONNX Runtime
- YOLOv8 Face Detection
- NumPy
- MJPEG Streaming

---

## Repository Structure

```
.
├── main.py                 # Main inference pipeline
├── laptop_stream.py        # Camera streaming server
├── download_models.py      # Downloads required models
├── requirements.txt
├── models/
├── dataset/
└── ...
```

---

## Getting Started(Go through the Startup Guide for better instructions) -

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download pretrained models

```bash
python download_models.py
```

### 3. Allow the camera stream port (Windows only)

```powershell
netsh advfirewall firewall add rule name="LaptopCamStream" dir=in action=allow protocol=TCP localport=8081
```

### 4. Start the camera stream

Run on each laptop acting as a camera source.

```bash
python laptop_stream.py
```

### 5. Launch the inference pipeline

```bash
python main.py
```

### 6. View the output

Open your browser and navigate to

```
http://<Host-IP>:8080
```

where `<Host-IP>` is the Raspberry Pi or host machine running the processing pipeline.

---

## Current Pipeline

```text
Laptop Camera 1 ─┐
                 ├──► Face Detection (YOLOv8)
Laptop Camera 2 ─┘
                        │
                        ▼
                Face Recognition
                        │
                        ▼
                  Anti-Spoofing
                        │
                        ▼
        Local Display / MJPEG Stream
```

---

## Future Work

🚧 **Depth Estimation Module**

A depth estimation pipeline has already been implemented and evaluated. While the preliminary results are promising, the module is currently not stable enough for deployment within the main pipeline.

Once its performance and robustness improve, it will be integrated into this repository as an additional anti-spoofing and scene understanding component.

Potential future improvements include:

- Monocular depth estimation
- Multi-view depth fusion
- Improved liveness verification
- Face embedding database optimization

---

## Notes

- The core experiment focused on **simultaneously processing two live laptop camera feeds** to increase the effective field of view.
- The system performs **real-time face detection, recognition, and anti-spoofing** on both streams concurrently.
- The architecture is intentionally modular, allowing additional computer vision components to be integrated with minimal changes.

---

## License

This project is intended for educational and research purposes.
