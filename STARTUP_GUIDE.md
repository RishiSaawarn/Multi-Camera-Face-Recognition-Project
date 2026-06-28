# Face Recognition System — Startup Guide

This guide explains how to start the distributed face recognition system, where your **Windows laptop** provides the camera feed, the **Raspberry Pi 5** processes the faces, and your **laptop browser** displays the final annotated output.

---
## 💻 Step 0: Allow the firewall access to the camera and the port 8080
run the following command in administrator powershell 
"netsh advfirewall firewall add rule name="LaptopCamStream" dir=in action=allow protocol=TCP localport=8081" --- without quotes


## 💻 Step 1: Start the Camera on your Windows Laptop

The Raspberry Pi needs a webcam feed. We use a Python script on your Windows machine to turn your built-in laptop camera into a network stream.

1. Open your Windows **PowerShell**.
2. Navigate to your project folder:
   ```powershell
   cd d:\PD_LAB\face_recognition_project2
   ```
3. Run the streaming script:
   ```powershell
   # Keep this window completely open and running in the background!
   python laptop_stream.py
   ```
4. Note the **IPv4 Address** printed in the console (e.g., `http://192.168.0.100:8081/video`).

---

## 🍓 Step 2: Start Processing on the Raspberry Pi

Now we tell the Raspberry Pi to pull that video feed, run YOLOv8, and draw the face bounding boxes.

1. Open a new terminal and **SSH** into your Raspberry Pi as you normally do.
2. Go to your project directory and activate the Python environment:
   ```bash
   cd ~/face_recognition_project2
   source pdenv2/bin/activate
   ```
3. *(Optional check)* Configure your Network IP Addresses in `config.py`:
   ```bash
   nano config.py
   ```
   **If deploying on the Raspberry Pi with multiple laptops:**
   Find the `CAMERA_SOURCES` list and insert the IPs printed out by both your laptops in Step 1.
   ```python
   CAMERA_SOURCES = [
       "http://192.168.0.100:8081/video", 
       "http://192.168.0.105:8081/video"
   ]
   ```
   **If testing locally on your single Windows Laptop:**
   Change the IP addresses to `0` (which refers to the builtin Windows webcam). You can even put two `0`s to test the split-screen math!
   ```python
   CAMERA_SOURCES = [0, 0]
   ```
4. Start the main processing script:
   ```bash
   # Keep this running! Press Ctrl+C if you want to stop it later.
   python main.py
   ```

*(Note: The ONNX `GPU device discovery failed` warning is perfectly normal, it just means the RPi is successfully using its CPU).*

---

## 📺 Step 3: Watch the Finished Video

Because the Raspberry Pi processes the frames without a physical screen attached, it creates a second web server to stream the final images back to you.

1. On your Windows laptop, open any web browser (Chrome, Edge, etc.).
2. In the URL bar, type the **Raspberry Pi's IP Address** on port 8080:
   ```
   http://[YOUR-RASPBERRY-PI-IP]:8080/
   ```
3. You will now see the live, real-time video feed with YOLOv8 face detection!

---

## 🛑 How to Shut Down Cleanly
1. On the Raspberry Pi terminal, press **`Ctrl+C`** (or `ESC` if you have local GUI window) to exit `main.py`.
2. On your Windows Laptop terminal, press **`Ctrl+C`** to turn off the laptop camera stream.
