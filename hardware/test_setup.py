#!/usr/bin/env python3
"""Test script to verify WaveformBuddy setup"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from waveformgpt.buddy import WaveformBuddy, ESP32Bridge

# Test initialization
print("Testing WaveformBuddy setup...")
print()

buddy = WaveformBuddy(on_status=lambda s: print(f"[Status] {s}"))
print("✓ WaveformBuddy initialized")

# Test webcam
try:
    import cv2
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        ret, frame = cap.read()
        cap.release()
        if ret:
            print(f"✓ Webcam working - captured {frame.shape[1]}x{frame.shape[0]} image")
        else:
            print("✗ Webcam opened but failed to capture")
    else:
        print("✗ Could not open webcam")
except Exception as e:
    print(f"✗ Webcam error: {e}")

# Show IP
import socket
try:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    ip = s.getsockname()[0]
    s.close()
    print(f"✓ Your Mac IP: {ip}")
except:
    ip = "localhost"
    print(f"✓ Using localhost")

print()
print("=" * 50)
print("Everything ready! When ESP32-CAM arrives:")
print("=" * 50)
print()
print(f'  1. Edit esp32_buddy.ino:')
print(f'     const char* SERVER_URL = "http://{ip}:8080";')
print(f'     const char* WIFI_SSID = "YOUR_WIFI";')
print(f'     const char* WIFI_PASSWORD = "YOUR_PASSWORD";')
print()
print(f'  2. Flash to ESP32-CAM via Arduino IDE')
print()
print(f'  3. Run server:')
print(f'     python hardware/run_server.py')
print()
print(f'  4. Point camera at circuit → press button')
print(f'     Point camera at scope → double-press')
print(f'     Hold button → speak question')
print()
