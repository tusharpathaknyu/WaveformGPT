# WaveformBuddy Hardware Wiring Guide

## Components List

| Component | Quantity | Purpose |
|-----------|----------|---------|
| ESP32-CAM (AI-Thinker) | 1 | Camera + WiFi brain |
| INMP441 I2S Microphone | 1 | Voice input |
| PAM8302 2.5W Amplifier | 1 | Audio output |
| 3W 4Ω Speaker | 1 | Voice output |
| 0.96" OLED Display | 1 | Status display |
| Tactile Button | 1 | Manual trigger |
| 100nF Capacitors | 3 | Power filtering |
| Breadboard | 1 | Prototyping |
| Dupont Wires | Many | Connections |

---

## Wiring Diagram

```
                            ┌─────────────────────────┐
                            │       ESP32-CAM         │
                            │                         │
                            │  GND ────────┬──────────┼──────────┐
                            │  3.3V ───────┼────┬─────┼────┐     │
                            │  5V ─────────┼────┼─────┼────┼──┐  │
                            │              │    │     │    │  │  │
                            │  GPIO 0 ─────┼────┼─────┼────┼──┼──┼─── Button ─── GND
                            │              │    │     │    │  │  │
                            │  GPIO 21 ────┼────┼─────┼────┼──┼──┼─── OLED SDA
                            │  GPIO 22 ────┼────┼─────┼────┼──┼──┼─── OLED SCL
                            │              │    │     │    │  │  │
                            └──────────────┼────┼─────┼────┼──┼──┼───────────────────┘
                                           │    │     │    │  │  │
┌──────────────────────────────────────────┼────┼─────┼────┼──┼──┼───────────────────┐
│ INMP441 Microphone                       │    │     │    │  │  │                   │
│                                          │    │     │    │  │  │                   │
│   VDD ───────────────────────────────────┼────┘     │    │  │  │                   │
│   GND ───────────────────────────────────┘          │    │  │  │                   │
│   L/R ──────────────────────────────────────────────┴────┘  │  │                   │
│   WS  ──────── GPIO 25 (separate ESP32)                     │  │                   │
│   SD  ──────── GPIO 32 (separate ESP32)                     │  │                   │
│   SCK ──────── GPIO 33 (separate ESP32)                     │  │                   │
└─────────────────────────────────────────────────────────────┼──┼───────────────────┘
                                                              │  │
┌─────────────────────────────────────────────────────────────┼──┼───────────────────┐
│ PAM8302 Amplifier                                           │  │                   │
│                                                             │  │                   │
│   VIN ──────────────────────────────────────────────────────┘  │                   │
│   GND ─────────────────────────────────────────────────────────┘                   │
│   A+  ──────── GPIO 26 (I2S data from separate ESP32)                              │
│   A-  ──────── GND                                                                 │
│   +   ──────── Speaker +                                                           │
│   -   ──────── Speaker -                                                           │
└────────────────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────────────────┐
│ 0.96" OLED Display (SSD1306)                                                       │
│                                                                                    │
│   VCC ──────── 3.3V                                                                │
│   GND ──────── GND                                                                 │
│   SDA ──────── GPIO 21                                                             │
│   SCL ──────── GPIO 22                                                             │
└────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Important Notes

### ESP32-CAM Limitation
The ESP32-CAM uses most GPIO pins for the camera. For audio I2S, you have two options:

**Option A: Two ESP32 Setup (Recommended)**
- ESP32-CAM: Camera only, sends images via WiFi
- Regular ESP32: Audio I/O (mic + speaker)
- Both connect to same WiFi, talk to Python server

**Option B: Single ESP32-CAM (Limited)**
- Use GPIO 12, 13, 14 for I2S (these are available after camera init)
- May have conflicts, less reliable

### Power Considerations
```
Power Budget:
- ESP32-CAM: 160-260mA (with camera active)
- INMP441: 1.4mA
- PAM8302: Up to 500mA at full volume
- OLED: 20mA
- Total: ~800mA peak

Recommended Power:
- 5V 2A USB power supply
- OR 3.7V LiPo with boost converter
```

### Capacitor Placement
Add 100nF capacitors:
1. Between VCC and GND on INMP441
2. Between VIN and GND on PAM8302  
3. Between 3.3V and GND on ESP32

---

## Two-ESP32 Architecture (Recommended)

```
┌─────────────────┐                    ┌─────────────────┐
│   ESP32-CAM     │                    │    ESP32        │
│   (Camera)      │                    │   (Audio)       │
│                 │                    │                 │
│ [Camera Module] │                    │ [INMP441 Mic]   │
│       ↓         │                    │       ↓         │
│   Capture       │                    │   Record        │
│       ↓         │                    │       ↓         │
│   WiFi POST ────┼─────────┬──────────┼── WiFi POST     │
└─────────────────┘         │          │       ↑         │
                            │          │   Play          │
                            ↓          │       ↑         │
                    ┌───────────────┐  │ [PAM8302 Amp]   │
                    │   Mac/PC      │  │ [Speaker]       │
                    │   Python      │  └─────────────────┘
                    │   Server      │
                    │               │
                    │ WaveformBuddy │
                    │       ↓       │
                    │  OpenAI API   │
                    └───────────────┘
```

---

## Enclosure Ideas

For a handheld gadget:
1. **3D Print** - Custom case with camera window, mic hole, speaker grille
2. **Project Box** - Drill holes for camera, buttons
3. **Altoids Tin** - Classic maker aesthetic

Key features to include:
- Camera window (clear acrylic)
- Microphone hole (small, near mouth position)
- Speaker grille (mesh or holes)
- Button accessible from outside
- USB port for power/programming
- OLED window

---

## Quick Test

1. Power up ESP32-CAM
2. Connect to Serial Monitor (115200 baud)
3. Watch for WiFi connection
4. Note the IP address shown
5. Update `SERVER_URL` in code with your Mac's IP
6. Run Python server: `python -m waveformgpt.buddy`
7. Press button on ESP32 - should capture and send!

Commands via Serial:
- `c` - Capture circuit
- `w` - Capture waveform  
- `v` - Voice record
- `r` - Restart ESP32
