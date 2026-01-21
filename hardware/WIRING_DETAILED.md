# WaveformBuddy Wiring Guide - Step by Step

## Your Components

From your orders:
- ✅ ESP32 Development Board (from LAFVIN starter kit)
- ✅ INMP441 I2S Microphone (3-pack)
- ✅ PAM8302 2.5W Amplifier
- ✅ 3W 4Ω Mini Speaker
- ✅ 0.96" OLED Display (from LAFVIN kit)
- ✅ 100nF Capacitors
- ✅ Breadboard + Dupont wires
- 📦 ESP32-CAM (arriving soon)

---

## ESP32 Pinout Reference

```
                    ┌─────────────────────┐
                    │      ESP32 Dev      │
                    │       Board         │
                    │                     │
              3.3V ─┤ 3V3           VIN  ├─ 5V
               GND ─┤ GND           GND  ├─ GND
 (Touch)      GP15 ─┤ D15           D13  ├─ GP13
              GP2  ─┤ D2            D12  ├─ GP12
              GP4  ─┤ D4            D14  ├─ GP14
 (RX2)        GP16 ─┤ RX2           D27  ├─ GP27
 (TX2)        GP17 ─┤ TX2           D26  ├─ GP26 ◄── Speaker DAC
              GP5  ─┤ D5            D25  ├─ GP25 ◄── Mic WS
              GP18 ─┤ D18           D33  ├─ GP33 ◄── Mic SD (Data)
              GP19 ─┤ D19           D32  ├─ GP32 ◄── Mic SCK
 (I2C SDA)    GP21 ─┤ D21           D35  ├─ GP35 (Input only)
 (RX0)        GP3  ─┤ RX0           D34  ├─ GP34 (Input only)
 (TX0)        GP1  ─┤ TX0           VN   ├─ GP39 (Input only)
 (I2C SCL)    GP22 ─┤ D22           VP   ├─ GP36 (Input only)
              GP23 ─┤ D23           EN   ├─ Enable
                    │                     │
                    │    [USB Port]       │
                    └─────────────────────┘
```

---

## STEP 1: OLED Display (Easiest - Start Here!)

### Components Needed:
- 0.96" OLED Display (SSD1306, I2C)
- 4x Female-to-Female Dupont wires

### Wiring:

```
   OLED Display                    ESP32
   ┌──────────┐                   ┌──────┐
   │  ┌────┐  │                   │      │
   │  │    │  │                   │      │
   │  │OLED│  │                   │      │
   │  │    │  │                   │      │
   │  └────┘  │                   │      │
   │          │                   │      │
   │ GND VCC  │                   │      │
   │ SCL SDA  │                   │      │
   └──┬──┬────┘                   └──────┘
      │  │  │  │                     │
      │  │  │  └─────────────────────┤ GPIO 21 (SDA)
      │  │  └────────────────────────┤ GPIO 22 (SCL)
      │  └───────────────────────────┤ 3.3V
      └──────────────────────────────┤ GND
```

### Connection Table:

| OLED Pin | Wire Color (suggested) | ESP32 Pin |
|----------|------------------------|-----------|
| GND      | Black                  | GND       |
| VCC      | Red                    | 3.3V      |
| SCL      | Yellow                 | GPIO 22   |
| SDA      | Blue                   | GPIO 21   |

### ⚠️ Important:
- OLED is 3.3V! Don't connect to 5V or it may damage.
- I2C address is usually 0x3C (already set in firmware)

---

## STEP 2: INMP441 I2S Microphone

### Components Needed:
- INMP441 Microphone module
- 6x Female-to-Female Dupont wires
- 1x 100nF capacitor (optional but recommended)

### Wiring:

```
   INMP441 Microphone              ESP32
   ┌──────────────┐               ┌──────┐
   │   ┌────┐     │               │      │
   │   │ O  │ MIC │               │      │
   │   └────┘     │               │      │
   │              │               │      │
   │ L/R SCK WS   │               │      │
   │ GND SD  VDD  │               │      │
   └──┬──┬──┬──┬──┘               └──────┘
      │  │  │  │  │  │               │
      │  │  │  │  │  └───────────────┤ 3.3V
      │  │  │  │  └──────────────────┤ GPIO 33 (SD/Data)
      │  │  │  └─────────────────────┤ GPIO 25 (WS/Word Select)
      │  │  └────────────────────────┤ GPIO 32 (SCK/Clock)
      │  └───────────────────────────┤ GND
      └──────────────────────────────┤ GND (for Left channel)
```

### Connection Table:

| INMP441 Pin | Function      | Wire Color | ESP32 Pin   |
|-------------|---------------|------------|-------------|
| VDD         | Power         | Red        | 3.3V        |
| GND         | Ground        | Black      | GND         |
| SD          | Serial Data   | Green      | GPIO 33     |
| WS          | Word Select   | Blue       | GPIO 25     |
| SCK         | Serial Clock  | Yellow     | GPIO 32     |
| L/R         | Channel Sel   | Black      | GND (Left)  |

### ⚠️ Important:
- L/R pin MUST be connected to GND for left channel
- If L/R is floating or HIGH, you may get no audio!
- Add 100nF capacitor between VDD and GND for noise filtering

### Capacitor Placement:
```
       100nF
    ┌───┤├───┐
    │        │
   VDD      GND
   (on INMP441)
```

---

## STEP 3: PAM8302 Amplifier + Speaker

### Components Needed:
- PAM8302 Amplifier module
- 3W 4Ω Speaker
- 4x Dupont wires (for amplifier)
- Speaker already has wires attached

### Wiring:

```
   PAM8302 Amplifier                    ESP32
   ┌────────────────┐                  ┌──────┐
   │    ┌─────┐     │                  │      │
   │    │ IC  │     │                  │      │
   │    └─────┘     │                  │      │
   │                │                  │      │
   │ VIN GND  A+ A- │                  │      │
   │ SD  +   -      │                  │      │
   └─┬───┬───┬───┬──┘                  └──────┘
     │   │   │   │                        │
     │   │   │   └────────────────────────┤ GND
     │   │   └────────────────────────────┤ GPIO 26
     │   └────────────────────────────────┤ GND
     └────────────────────────────────────┤ 5V (VIN on ESP32)


   Speaker Connection:
   ┌────────────────┐
   │ PAM8302        │      ┌─────────┐
   │            +  ─┼──────┤  (+)    │
   │            -  ─┼──────┤  (-)    │
   │                │      │ Speaker │
   └────────────────┘      └─────────┘
```

### Connection Table:

| PAM8302 Pin | Function       | Wire Color | Connection    |
|-------------|----------------|------------|---------------|
| VIN         | Power (5V)     | Red        | 5V (VIN)      |
| GND         | Ground         | Black      | GND           |
| A+          | Audio Input +  | White      | GPIO 26       |
| A-          | Audio Input -  | Black      | GND           |
| +           | Speaker +      | (to spkr)  | Speaker +     |
| -           | Speaker -      | (to spkr)  | Speaker -     |
| SD          | Shutdown       | -          | Leave floating (or 3.3V for always-on) |

### ⚠️ Important:
- PAM8302 needs 5V for proper volume!
- SD pin can be left unconnected (internally pulled high)
- If no sound, check SD pin is not accidentally grounded

---

## STEP 4: Add Decoupling Capacitors (Recommended)

For stable operation, add 100nF capacitors:

```
Breadboard Layout with Capacitors:

     + Rail (3.3V) ─────────────────────────────────────
                        │          │
                       ═══        ═══  ← 100nF caps
                        │          │
     - Rail (GND) ──────┴──────────┴────────────────────
                        ↑          ↑
                     INMP441    OLED
```

---

## COMPLETE BREADBOARD LAYOUT

```
                         Breadboard Top View
    ┌─────────────────────────────────────────────────────────────────┐
    │  + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + +  │ ← 3.3V Rail
    │  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  │ ← GND Rail
    │                                                                 │
    │  ┌─────────────────────────────────┐                           │
    │  │                                 │                           │
    │  │           ESP32 Board           │                           │
    │  │                                 │                           │
    │  │  3V3  GND  D21  D22  ...  D25  D32  D33  D26  VIN  GND     │
    │  └──┬────┬────┬────┬─────────┬────┬────┬────┬────┬────┬───────┘│
    │     │    │    │    │         │    │    │    │    │    │        │
    │     │    │    │    │         │    │    │    │    │    │        │
    │     │    │    │    │    ┌────┘    │    │    │    │    │        │
    │     │    │    │    │    │    ┌────┘    │    │    │    │        │
    │     │    │    │    │    │    │    ┌────┘    │    │    │        │
    │     │    │    │    │    │    │    │         │    │    │        │
    │  ┌──┴────┴────┴────┴────┴────┴────┴────┐    │    │    │        │
    │  │   INMP441 Microphone                │    │    │    │        │
    │  │  VDD GND  L/R SCK  WS  SD           │    │    │    │        │
    │  └─────────────────────────────────────┘    │    │    │        │
    │     │    │                                  │    │    │        │
    │  ┌──┴────┴────┐                             │    │    │        │
    │  │   OLED     │                             │    │    │        │
    │  │ VCC GND    │                             │    │    │        │
    │  │ SCL SDA    │ ←────── GPIO 21, 22         │    │    │        │
    │  └────────────┘                             │    │    │        │
    │                                             │    │    │        │
    │  ┌──────────────────────────────────────────┴────┴────┴───┐    │
    │  │            PAM8302 Amplifier                           │    │
    │  │           VIN  GND   A+   A-    +    -                 │    │
    │  └───────────────────────────────────┬────┬───────────────┘    │
    │                                      │    │                    │
    │                                   ┌──┴────┴──┐                 │
    │                                   │  Speaker │                 │
    │                                   │   🔊     │                 │
    │                                   └──────────┘                 │
    │  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - │ ← GND Rail
    │  + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + │ ← 5V Rail
    └─────────────────────────────────────────────────────────────────┘
```

---

## WIRING CHECKLIST

Before powering on, verify each connection:

### OLED Display
- [ ] GND → ESP32 GND
- [ ] VCC → ESP32 3.3V
- [ ] SCL → ESP32 GPIO 22
- [ ] SDA → ESP32 GPIO 21

### INMP441 Microphone  
- [ ] VDD → ESP32 3.3V
- [ ] GND → ESP32 GND
- [ ] L/R → ESP32 GND (IMPORTANT!)
- [ ] SCK → ESP32 GPIO 32
- [ ] WS → ESP32 GPIO 25
- [ ] SD → ESP32 GPIO 33
- [ ] 100nF cap between VDD and GND (optional)

### PAM8302 Amplifier
- [ ] VIN → ESP32 5V (VIN)
- [ ] GND → ESP32 GND
- [ ] A+ → ESP32 GPIO 26
- [ ] A- → ESP32 GND
- [ ] + → Speaker positive
- [ ] - → Speaker negative

---

## POWER CONSIDERATIONS

```
Power Flow:
                          USB Cable (5V, 500mA typical)
                                │
                                ▼
                          ┌──────────┐
                          │  ESP32   │
                          │          │
               3.3V (from │   LDO    │ 5V (pass-through)
               internal   │          │
               regulator) │          │
                          └──────────┘
                           │        │
                           ▼        ▼
                    ┌──────────┐  ┌─────────┐
                    │  OLED    │  │ PAM8302 │
                    │ INMP441  │  │  (5V)   │
                    │  (3.3V)  │  └─────────┘
                    └──────────┘

Total Power Budget:
  - ESP32:     ~80mA typical, 240mA peak (WiFi)
  - OLED:      ~20mA
  - INMP441:   ~1.5mA
  - PAM8302:   ~5mA idle, up to 500mA at full volume
  
  Total: ~350mA typical, ~750mA peak
  
  Recommendation: Use good USB cable and powered USB hub
                  if experiencing brownouts
```

---

## TESTING SEQUENCE

After wiring, follow this test sequence:

### 1. Flash the Firmware
```bash
# In Arduino IDE:
# 1. Open: hardware/esp32_audio/esp32_audio.ino
# 2. Set Board: ESP32 Dev Module
# 3. Set Port: /dev/cu.usbserial-XXXX (your ESP32)
# 4. Upload
```

### 2. Open Serial Monitor (115200 baud)

### 3. Test Each Component:
```
Type in Serial Monitor:

d    → Test OLED display (shows patterns)
m    → Test microphone (speak loudly, check amplitude)
w    → Test WiFi connection
h    → Show help
```

### 4. Expected Output:
```
=== WaveformBuddy Audio Module ===

SSD1306 OLED initialized ✓
I2S Microphone initialized ✓
Connecting to WiFi...
Connected! IP: 192.168.1.xxx ✓

=== Ready ===
Hold BOOT button to speak
```

---

## TROUBLESHOOTING

| Problem | Possible Cause | Solution |
|---------|---------------|----------|
| OLED blank | Wrong I2C address | Try 0x3D instead of 0x3C |
| OLED blank | SDA/SCL swapped | Swap GPIO 21 ↔ 22 |
| No mic audio | L/R pin floating | Connect L/R to GND |
| Mic very quiet | SCK/WS swapped | Swap GPIO 32 ↔ 25 |
| No speaker sound | SD pin grounded | Leave SD floating or connect to 3.3V |
| Speaker distorted | Not enough power | Use 5V, not 3.3V for PAM8302 |
| WiFi fails | Wrong credentials | Check SSID/password in code |
| Brownouts/resets | Power insufficient | Use powered USB hub |

---

## NEXT STEPS

After successful testing:

1. ✅ Wire complete and tested
2. 📦 Wait for ESP32-CAM delivery  
3. 🔧 Flash camera firmware
4. 🎯 Point at circuit/scope and debug!

**Your server is ready at:** `http://192.168.1.204:8080`

Update your ESP32 firmware:
```cpp
const char* WIFI_SSID = "YOUR_WIFI_NAME";
const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";  
const char* SERVER_URL = "http://192.168.1.204:8080";
```
