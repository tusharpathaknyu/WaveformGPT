/*
 * WaveformBuddy ESP32 Firmware
 * 
 * Hardware Required:
 * - ESP32-CAM (or ESP32 + OV2640)
 * - INMP441 I2S Microphone
 * - PAM8302 Amplifier + Speaker
 * - 0.96" OLED Display (optional)
 * 
 * Wiring:
 * 
 * INMP441 Mic -> ESP32:
 *   VDD  -> 3.3V
 *   GND  -> GND
 *   SD   -> GPIO 32 (data)
 *   WS   -> GPIO 25 (word select)
 *   SCK  -> GPIO 33 (clock)
 *   L/R  -> GND (left channel)
 * 
 * PAM8302 Amp -> ESP32:
 *   VIN  -> 5V
 *   GND  -> GND
 *   A+   -> GPIO 26 (I2S out)
 *   A-   -> GND
 *   +/-  -> Speaker
 * 
 * OLED Display -> ESP32:
 *   VCC  -> 3.3V
 *   GND  -> GND
 *   SDA  -> GPIO 21
 *   SCL  -> GPIO 22
 * 
 * Button -> ESP32:
 *   One side -> GPIO 0 (boot button)
 *   Other side -> GND
 */

#include <WiFi.h>
#include <HTTPClient.h>
#include <esp_camera.h>
#include <driver/i2s.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

// =============================================================================
// CONFIGURATION - EDIT THESE
// =============================================================================

const char* WIFI_SSID = "YOUR_WIFI_SSID";
const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";
const char* SERVER_URL = "http://YOUR_MAC_IP:8080";  // WaveformBuddy server

// =============================================================================
// PIN DEFINITIONS
// =============================================================================

// ESP32-CAM pins (AI-Thinker module)
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

// I2S Microphone pins (use different ESP32 for audio)
#define I2S_MIC_WS        25
#define I2S_MIC_SD        32
#define I2S_MIC_SCK       33

// I2S Speaker pins
#define I2S_SPK_BCLK      26
#define I2S_SPK_LRC       25
#define I2S_SPK_DOUT      22

// Button
#define BUTTON_PIN        0

// OLED Display
#define SCREEN_WIDTH      128
#define SCREEN_HEIGHT     64
#define OLED_RESET        -1

// =============================================================================
// GLOBALS
// =============================================================================

Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);

enum Mode {
    MODE_IDLE,
    MODE_CIRCUIT,
    MODE_WAVEFORM,
    MODE_LISTENING,
    MODE_WATCHING
};

Mode currentMode = MODE_IDLE;
bool buttonPressed = false;
unsigned long lastButtonTime = 0;

// =============================================================================
// SETUP
// =============================================================================

void setup() {
    Serial.begin(115200);
    Serial.println("WaveformBuddy Starting...");
    
    // Initialize button
    pinMode(BUTTON_PIN, INPUT_PULLUP);
    
    // Initialize display
    if (display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) {
        display.clearDisplay();
        display.setTextSize(1);
        display.setTextColor(SSD1306_WHITE);
        display.setCursor(0, 0);
        display.println("WaveformBuddy");
        display.println("Initializing...");
        display.display();
    }
    
    // Initialize camera
    initCamera();
    
    // Connect to WiFi
    connectWiFi();
    
    // Initialize I2S for mic
    initI2SMic();
    
    showStatus("Ready");
    showInstructions();
}

void initCamera() {
    camera_config_t config;
    config.ledc_channel = LEDC_CHANNEL_0;
    config.ledc_timer = LEDC_TIMER_0;
    config.pin_d0 = Y2_GPIO_NUM;
    config.pin_d1 = Y3_GPIO_NUM;
    config.pin_d2 = Y4_GPIO_NUM;
    config.pin_d3 = Y5_GPIO_NUM;
    config.pin_d4 = Y6_GPIO_NUM;
    config.pin_d5 = Y7_GPIO_NUM;
    config.pin_d6 = Y8_GPIO_NUM;
    config.pin_d7 = Y9_GPIO_NUM;
    config.pin_xclk = XCLK_GPIO_NUM;
    config.pin_pclk = PCLK_GPIO_NUM;
    config.pin_vsync = VSYNC_GPIO_NUM;
    config.pin_href = HREF_GPIO_NUM;
    config.pin_sccb_sda = SIOD_GPIO_NUM;
    config.pin_sccb_scl = SIOC_GPIO_NUM;
    config.pin_pwdn = PWDN_GPIO_NUM;
    config.pin_reset = RESET_GPIO_NUM;
    config.xclk_freq_hz = 20000000;
    config.pixel_format = PIXFORMAT_JPEG;
    
    // High quality for better analysis
    config.frame_size = FRAMESIZE_VGA;  // 640x480
    config.jpeg_quality = 10;  // Lower = better quality
    config.fb_count = 1;
    
    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK) {
        Serial.printf("Camera init failed: 0x%x\n", err);
        showStatus("Camera Error!");
        return;
    }
    
    Serial.println("Camera initialized");
}

void initI2SMic() {
    i2s_config_t i2s_config = {
        .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
        .sample_rate = 16000,
        .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
        .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
        .communication_format = I2S_COMM_FORMAT_I2S,
        .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
        .dma_buf_count = 4,
        .dma_buf_len = 1024,
        .use_apll = false,
        .tx_desc_auto_clear = false,
        .fixed_mclk = 0
    };
    
    i2s_pin_config_t pin_config = {
        .bck_io_num = I2S_MIC_SCK,
        .ws_io_num = I2S_MIC_WS,
        .data_out_num = I2S_PIN_NO_CHANGE,
        .data_in_num = I2S_MIC_SD
    };
    
    i2s_driver_install(I2S_NUM_0, &i2s_config, 0, NULL);
    i2s_set_pin(I2S_NUM_0, &pin_config);
    
    Serial.println("I2S Mic initialized");
}

void connectWiFi() {
    showStatus("Connecting WiFi...");
    
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
    
    int attempts = 0;
    while (WiFi.status() != WL_CONNECTED && attempts < 20) {
        delay(500);
        Serial.print(".");
        attempts++;
    }
    
    if (WiFi.status() == WL_CONNECTED) {
        Serial.println();
        Serial.print("Connected! IP: ");
        Serial.println(WiFi.localIP());
        showStatus("WiFi Connected");
    } else {
        Serial.println("WiFi Failed!");
        showStatus("WiFi Failed!");
    }
}

// =============================================================================
// DISPLAY
// =============================================================================

void showStatus(const char* status) {
    display.clearDisplay();
    display.setCursor(0, 0);
    display.setTextSize(1);
    display.println("WaveformBuddy");
    display.println("----------------");
    display.setTextSize(2);
    display.println(status);
    display.display();
}

void showInstructions() {
    display.clearDisplay();
    display.setCursor(0, 0);
    display.setTextSize(1);
    display.println("WaveformBuddy Ready");
    display.println("----------------");
    display.println("1 click: Circuit");
    display.println("2 click: Waveform");
    display.println("Hold: Voice");
    display.println("");
    display.print("IP: ");
    display.println(WiFi.localIP());
    display.display();
}

// =============================================================================
// CAPTURE & SEND
// =============================================================================

void captureAndSend(const char* endpoint) {
    showStatus("Capturing...");
    
    camera_fb_t* fb = esp_camera_fb_get();
    if (!fb) {
        Serial.println("Camera capture failed");
        showStatus("Capture Error!");
        return;
    }
    
    showStatus("Sending...");
    
    HTTPClient http;
    String url = String(SERVER_URL) + endpoint;
    
    http.begin(url);
    http.addHeader("Content-Type", "image/jpeg");
    
    int httpCode = http.POST(fb->buf, fb->len);
    
    esp_camera_fb_return(fb);
    
    if (httpCode == 200) {
        String response = http.getString();
        Serial.println("Response: " + response);
        showStatus("Done!");
    } else {
        Serial.printf("HTTP error: %d\n", httpCode);
        showStatus("Send Error!");
    }
    
    http.end();
    
    delay(2000);
    showInstructions();
}

void recordAndSend() {
    showStatus("Listening...");
    
    // Record audio buffer
    const int RECORD_SECONDS = 5;
    const int SAMPLE_RATE = 16000;
    const int BUFFER_SIZE = SAMPLE_RATE * RECORD_SECONDS * 2;  // 16-bit = 2 bytes
    
    uint8_t* audioBuffer = (uint8_t*)malloc(BUFFER_SIZE);
    if (!audioBuffer) {
        showStatus("Memory Error!");
        return;
    }
    
    size_t bytesRead = 0;
    size_t totalRead = 0;
    
    unsigned long startTime = millis();
    while (millis() - startTime < RECORD_SECONDS * 1000) {
        i2s_read(I2S_NUM_0, audioBuffer + totalRead, 
                 min(1024, BUFFER_SIZE - (int)totalRead), &bytesRead, portMAX_DELAY);
        totalRead += bytesRead;
        
        if (totalRead >= BUFFER_SIZE) break;
    }
    
    showStatus("Processing...");
    
    // Send to server
    HTTPClient http;
    String url = String(SERVER_URL) + "/audio";
    
    http.begin(url);
    http.addHeader("Content-Type", "audio/raw");
    
    int httpCode = http.POST(audioBuffer, totalRead);
    
    free(audioBuffer);
    
    if (httpCode == 200) {
        String response = http.getString();
        Serial.println("Response: " + response);
        showStatus("Done!");
    } else {
        showStatus("Error!");
    }
    
    http.end();
    delay(2000);
    showInstructions();
}

// =============================================================================
// BUTTON HANDLING
// =============================================================================

void handleButton() {
    static int clickCount = 0;
    static unsigned long lastClickTime = 0;
    static bool wasPressed = false;
    
    bool isPressed = (digitalRead(BUTTON_PIN) == LOW);
    unsigned long now = millis();
    
    // Detect press
    if (isPressed && !wasPressed) {
        lastClickTime = now;
        wasPressed = true;
    }
    
    // Detect release
    if (!isPressed && wasPressed) {
        unsigned long pressDuration = now - lastClickTime;
        wasPressed = false;
        
        if (pressDuration > 1000) {
            // Long press - voice mode
            recordAndSend();
            clickCount = 0;
        } else {
            // Short click
            clickCount++;
        }
    }
    
    // Process clicks after timeout
    if (clickCount > 0 && (now - lastClickTime > 500) && !wasPressed) {
        if (clickCount == 1) {
            // Single click - capture circuit
            captureAndSend("/circuit");
        } else if (clickCount >= 2) {
            // Double click - capture waveform
            captureAndSend("/waveform");
        }
        clickCount = 0;
    }
}

// =============================================================================
// MAIN LOOP
// =============================================================================

void loop() {
    handleButton();
    
    // Check for serial commands (for testing without button)
    if (Serial.available()) {
        char cmd = Serial.read();
        switch (cmd) {
            case 'c':
                captureAndSend("/circuit");
                break;
            case 'w':
                captureAndSend("/waveform");
                break;
            case 'v':
                recordAndSend();
                break;
            case 'r':
                ESP.restart();
                break;
        }
    }
    
    delay(10);
}
