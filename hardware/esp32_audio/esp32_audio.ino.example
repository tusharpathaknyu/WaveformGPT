/*
 * WaveformBuddy Audio Module
 * 
 * This runs on the ESP32 from your starter kit (NOT the ESP32-CAM).
 * Handles: Microphone input, Speaker output, OLED display
 * 
 * Hardware Connections:
 * 
 * INMP441 I2S Microphone:
 *   VDD  -> 3.3V
 *   GND  -> GND
 *   L/R  -> GND (left channel)
 *   WS   -> GPIO 25
 *   SD   -> GPIO 33
 *   SCK  -> GPIO 32
 * 
 * PAM8302 Amplifier (for speaker):
 *   VIN  -> 5V
 *   GND  -> GND
 *   A+   -> GPIO 26
 *   A-   -> GND
 *   +/-  -> Connect to 3W 4ohm speaker
 * 
 * 0.96" OLED Display (SSD1306):
 *   VCC  -> 3.3V
 *   GND  -> GND
 *   SDA  -> GPIO 21
 *   SCL  -> GPIO 22
 * 
 * Buttons (optional):
 *   Boot button (GPIO 0) - Hold to record voice
 */

#include <WiFi.h>
#include <HTTPClient.h>
#include <driver/i2s.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

// =============================================================================
// CONFIGURATION - EDIT THESE!
// =============================================================================

const char* WIFI_SSID = "YOUR_WIFI_SSID";
const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";
const char* SERVER_URL = "http://YOUR_MAC_IP:8080";  // Your Mac's IP address

// =============================================================================
// PIN DEFINITIONS
// =============================================================================

// I2S Microphone (INMP441)
#define I2S_MIC_SERIAL_CLOCK    32
#define I2S_MIC_LEFT_RIGHT_CLOCK 25
#define I2S_MIC_SERIAL_DATA     33

// I2S Speaker output
#define I2S_SPK_BCLK            26
#define I2S_SPK_LRC             27
#define I2S_SPK_DOUT            14

// OLED Display
#define SCREEN_WIDTH            128
#define SCREEN_HEIGHT           64
#define OLED_RESET              -1
#define OLED_I2C_ADDRESS        0x3C

// Button
#define BUTTON_PIN              0   // Boot button

// LED indicator
#define LED_PIN                 2   // Built-in LED

// =============================================================================
// GLOBALS
// =============================================================================

Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);

// Audio buffer
#define SAMPLE_RATE             16000
#define RECORD_SECONDS          5
#define BUFFER_SIZE             (SAMPLE_RATE * RECORD_SECONDS * 2)  // 16-bit samples

bool isRecording = false;
bool wifiConnected = false;

// =============================================================================
// SETUP
// =============================================================================

void setup() {
    Serial.begin(115200);
    Serial.println("\n\n=== WaveformBuddy Audio Module ===\n");
    
    // Initialize pins
    pinMode(BUTTON_PIN, INPUT_PULLUP);
    pinMode(LED_PIN, OUTPUT);
    digitalWrite(LED_PIN, LOW);
    
    // Initialize OLED display
    initDisplay();
    showStatus("Starting...");
    
    // Connect to WiFi
    connectWiFi();
    
    // Initialize I2S for microphone
    initI2SMicrophone();
    
    // Show ready status
    if (wifiConnected) {
        showReady();
    }
}

// =============================================================================
// INITIALIZATION FUNCTIONS
// =============================================================================

void initDisplay() {
    Wire.begin(21, 22);  // SDA, SCL
    
    if (!display.begin(SSD1306_SWITCHCAPVCC, OLED_I2C_ADDRESS)) {
        Serial.println("SSD1306 OLED not found!");
        return;
    }
    
    display.clearDisplay();
    display.setTextSize(1);
    display.setTextColor(SSD1306_WHITE);
    display.display();
    
    Serial.println("OLED initialized");
}

void initI2SMicrophone() {
    i2s_config_t i2s_config = {
        .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
        .sample_rate = SAMPLE_RATE,
        .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
        .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
        .communication_format = I2S_COMM_FORMAT_STAND_I2S,
        .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
        .dma_buf_count = 8,
        .dma_buf_len = 1024,
        .use_apll = false,
        .tx_desc_auto_clear = false,
        .fixed_mclk = 0
    };
    
    i2s_pin_config_t pin_config = {
        .bck_io_num = I2S_MIC_SERIAL_CLOCK,
        .ws_io_num = I2S_MIC_LEFT_RIGHT_CLOCK,
        .data_out_num = I2S_PIN_NO_CHANGE,
        .data_in_num = I2S_MIC_SERIAL_DATA
    };
    
    esp_err_t err = i2s_driver_install(I2S_NUM_0, &i2s_config, 0, NULL);
    if (err != ESP_OK) {
        Serial.printf("I2S driver install failed: %d\n", err);
        return;
    }
    
    err = i2s_set_pin(I2S_NUM_0, &pin_config);
    if (err != ESP_OK) {
        Serial.printf("I2S set pin failed: %d\n", err);
        return;
    }
    
    Serial.println("I2S Microphone initialized");
}

void connectWiFi() {
    showStatus("Connecting WiFi...");
    Serial.print("Connecting to WiFi");
    
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
    
    int attempts = 0;
    while (WiFi.status() != WL_CONNECTED && attempts < 30) {
        delay(500);
        Serial.print(".");
        attempts++;
    }
    
    Serial.println();
    
    if (WiFi.status() == WL_CONNECTED) {
        wifiConnected = true;
        Serial.print("Connected! IP: ");
        Serial.println(WiFi.localIP());
        showStatus("WiFi OK");
        delay(1000);
    } else {
        Serial.println("WiFi connection failed!");
        showStatus("WiFi FAILED");
    }
}

// =============================================================================
// DISPLAY FUNCTIONS
// =============================================================================

void showStatus(const char* message) {
    display.clearDisplay();
    display.setCursor(0, 0);
    display.setTextSize(1);
    display.println("WaveformBuddy");
    display.println("Audio Module");
    display.println("----------------");
    display.setTextSize(1);
    display.println(message);
    display.display();
}

void showReady() {
    display.clearDisplay();
    display.setCursor(0, 0);
    display.setTextSize(1);
    display.println("WaveformBuddy");
    display.println("----------------");
    display.println("");
    display.println("Hold BOOT button");
    display.println("to speak");
    display.println("");
    display.print("IP: ");
    display.println(WiFi.localIP());
    display.display();
}

void showRecording() {
    display.clearDisplay();
    display.setCursor(0, 0);
    display.setTextSize(2);
    display.println("Recording");
    display.setTextSize(1);
    display.println("");
    display.println("Speak now...");
    display.display();
    
    digitalWrite(LED_PIN, HIGH);
}

void showProcessing() {
    display.clearDisplay();
    display.setCursor(0, 0);
    display.setTextSize(1);
    display.println("Processing...");
    display.println("");
    display.println("Sending to AI");
    display.display();
}

void showResponse(const char* response) {
    display.clearDisplay();
    display.setCursor(0, 0);
    display.setTextSize(1);
    
    // Word wrap the response
    int x = 0;
    int y = 0;
    int charWidth = 6;
    int lineHeight = 8;
    int maxCharsPerLine = SCREEN_WIDTH / charWidth;
    
    String text = String(response);
    int len = text.length();
    int chars = 0;
    
    for (int i = 0; i < len && y < SCREEN_HEIGHT; i++) {
        if (text[i] == '\n' || chars >= maxCharsPerLine) {
            y += lineHeight;
            x = 0;
            chars = 0;
            if (text[i] == '\n') continue;
        }
        
        display.setCursor(x, y);
        display.print(text[i]);
        x += charWidth;
        chars++;
    }
    
    display.display();
}

// =============================================================================
// AUDIO FUNCTIONS
// =============================================================================

bool recordAudio(uint8_t* buffer, size_t bufferSize, size_t* bytesRecorded) {
    showRecording();
    
    size_t totalRead = 0;
    size_t bytesRead = 0;
    
    unsigned long startTime = millis();
    
    while (digitalRead(BUTTON_PIN) == LOW && 
           totalRead < bufferSize && 
           (millis() - startTime) < (RECORD_SECONDS * 1000)) {
        
        // Read from I2S
        esp_err_t result = i2s_read(I2S_NUM_0, 
                                     buffer + totalRead, 
                                     1024, 
                                     &bytesRead, 
                                     portMAX_DELAY);
        
        if (result == ESP_OK) {
            totalRead += bytesRead;
        }
        
        // Visual feedback - blink LED
        if ((millis() / 200) % 2) {
            digitalWrite(LED_PIN, HIGH);
        } else {
            digitalWrite(LED_PIN, LOW);
        }
    }
    
    digitalWrite(LED_PIN, LOW);
    *bytesRecorded = totalRead;
    
    Serial.printf("Recorded %d bytes\n", totalRead);
    return totalRead > 0;
}

bool sendAudioToServer(uint8_t* buffer, size_t size) {
    showProcessing();
    
    if (!wifiConnected) {
        Serial.println("WiFi not connected!");
        showStatus("No WiFi!");
        return false;
    }
    
    HTTPClient http;
    String url = String(SERVER_URL) + "/audio";
    
    Serial.printf("Sending %d bytes to %s\n", size, url.c_str());
    
    http.begin(url);
    http.addHeader("Content-Type", "audio/raw");
    http.addHeader("X-Sample-Rate", String(SAMPLE_RATE));
    http.addHeader("X-Bits-Per-Sample", "16");
    http.addHeader("X-Channels", "1");
    
    int httpCode = http.POST(buffer, size);
    
    if (httpCode == HTTP_CODE_OK) {
        String response = http.getString();
        Serial.println("Server response: " + response);
        
        // Show on display
        showResponse(response.c_str());
        
        http.end();
        return true;
    } else {
        Serial.printf("HTTP error: %d\n", httpCode);
        showStatus("Send failed!");
        http.end();
        return false;
    }
}

// =============================================================================
// TEST FUNCTIONS
// =============================================================================

void testMicrophone() {
    Serial.println("\n=== Microphone Test ===");
    Serial.println("Reading audio samples for 2 seconds...");
    
    int16_t samples[1024];
    size_t bytesRead;
    int maxAmplitude = 0;
    int minAmplitude = 0;
    
    unsigned long startTime = millis();
    
    while (millis() - startTime < 2000) {
        i2s_read(I2S_NUM_0, samples, sizeof(samples), &bytesRead, portMAX_DELAY);
        
        for (int i = 0; i < bytesRead / 2; i++) {
            if (samples[i] > maxAmplitude) maxAmplitude = samples[i];
            if (samples[i] < minAmplitude) minAmplitude = samples[i];
        }
    }
    
    Serial.printf("Max amplitude: %d\n", maxAmplitude);
    Serial.printf("Min amplitude: %d\n", minAmplitude);
    Serial.printf("Range: %d\n", maxAmplitude - minAmplitude);
    
    if (maxAmplitude - minAmplitude > 100) {
        Serial.println("✓ Microphone is working!");
        showStatus("Mic OK!");
    } else {
        Serial.println("✗ Microphone may not be connected properly");
        showStatus("Mic ERROR!");
    }
}

void testDisplay() {
    Serial.println("\n=== Display Test ===");
    
    // Show test pattern
    display.clearDisplay();
    display.fillRect(0, 0, 64, 32, SSD1306_WHITE);
    display.fillRect(64, 32, 64, 32, SSD1306_WHITE);
    display.display();
    delay(500);
    
    display.clearDisplay();
    display.drawRect(0, 0, 128, 64, SSD1306_WHITE);
    display.drawLine(0, 0, 127, 63, SSD1306_WHITE);
    display.drawLine(127, 0, 0, 63, SSD1306_WHITE);
    display.display();
    delay(500);
    
    display.clearDisplay();
    display.setTextSize(2);
    display.setCursor(10, 20);
    display.println("Display OK");
    display.display();
    delay(1000);
    
    Serial.println("✓ Display test complete");
}

void testWiFiConnection() {
    Serial.println("\n=== WiFi Connection Test ===");
    
    HTTPClient http;
    String url = String(SERVER_URL) + "/ping";
    
    http.begin(url);
    int httpCode = http.GET();
    
    if (httpCode > 0) {
        Serial.printf("Server responded with code: %d\n", httpCode);
        showStatus("Server OK!");
    } else {
        Serial.printf("Connection failed: %s\n", http.errorToString(httpCode).c_str());
        showStatus("Server Error!");
    }
    
    http.end();
}

// =============================================================================
// MAIN LOOP
// =============================================================================

void loop() {
    // Check for button press (hold to record)
    if (digitalRead(BUTTON_PIN) == LOW) {
        delay(50);  // Debounce
        
        if (digitalRead(BUTTON_PIN) == LOW) {
            // Allocate buffer
            uint8_t* audioBuffer = (uint8_t*)malloc(BUFFER_SIZE);
            
            if (audioBuffer) {
                size_t bytesRecorded = 0;
                
                // Record while button is held
                if (recordAudio(audioBuffer, BUFFER_SIZE, &bytesRecorded)) {
                    // Send to server
                    sendAudioToServer(audioBuffer, bytesRecorded);
                    
                    // Wait for button release
                    while (digitalRead(BUTTON_PIN) == LOW) {
                        delay(10);
                    }
                    
                    // Show for a moment then return to ready
                    delay(3000);
                }
                
                free(audioBuffer);
            } else {
                Serial.println("Failed to allocate audio buffer!");
                showStatus("Memory Error!");
            }
            
            showReady();
        }
    }
    
    // Serial commands for testing
    if (Serial.available()) {
        char cmd = Serial.read();
        
        switch (cmd) {
            case 'm':  // Test microphone
                testMicrophone();
                delay(2000);
                showReady();
                break;
                
            case 'd':  // Test display
                testDisplay();
                showReady();
                break;
                
            case 'w':  // Test WiFi
                testWiFiConnection();
                delay(2000);
                showReady();
                break;
                
            case 'r':  // Restart
                ESP.restart();
                break;
                
            case 'h':  // Help
                Serial.println("\n=== Commands ===");
                Serial.println("m - Test microphone");
                Serial.println("d - Test display");
                Serial.println("w - Test WiFi/server connection");
                Serial.println("r - Restart ESP32");
                Serial.println("h - Show this help");
                break;
        }
    }
    
    delay(10);
}
