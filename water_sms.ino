#include <ESP8266WiFi.h>
#include <ESP8266HTTPClient.h>

#define sensorPower D1  // GPIO pin for powering the sensor
#define sensorPin A0    // Analog pin on ESP8266
#define redLED D3       // Red LED pin (water level < 150)
#define blueLED D2      // Blue LED pin (water level >= 150)

const char* ssid = "RashMuk47";
const char* password = "knrm2003";
const char* serverUrl = "http://192.168.29.130:5000/sms";  // Replace with your Flask server IP

int baseline = 19;  
float calibrationConstant = 0.6;  // Adjust based on your sensor calibration
bool smsSent = false;  // To prevent sending SMS repeatedly

void setup() {
  Serial.begin(115200);
  WiFi.begin(ssid, password);

  Serial.print("Connecting to WiFi...");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("Connected!");

  pinMode(sensorPower, OUTPUT);
  pinMode(redLED, OUTPUT);
  pinMode(blueLED, OUTPUT);
  digitalWrite(sensorPower, LOW);  // Sensor initially OFF
}

void loop() {
  int rawValue = readSensor();
  int correctedValue = rawValue - baseline;
  float waterVolume = correctedValue * calibrationConstant;

  Serial.print("Water Volume: ");
  Serial.print(waterVolume);
  Serial.println(" mL");

  if (waterVolume < 150) {
    digitalWrite(redLED, HIGH);
    digitalWrite(blueLED, LOW);

    if (!smsSent) {  // Only send SMS once when threshold is crossed
      sendSMS(waterVolume);
      smsSent = true;
    }
  } else {
    digitalWrite(redLED, LOW);
    digitalWrite(blueLED, HIGH);
    smsSent = false;  // Reset flag when water level recovers
  }

  delay(10000);  // Check every 10 seconds
}

int readSensor() {
  digitalWrite(sensorPower, HIGH);
  delay(10);  
  int val = analogRead(sensorPin);
  digitalWrite(sensorPower, LOW);
  return val;
}

void sendSMS(float waterVolume) {
  if (WiFi.status() == WL_CONNECTED) {
    WiFiClient client;
    HTTPClient http;

    http.begin(client, serverUrl);
    http.addHeader("Content-Type", "application/json");

    String message = "Alert!Water level is LOW: " + String(waterVolume) + " mL!";
    String payload = "{\"phone\":\"9150599208\",\"message\":\"" + message + "\"}";

    http.POST(payload);  // 🔹 Send request but ignore response
    http.end();

    Serial.println("SMS Triggered: " + message);
  } else {
    Serial.println("WiFi not connected!");
  }
}
