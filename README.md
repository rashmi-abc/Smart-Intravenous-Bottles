# Smart-Intravenous-Bottles
An IoT-based Smart IV Monitoring System using sensors and GSM alerts for real-time tracking of fluid levels, improving patient care by reducing manual checks and preventing complications.

The system uses:
- **Load cell amplifier** to measure fluid weight
- **Ultrasonic sensor** to detect fluid levels
- **xkc-y25-v liquid level sensor** to monitor IV fluid level
- **Microcontrollers** like Arduino UNO or ESP8266 module
- **Fast2SMS API** for SMS and call alerts to notify nurses or doctors

This solution is especially useful in busy hospital environments and for patients requiring critical care, such as those undergoing oncology treatments.

## Features

- Real-time monitoring of IV fluid level and flow rate
- SMS and call alerts on low fluid levels
- Wireless data transmission for remote monitoring
- Low-cost, energy-efficient, and non-invasive
- Easily integrable with hospital management systems
- Web app integration for real-time updates
- ML-based detection of reverse blood flow to prevent critical complications

## Tech Stack

- Arduino UNO/ESP8266
- Load Cell
- xkc-y25-v liquid level sensor
- GSM Module (SIM800L/SIM900)
- Fast2SMS API
- Serial  Communication (optional)
- C/C++ (Arduino IDE)
- Python (for backend and ML model integration)
- Flask (for REST API and web interface)
- Keras & TensorFlow (for machine learning-based reverse blood flow detection)

## Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/smart-iv-monitoring.git
   cd smart-iv-monitoring
Open the Arduino sketch in the Arduino IDE.

2. Connect your sensors and modules as per the circuit diagram.

   Upload the code to your microcontroller.

3. Create an API key in Fast2SMS  and ensure it has sufficient balance for SMS/calls.

## Future Enhancements

-Cloud-based data logging and analytics

-Advanced sensor calibration for higher accuracy

## License
This project is licensed under the MIT License.
