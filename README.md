
# Real-Time Anomaly Detection with Isolation Forest

This repository contains a Python implementation for **real-time anomaly detection** using the **Isolation Forest** algorithm. It dynamically detects outliers in a continuous data stream and visualizes the results in real time, making it ideal for monitoring systems, detecting fraud, and identifying unusual patterns.

## 🛠️ Features:
- **Data Stream Simulation**: Generates a stream of Gaussian-distributed data points with injected anomalies.
- **Sliding Window Detection**: The Isolation Forest model is retrained on a sliding window of data to ensure it adapts to evolving patterns.
- **Real-Time Anomaly Detection**: Each incoming data point is classified as normal or anomalous, with anomalies highlighted in real time.
- **Interactive Visualization**: Continuous data points are plotted live, with normal data shown in blue and anomalies in red.

## 🔧 Dependencies:
- `numpy`
- `matplotlib`
- `scikit-learn`

You can install all dependencies via:
```
pip install -r requirements.txt
```

## 🚀 Usage:
1. Clone this repository:
   ```
   git clone https://github.com/yourusername/anomaly-detection-stream.git
   ```
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the script to see real-time anomaly detection in action:
   ```
   python anomaly_detection.py
   ```

## 📊 Visualization:
The visualization dynamically updates as data flows in, showing how the Isolation Forest detects anomalies in real time. Normal points are plotted in blue, and anomalies are flagged in red.

## 💡 Use Cases:
- Fraud detection in financial systems
- Intrusion detection in cybersecurity
- Monitoring and alerting for system failures
- Detecting abnormal patterns in IoT data streams
