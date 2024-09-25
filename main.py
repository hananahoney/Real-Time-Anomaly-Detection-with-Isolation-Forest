import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from collections import deque

def data_generation(num_points=1000, data_std=0.5, anomaly_fraction=0.05):
    np.random.seed(42)

    #Generate Data using Gaussian distribution
    data = []
    data_center = np.random.uniform(-10, 10, 2)  # Random data center
    print(data_center)
    data_points = np.random.normal(loc=data_center, scale=data_std, size=(num_points, 2))
    data.append(data_points)

    data = np.vstack(data) #Stacking data to form a single array

    num_anomalies = int(anomaly_fraction * num_points) #Generating anomalies

    anomalies = np.random.uniform(-15, 15, size=(num_anomalies // 2, 2)) #Anomalies

    full_data = np.vstack([data, anomalies]) #Final dataset

    labels = np.hstack([np.ones(len(data)), -1 * np.ones(len(anomalies))]) #Labels (1 for normal data, -1 for anomalies)

    return full_data, labels


class IsolationForestDetector:
    def __init__(self, window_size=100, contamination=0.05):
        self.window_size = window_size
        self.contamination = contamination
        self.model = IsolationForest(contamination=self.contamination)
        self.data_window = deque(maxlen=window_size)

    def update_and_detect(self, new_value):
        self.data_window.append(new_value) #Appending the new data point to the sliding window

        #Training the model once we have enough data in the window
        if len(self.data_window) == self.window_size:
            data_array = np.array(self.data_window)

            self.model.fit(data_array) #For real-time updates, re-training the model using only the sliding window

            is_anomaly = self.model.predict([new_value])[0] == -1  #Predicting if the new value is an anomaly (-1 means anomaly, 1 means normal)

            return is_anomaly
        return False  # Not enough data to detect


def real_time_visualization(data_stream, anomaly_detector, refresh_rate=0.01): #Function to Real-Time Visualization of data
    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots(figsize=(10, 5))
    xdata, ydata = [], []
    anomaly_x, anomaly_y = [], []

    for idx, new_value in enumerate(data_stream):
        xdata.append(new_value[0])
        ydata.append(new_value[1])


        is_anomaly = anomaly_detector.update_and_detect(new_value) #Detecting anomalies using Isolation Forest
        if is_anomaly:
            anomaly_x.append(new_value[0])
            anomaly_y.append(new_value[1])

        # Clearing and updating the plot with new data point
        ax.clear()
        ax.scatter(xdata, ydata, label="Data Stream", color='blue', s=10)
        ax.scatter(anomaly_x, anomaly_y, color='red', label="Anomalies", s=30)
        ax.legend()
        ax.set_title("Real-Time Anomaly Detection with Isolation Forest")
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")

        plt.pause(refresh_rate)  #Pausing to mimic real-time updates

    plt.ioff()  #Turning off interactive mode after the loop is done
    plt.show()


if __name__ == "__main__":

    data_stream, labels = data_generation(num_points=1000) #Generating data
    anomaly_detector = IsolationForestDetector(window_size=100, contamination=0.05) #Initializing Isolation Forest detector with a sliding window
    real_time_visualization(data_stream, anomaly_detector, refresh_rate=0.01) #Real-time visualization
