import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the CSV data
data = pd.read_csv("EEG_Eye_State_Classification.csv")

# Calculate the time axis
total_duration = 117  # Total duration in seconds
num_samples = 14980  # Total number of samples
time_axis = np.linspace(0, total_duration, num_samples)  # Time over the 117 seconds

# Number of EEG features (assuming first 13 columns are EEG measurements)
num_features = 13  # Adjust according to your data structure

# Create a figure with multiple subplots
plt.figure(figsize=(15, 30))  # Adjust as needed
plt.suptitle("EEG Measurements over Time with Eye Detection State")

# Plot each EEG measurement over time with eye state indication
for i in range(num_features):
    plt.subplot(num_features, 1, i + 1)  # Create a subplot for each EEG sensor
    feature_name = data.columns[i]  # EEG sensor name
    plt.plot(time_axis, data[feature_name], label=f"Sensor {i + 1}")  # Line plot with time
    plt.scatter(time_axis, data[feature_name], c=data["eyeDetection"], cmap='bwr', alpha=0.5)  # Scatter with color mapping
    plt.xlabel("Time (seconds)")
    plt.ylabel("Voltage (mV)")  # Adjust if different units
    plt.title(f"{feature_name} over Time")
    plt.legend()

# Adjust layout to avoid overlapping
plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Leave room for the suptitle
plt.show()
