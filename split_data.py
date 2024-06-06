import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader, Subset
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.num_columns = self.data.shape[1]  # Number of columns in the CSV file
        # Selecting the last 3 columns as labels
        self.features = self.data.iloc[:, :-3].values
        self.labels = np.argmax(self.data.iloc[:, -3:].values, axis=1)  # Extract class indices

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)  
        return features, label

# Load CSV dataset
csv_file_path = 'EEG_Eye_State_Classification_test_eyuda_average.csv'
dataset = CustomDataset(csv_file_path)

print(f"dataset.labels: {dataset.labels}")
# Count occurrences of each class
class_counts = np.bincount(dataset.labels)
print("Class counts:", class_counts)

# Determine number of samples needed for each class in training set (70%) and validation set (10%) and test set (20%)
train_class_counts = (0.7 * class_counts).astype(int)
validation_class_counts = (0.1 * class_counts).astype(int)
test_class_counts = class_counts - train_class_counts - validation_class_counts

print("Train class counts:", train_class_counts)
print("Validation class counts:", validation_class_counts)
print("Test class counts:", test_class_counts)

# Initialize lists to store selected samples for each class
train_indices = []
validation_indices = []
test_indices = []

# Iterate through dataset.labels and select samples for each set
for i, label in enumerate(dataset.labels):
    if train_class_counts[label] > 0:
        train_indices.append(i)
        train_class_counts[label] -= 1
    elif validation_class_counts[label] > 0:
        validation_indices.append(i)
        validation_class_counts[label] -= 1
    else:
        test_indices.append(i)

print("Train indices:", train_indices)
print("Validation indices:", validation_indices)
print("Test indices:", test_indices)

# Create subsets
train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, validation_indices)
test_dataset = Subset(dataset, test_indices)

# Print the lengths of the subsets
print("Length of training set:", len(train_dataset))
print("Length of validation set:", len(val_dataset))
print("Length of test set:", len(test_dataset))
