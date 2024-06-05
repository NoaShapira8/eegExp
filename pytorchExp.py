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
        self.labels = np.argmax(self.data.iloc[:, -3:].values, axis=1)  # Extract class indices - 0, 1, 2

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)  
        return features, label

# Define the neural network architecture
class CustomDNN(nn.Module):
    def __init__(self):
        super(CustomDNN, self).__init__()
        self.fc1 = nn.Linear(1792, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 3)
        self.threshold = nn.Threshold(0, 0)  # Thresholding function
        self.elu = nn.ELU()
        self.hardsigmoid = nn.Hardsigmoid()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.fc5(x)
        return x


# Load CSV dataset
csv_file_path = 'EEG_Eye_State_Classification_test_eyuda_average.csv'
dataset = CustomDataset(csv_file_path)

# Count occurrences of each class
class_counts = np.bincount(dataset.labels)
#print("Class counts:", class_counts)

# Determine number of samples needed for each class in training set (70%) and validation set (10%) and test set (20%)
train_class_counts = (0.7 * class_counts).astype(int)
validation_class_counts = (0.1 * class_counts).astype(int)
test_class_counts = class_counts - train_class_counts - validation_class_counts

#print("Train class counts:", train_class_counts)
#print("Validation class counts:", validation_class_counts)
#print("Test class counts:", test_class_counts)

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

#print("Train indices:", train_indices)
#print("Validation indices:", validation_indices)
#print("Test indices:", test_indices)

# Create subsets
train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, validation_indices)
test_dataset = Subset(dataset, test_indices)

#print(f"test_dataset: {test_dataset.indices}")

# Create a data loader
batch_size = 2
trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Create an instance of the model
model = CustomDNN()

# Define the loss function (Mean Squared Error) and optimizer Adam
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

train_losses = []
val_losses = []
print("Start training")
# Train the model 
for epoch in range(10):
    model.train()
    for inputs, labels in trainloader: 
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Calculate loss
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Train Loss: {loss.item()}")
    # Store the loss for plotting
    train_losses.append(loss.item())

    # Validation loop
    model.eval()
    with torch.no_grad():
        for inputs, labels in valloader:
            outputs = model(inputs)
            val_loss = criterion(outputs, labels)
    
    print(f"Epoch {epoch+1}, Validation Loss: {val_loss.item()}")
    val_losses.append(val_loss.item())

print('Finished Training')


# Plot the loss curve
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()

print("Start testing")
# Evaluate the model and display the confusion matrix
actual_labels = []
all_predictions = []
model.eval()
with torch.no_grad():
    for inputs, labels in testloader:
        outputs = model(inputs)
        #print(labels)
        _, predicted = torch.max(outputs, 1)
        actual_labels.extend(labels.numpy())
        all_predictions.extend(predicted.numpy())

print(f"labels: {actual_labels} ")
print(f"predictions: {all_predictions}")

# Plot the confusion matrix
cm = confusion_matrix(actual_labels, all_predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()
print("Finished testing")