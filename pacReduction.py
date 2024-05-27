import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.decomposition import PCA

# Define a custom dataset class to load data from CSV
class CustomDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        # Assuming the last 3 columns are the target variables
        self.features = self.data.iloc[:, :-3].values
        self.labels = self.data.iloc[:, -3:].values  # Selecting the last 3 columns as labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)  # Adjust dtype if necessary
        return features, label

# Define the neural network architecture
class CustomDNN(nn.Module):
    def __init__(self, input_dim):
        super(CustomDNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 448)
        self.fc2 = nn.Linear(448, 224)
        self.fc3 = nn.Linear(224, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 32)
        self.fc6 = nn.Linear(32, 3)  # Adjust output size to 3
        self.threshold = nn.Threshold(0, 0)  # Thresholding function
        self.elu = nn.ELU()
        self.hardsigmoid = nn.Hardsigmoid()

    def forward(self, x):
        x = self.elu(self.fc1(x))
        x = self.elu(self.fc2(x))
        x = self.elu(self.fc3(x))
        x = self.elu(self.fc4(x))
        x = self.elu(self.fc5(x))
        x = self.hardsigmoid(self.fc6(x))
        return x

# Define accuracy function
def calculate_accuracy(outputs, labels):
    with torch.no_grad():
        # Convert outputs to predicted class (argmax)
        _, predicted = torch.max(outputs, 1)
        # Convert labels to one-hot encoding
        _, true_labels = torch.max(labels, 1)
        # Compare predicted and true labels
        correct = (predicted == true_labels).sum().item()
        total = labels.size(0)
        accuracy = correct / total
    return accuracy

# Load CSV dataset
csv_file_path = 'EEG_Eye_State_Classification_test_eyuda.csv'
dataset = CustomDataset(csv_file_path)

# Apply PCA for dimensionality reduction
n_components = min(dataset.features.shape[0], dataset.features.shape[1])
pca = PCA(n_components=n_components)
features_reduced = pca.fit_transform(dataset.features)

# Create a data loader with batch size 2
batch_size = 2
reduced_dataset = [(torch.tensor(features_reduced[i], dtype=torch.float32), torch.tensor(dataset.labels[i], dtype=torch.float32)) for i in range(len(dataset))]
trainloader = DataLoader(reduced_dataset, batch_size=batch_size, shuffle=True)

# Create an instance of the model
model = CustomDNN(input_dim=896)  # Adjust input dimension to 896

# Define the loss function (Mean Squared Error) and optimizer (Adam with learning rate 0.01)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Train the model for 1 epoch
for epoch in range(1):
    running_loss = 0.0
    total_accuracy = 0.0
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

        running_loss += loss.item()

        # Calculate accuracy
        batch_accuracy = calculate_accuracy(outputs, labels)
        total_accuracy += batch_accuracy

    # Print average loss and accuracy for the epoch
    average_loss = running_loss / len(trainloader)
    average_accuracy = total_accuracy / len(trainloader)
    print(f"Epoch {epoch+1}, Loss: {average_loss}, Accuracy: {average_accuracy}")

print('Finished Training')
