import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


class CustomDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.num_columns = self.data.shape[1]  # Number of columns in the CSV file
        # Selecting the last 3 columns as labels
        self.features = self.data.iloc[:, :-3].values
        self.labels = self.data.iloc[:, -3:].values  

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)  # Adjust dtype if necessary
        return features, label

# Define the neural network architecture
class CustomDNN(nn.Module):
    def __init__(self):
        super(CustomDNN, self).__init__()
        self.fc1 = nn.Linear(1792, 896)
        self.fc2 = nn.Linear(896, 448)
        self.fc3 = nn.Linear(448, 224)
        self.fc4 = nn.Linear(224, 128)
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, 32)
        self.fc7 = nn.Linear(32, 3)
        self.threshold = nn.Threshold(0, 0)  # Thresholding function
        self.elu = nn.ELU()
        self.hardsigmoid = nn.Hardsigmoid()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.threshold(x)  # Apply scaling
        x = self.hardsigmoid(self.fc1(x)) 
        x = self.hardsigmoid(self.fc2(x))
        x = self.elu(self.fc3(x))
        x = self.elu(self.fc4(x))
        x = self.elu(self.fc5(x))
        x = self.elu(self.fc6(x))
        x = self.hardsigmoid(self.fc7(x))
        return x


# Load CSV dataset
csv_file_path = 'EEG_Eye_State_Classification_test_eyuda.csv'
dataset = CustomDataset(csv_file_path)

# Print the length of the CSV file and the number of columns
#print("Length of the CSV file:", len(dataset))
#print("Number of columns in the CSV file:", dataset.num_columns)

# Create a data loader with batch size 
batch_size = 2
trainloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
# Create an instance of the model
model = CustomDNN()

# Define the loss function (Mean Squared Error) and optimizer (Adam with learning rate 0.01)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

losses = []
# Train the model 
for epoch in range(10):
    total_accuracy = 0.0
    running_loss = 0.0
    for inputs, labels in trainloader:  # Assuming trainloader is your data loader
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

    # Print average loss for the epoch
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

    # Store the loss for plotting
    losses.append(loss.item())
print('Finished Training')

# Plot the loss curve
plt.plot(losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.show()