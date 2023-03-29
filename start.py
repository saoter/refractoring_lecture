import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split

# Define the neural network
class IrisClassifier(nn.Module):
    def __init__(self):
        super(IrisClassifier, self).__init__()
        self.fc1 = nn.Linear(4, 16)
        self.fc2 = nn.Linear(16, 3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# import dataset
df = pd.read_csv("iris.csv")

# Split the data into inputs (X) and labels (y)
X = df.iloc[:, :4].values
y = df.iloc[:, 4].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Convert the labels to one-hot encoded tensors
y_train_onehot = torch.zeros(len(y_train), 3)
y_train_onehot[range(len(y_train)), y_train] = 1

# Define the hyperparameters
learning_rate = 0.01
num_epochs = 500

# Initialize the model, loss function, and optimizer
model = IrisClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    # Convert the inputs and labels to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)

    # Forward pass
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print the loss every 50 epochs
    if (epoch+1) % 50 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

# Evaluate the model on the test set
model.eval()
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)
with torch.no_grad():
    outputs = model(X_test_tensor)
    predicted_classes = torch.argmax(outputs, dim=1)
    accuracy = (predicted_classes == y_test_tensor).float().mean()
print(f"Test accuracy: {accuracy:.4f}")

# Define the input data
input_data = torch.tensor([3.2, 4.0, 0.3, 2.0], dtype=torch.float32)

# Make the prediction
with torch.no_grad():
    output = model(input_data)
    predicted_class = torch.argmax(output).item()

# Print the predicted class
class_names = ["Setosa", "Versicolor", "Virginica"]
print(f"The predicted class is {class_names[predicted_class]}")
