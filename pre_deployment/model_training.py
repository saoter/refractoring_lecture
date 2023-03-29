import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import sqlite3
from datetime import datetime
import os

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

# Create a SQLite database connection
conn = sqlite3.connect('database/iris.db')

# Generate the training data by querying the database
query = '''
        SELECT sepal_width, sepal_length, petal_width, petal_length, Species
        FROM sepal_table
        INNER JOIN petal_table ON sepal_table.item_id = petal_table.item_id
        INNER JOIN ident_table ON sepal_table.item_id = ident_table.item_id
        '''
df = pd.read_sql_query(query, conn)
print(df.head)

# Close the database connection
conn.close()

# Split the data into inputs (X) and labels (y)
X = df.iloc[:, :4].values
y = df.iloc[:, 4].values

# Convert the labels to one-hot encoded tensors
y_onehot = torch.zeros(len(y), 3)
y_onehot[range(len(y)), y] = 1

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
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    # Forward pass
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print the loss every 50 epochs
    if (epoch+1) % 50 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

# Save the model with current date in folder 'models'
now = datetime.now().strftime("%Y-%m-%d")
model_name = f"iris_classifier_{now}.pth"
model_folder = 'models'
if not os.path.exists(model_folder):
    os.makedirs(model_folder)
model_path = os.path.join(model_folder, model_name)
torch.save(model.state_dict(), model_path)