import torch
import torch.nn as nn
import os
from functions import print_models


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

model_name = "models/iris_classifier_2023-03-28.pth"
model = IrisClassifier()
model.load_state_dict(torch.load(model_name))
model.eval()

input_data = torch.tensor([3.2, 4.0, 0.3, 2.0], dtype=torch.float32)

# Make the prediction
with torch.no_grad():
    output = model(input_data)
    predicted_class = torch.argmax(output).item()

# Print the predicted class
class_names = ["Setosa", "Versicolor", "Virginica"]
print(f"The predicted class is {class_names[predicted_class]}")


# Call the print_models function
print_models()