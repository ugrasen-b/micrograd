import torch
import torch.nn as nn
import torch.optim as optim

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(1, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    

torch.manual_seed(42)
x_train = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
y_train = 2 * x_train + 3 + 0.2 * torch.rand(x_train.size())

model = SimpleNN()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr = 0.1)

epochs = 500
for epoch in range(epochs):
    model.train()

    # Forward pass
    predictions = model(x_train)
    loss = criterion(predictions, y_train)

    # backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 50 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

model.eval()
with torch.no_grad():
    test_input = torch.tensor([[4.0]])
    test_output = model(test_input)

    print(f"Prediction for the input 4.0 is:  {test_output.item():.4f}")

torch.save(model.state_dict(), "simple_nn.pth")
print("Model saved, execution finished")
