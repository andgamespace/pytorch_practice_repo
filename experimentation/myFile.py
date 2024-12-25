import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pickle
import os

# Fix create_sequences function
def create_sequences(data, seq_length):
    xs, ys = [], []
    if len(data) <= seq_length:
        return torch.empty(0), torch.empty(0)
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return torch.stack(xs), torch.stack(ys)

# Define scaled_data
scaled_data = torch.randn(1000, 3)  # Example data, replace with actual scaled data

seq_length = 30  # Past month of data
middle_index = len(scaled_data) // 2
X, y = create_sequences(scaled_data[middle_index - seq_length:middle_index + seq_length], seq_length)

# Define neural network
class StockPredictor(nn.Module):
    def __init__(self):
        super(StockPredictor, self).__init__()
        self.lstm = nn.LSTM(input_size=3, hidden_size=100, num_layers=3, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(100, 3)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        return self.fc(last_out)

model = StockPredictor()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model if not already trained
model_path = 'trained_model.pkl'
if not os.path.exists(model_path):
    epochs = 200
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')

    # Save the trained model
    with open(model_path, 'wb') as f:
        pickle.dump(model.state_dict(), f)
else:
    # Load the trained model
    with open(model_path, 'rb') as f:
        model.load_state_dict(pickle.load(f))

# Evaluate the model
model.eval()
with torch.no_grad():
    predictions = model(X).numpy()

# Inverse transform the predictions
# Assuming scaler is defined and fitted elsewhere in the code
# predictions = scaler.inverse_transform(predictions)

# Plot predictions for AAPL
plt.figure(figsize=(14, 7))
plt.plot(range(seq_length, seq_length + len(predictions)), predictions[:, 0], label='Predicted AAPL Prices')
plt.plot(range(seq_length, seq_length + len(predictions)), y[:, 0], label='Actual AAPL Prices')
plt.title('Predicted vs Actual AAPL Prices')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

# Plot only predictions for AAPL
plt.figure(figsize=(14, 7))
plt.plot(range(seq_length, seq_length + len(predictions)), predictions[:, 0], label='Predicted AAPL Prices')
plt.title('Predicted AAPL Prices')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

# Plot predictions vs real
plt.figure(figsize=(14, 7))
plt.plot(range(seq_length, seq_length + len(predictions)), predictions[:, 0], label='Predicted AAPL Prices')
plt.plot(range(seq_length, seq_length + len(predictions)), y[:, 0], label='Real AAPL Prices')
plt.title('Predicted vs Real AAPL Prices')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

# Generate predictions for the next week based on past two months
future_seq_length = 60  # Past two months
if len(scaled_data) >= future_seq_length + 7:  # Ensure enough data for future predictions
    future_X, _ = create_sequences(scaled_data[-(future_seq_length + 7):], future_seq_length)
    if future_X.size(0) > 0:
        with torch.no_grad():
            future_predictions = model(future_X).numpy()

        # Plot future predictions for AAPL
        plt.figure(figsize=(14, 7))
        plt.plot(range(future_seq_length), scaled_data[-(future_seq_length + 7):-7, 0], label='Past AAPL Prices', color='blue')
        plt.plot(range(future_seq_length, future_seq_length + len(future_predictions)), future_predictions[:, 0], label='Future Predicted AAPL Prices', color='red')
        plt.title('Future Predicted AAPL Prices Based on Past Two Months')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        plt.show()
    else:
        print("Not enough data to create future sequences.")
else:
    print("Not enough data to create future sequences.")