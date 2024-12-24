import torch
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch import nn, optim

# Fetch data
tickers = ['AAPL', 'MSFT', 'AMZN']
data = {}
for ticker in tickers:
    data[ticker] = yf.download(ticker, period='59d', interval='2m')

# Create DataFrame
df = pd.concat([data[ticker]['Close'].rename(ticker) for ticker in tickers], axis=1)

# Preprocess data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

# Prepare training data
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return torch.tensor(xs, dtype=torch.float32), torch.tensor(ys, dtype=torch.float32)

seq_length = 60
X, y = create_sequences(scaled_data, seq_length)

# Define neural network
class StockPredictor(nn.Module):
    def __init__(self):
        super(StockPredictor, self).__init__()
        self.lstm = nn.LSTM(input_size=3, hidden_size=50, num_layers=2, batch_first=True)
        self.fc = nn.Linear(50, 3)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        return self.fc(last_out)

model = StockPredictor()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
epochs = 100
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    output = model(X)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

# Evaluate the model
model.eval()
with torch.no_grad():
    predictions = model(X).numpy()

# Inverse transform the predictions
predictions = scaler.inverse_transform(predictions)

# Plot predictions for AAPL
plt.figure(figsize=(14, 7))
plt.plot(df.index[seq_length:], predictions[:, 0], label='Predicted AAPL Prices')
plt.plot(df.index, df['AAPL'], label='Actual AAPL Prices')
plt.legend()
plt.show()
