import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Set random seeds for reproducibility
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed()

# ===========================
# 1. Data Acquisition
# ===========================

def fetch_bitcoin_data(ticker='BTC-USD', start_date='2018-01-01', end_date='2023-12-31'):
    data = yf.download(ticker, start=start_date, end=end_date)
    if data.empty:
        raise ValueError("No data fetched. Please check the ticker symbol and date range.")
    return data

# ===========================
# 2. Data Preprocessing
# ===========================

def preprocess_data(data, lag_days=10):
    # Calculate daily returns
    data['Return'] = data['Adj Close'].pct_change()
    data = data.dropna()

    # Create lag features
    for i in range(1, lag_days + 1):
        data[f'lag_{i}'] = data['Return'].shift(i)

    data = data.dropna()

    # Define feature columns
    feature_columns = [f'lag_{i}' for i in range(lag_days, 0, -1)]
    target_column = 'Return'

    X = data[feature_columns].values
    y = data[target_column].values

    return X, y, feature_columns

# ===========================
# 3. Train-Test Split and Scaling
# ===========================

def split_and_scale(X, y, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

# ===========================
# 4. PyTorch Dataset and DataLoader
# ===========================

class BitcoinReturnsDataset(Dataset):
    def __init__(self, features, targets):
        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(targets, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def create_data_loaders(X_train, X_test, y_train, y_test, batch_size=64):
    train_dataset = BitcoinReturnsDataset(X_train, y_train)
    test_dataset = BitcoinReturnsDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

# ===========================
# 5. Define the LSTM-based Deep Learning Model
# ===========================

class BitcoinReturnLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super(BitcoinReturnLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Define LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        
        # Define fully connected layer
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]  # Get the output from the last timestep
        out = self.fc(out)
        return out

# ===========================
# 6. Training Function
# ===========================

def train_model(model, train_loader, test_loader, criterion, optimizer, epochs=100, device='cpu'):
    model.to(device)
    train_losses = []
    test_losses = []
    
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * batch_X.size(0)
        
        avg_train_loss = epoch_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)
        
        # Evaluate on test data
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                test_loss += loss.item() * batch_X.size(0)
        avg_test_loss = test_loss / len(test_loader.dataset)
        test_losses.append(avg_test_loss)
        
        # Print progress every 10 epochs and first epoch
        if epoch % 10 == 0 or epoch == 1:
            print(f'Epoch {epoch}/{epochs} | Training Loss: {avg_train_loss:.6f} | Test Loss: {avg_test_loss:.6f}')
    
    return train_losses, test_losses

# ===========================
# 7. Evaluation Function
# ===========================

def evaluate_model(model, X_test, y_test, scaler, device='cpu'):
    model.to(device)
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        predictions = model(X_test_tensor).cpu().numpy()
        actuals = y_test.reshape(-1, 1)
    return predictions, actuals

# ===========================
# 8. Visualization Functions
# ===========================

def plot_losses(train_losses, test_losses):
    plt.figure(figsize=(10,6))
    plt.plot(range(1, len(train_losses)+1), train_losses, label='Training Loss')
    plt.plot(range(1, len(test_losses)+1), test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training and Test Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_predictions(actuals, predictions):
    plt.figure(figsize=(14,7))
    plt.plot(actuals, label='Actual Returns')
    plt.plot(predictions, label='Predicted Returns')
    plt.xlabel('Time')
    plt.ylabel('Daily Return')
    plt.title('Actual vs Predicted Bitcoin Daily Returns')
    plt.legend()
    plt.grid(True)
    plt.show()

# ===========================
# 9. Saving and Loading the Model
# ===========================

def save_model(model, path='bitcoin_return_predictor.pth'):
    torch.save(model.state_dict(), path)
    print(f'Model saved to {path}')

def load_model(input_size, path='bitcoin_return_predictor.pth', device='cpu'):
    model = BitcoinReturnLSTM(input_size)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    print(f'Model loaded from {path}')
    return model

# ===========================
# 10. Main Execution Flow
# ===========================

def main():
    # Parameters
    TICKER = 'BTC-USD'
    START_DATE = '2018-01-01'
    END_DATE = '2023-12-31'
    LAG_DAYS = 30
    TEST_SIZE = 0.2
    BATCH_SIZE = 64
    EPOCHS = 100
    LEARNING_RATE = 0.001
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f'Using device: {DEVICE}')
    
    # 1. Fetch Data
    print("Fetching Bitcoin data...")
    data = fetch_bitcoin_data(ticker=TICKER, start_date=START_DATE, end_date=END_DATE)
    print("Data fetched successfully.\n")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

# ===========================
# 4. PyTorch Dataset and DataLoader
# ===========================

class BitcoinReturnsDataset(Dataset):
    def __init__(self, features, targets):
        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(targets, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def create_data_loaders(X_train, X_test, y_train, y_test, batch_size=64):
    train_dataset = BitcoinReturnsDataset(X_train, y_train)
    test_dataset = BitcoinReturnsDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

# ===========================
# 5. Define the LSTM Model
# ===========================

class BitcoinReturnLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super(BitcoinReturnLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Define LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        
        # Define fully connected layer
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: (batch, seq_len, hidden_size)
        
        # Take the output from the last time step
        out = out[:, -1, :]  # (batch, hidden_size)
        
        # Pass through fully connected layers
        out = self.fc(out)  # (batch, 1)
        return out

# ===========================
# 6. Training Function
# ===========================

def train_model(model, train_loader, test_loader, criterion, optimizer, epochs=100, device='cpu'):
    model.to(device)
    train_losses = []
    test_losses = []
    
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * batch_X.size(0)
        
        avg_train_loss = epoch_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)
        
        # Evaluate on test data
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                test_loss += loss.item() * batch_X.size(0)
        avg_test_loss = test_loss / len(test_loader.dataset)
        test_losses.append(avg_test_loss)
        
        if epoch % 10 == 0 or epoch == 1:
            print(f'Epoch {epoch}/{epochs} | Training Loss: {avg_train_loss:.6f} | Test Loss: {avg_test_loss:.6f}')
    
    return train_losses, test_losses

# ===========================
# 7. Evaluation Function
# ===========================

def evaluate_model(model, X_test, y_test, scaler, device='cpu'):
    model.to(device)
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        predictions = model(X_test_tensor).cpu().numpy()
        actuals = y_test.reshape(-1, 1)
    return predictions, actuals

# ===========================
# 8. Visualization Functions
# ===========================

def plot_losses(train_losses, test_losses):
    plt.figure(figsize=(10,6))
    plt.plot(range(1, len(train_losses)+1), train_losses, label='Training Loss')
    plt.plot(range(1, len(test_losses)+1), test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training and Test Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_predictions(actuals, predictions):
    plt.figure(figsize=(14,7))
    plt.plot(actuals, label='Actual Returns')
    plt.plot(predictions, label='Predicted Returns')
    plt.xlabel('Time')
    plt.ylabel('Daily Return')
    plt.title('Actual vs Predicted Bitcoin Daily Returns')
    plt.legend()
    plt.grid(True)
    plt.show()

# ===========================
# 9. Saving and Loading the Model
# ===========================

def save_model(model, path='bitcoin_return_lstm.pth'):
    torch.save(model.state_dict(), path)
    print(f'Model saved to {path}')

def load_model(input_size, path='bitcoin_return_lstm.pth', device='cpu'):
    model = BitcoinReturnLSTM(input_size)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    print(f'Model loaded from {path}')
    return model

# ===========================
# 10. Making Future Predictions
# ===========================

def predict_next_return(model, recent_returns, scaler, device='cpu'):
    if len(recent_returns) != model.lstm.input_size:
        raise ValueError(f"Expected {model.lstm.input_size} recent returns, got {len(recent_returns)}")
    
    recent_returns = np.array(recent_returns).reshape(1, -1)  # Reshape for LSTM
    recent_returns_scaled = scaler.transform(recent_returns)
    
    input_tensor = torch.tensor(recent_returns_scaled, dtype=torch.float32).to(device)
    
    model.eval()
    with torch.no_grad():
        prediction = model(input_tensor).item()
    
    return prediction

# ===========================
# 11. Main Execution Flow
# ===========================

def main():
    TICKER = 'BTC-USD'
    START_DATE = '2018-01-01'
    END_DATE = '2023-12-31'
    LAG_DAYS = 30
    TEST_SIZE = 0.2
    BATCH_SIZE = 64
    EPOCHS = 100
    LEARNING_RATE = 0.001
    MODEL_PATH = 'bitcoin_return_lstm.pth'
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f'Using device: {DEVICE}')
    
    data = fetch_bitcoin_data(ticker=TICKER, start_date=START_DATE, end_date=END_DATE)
    X, y, feature_columns = preprocess_data(data, lag_days=LAG_DAYS)
    
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = split_and_scale(X, y, test_size=TEST_SIZE)
    train_loader, test_loader = create_data_loaders(X_train_scaled, X_test_scaled, y_train, y_test, batch_size=BATCH_SIZE)
    
    input_size = X_train_scaled.shape[1]
    model = BitcoinReturnLSTM(input_size)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    train_losses, test_losses = train_model(model, train_loader, test_loader, criterion, optimizer, epochs=EPOCHS, device=DEVICE)
    
    plot_losses(train_losses, test_losses)
    
    predictions, actuals = evaluate_model(model, X_test_scaled, y_test, scaler, device=DEVICE)
    
    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    
    print(f'Mean Absolute Error (MAE): {mae:.6f}')
    print(f'Root Mean Squared Error (RMSE): {rmse:.6f}')
    
    plot_predictions(actuals, predictions)
    
    save_model(model, path=MODEL_PATH)
    
    loaded_model = load_model(input_size, path=MODEL_PATH, device=DEVICE)
    
    last_returns = data['Return'].values[-LAG_DAYS:]
    next_return = predict_next_return(loaded_model, last_returns, scaler, device=DEVICE)
    print(f'Predicted Next Return: {next_return:.6f}')

if __name__ == "__main__":
    main()

