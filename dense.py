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

import os

# Set random seeds for reproducibility
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed()

# Suppress warnings (optional)
import warnings
warnings.filterwarnings('ignore')

# ===========================
# 1. Data Acquisition
# ===========================

def fetch_bitcoin_data(ticker='BTC-USD', start_date='2018-01-01', end_date='2023-12-31'):
    """
    Fetch historical Bitcoin data from Yahoo Finance.

    :param ticker: Ticker symbol for Bitcoin.
    :param start_date: Start date for fetching data.
    :param end_date: End date for fetching data.
    :return: Pandas DataFrame with historical data.
    """
    data = yf.download(ticker, start=start_date, end=end_date)
    if data.empty:
        raise ValueError("No data fetched. Please check the ticker symbol and date range.")
    return data

# ===========================
# 2. Data Preprocessing
# ===========================

def preprocess_data(data, lag_days=10):
    """
    Preprocess the Bitcoin data by calculating daily returns and creating lag features.

    :param data: Pandas DataFrame with historical Bitcoin data.
    :param lag_days: Number of lag days to use as features.
    :return: Features (X), Targets (y), and the scaler object.
    """
    # Calculate daily returns
    data['Return'] = data['Adj Close'].pct_change()
    
    # Drop the first row with NaN return
    data = data.dropna()
    
    # Create lag features
    for i in range(1, lag_days + 1):
        data[f'lag_{i}'] = data['Return'].shift(i)
    
    # Drop rows with NaN values due to lagging
    data = data.dropna()
    
    # Define feature columns and target column
    feature_columns = [f'lag_{i}' for i in range(lag_days, 0, -1)]
    target_column = 'Return'
    
    X = data[feature_columns].values
    y = data[target_column].values
    
    return X, y, feature_columns

# ===========================
# 3. Train-Test Split and Scaling
# ===========================

def split_and_scale(X, y, test_size=0.2):
    """
    Split the data into training and testing sets and apply feature scaling.

    :param X: Feature matrix.
    :param y: Target vector.
    :param test_size: Proportion of the dataset to include in the test split.
    :return: Scaled and split data along with the scaler object.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

# ===========================
# 4. PyTorch Dataset and DataLoader
# ===========================

class BitcoinReturnsDataset(Dataset):
    """
    Custom Dataset for Bitcoin Returns.
    """
    def __init__(self, features, targets):
        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(targets, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def create_data_loaders(X_train, X_test, y_train, y_test, batch_size=64):
    """
    Create DataLoader objects for training and testing.

    :param X_train: Training features.
    :param X_test: Testing features.
    :param y_train: Training targets.
    :param y_test: Testing targets.
    :param batch_size: Batch size for DataLoader.
    :return: train_loader, test_loader
    """
    train_dataset = BitcoinReturnsDataset(X_train, y_train)
    test_dataset = BitcoinReturnsDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

# ===========================
# 5. Define the Deep Learning Model
# ===========================

class BitcoinReturnPredictor(nn.Module):
    """
    Neural Network model for predicting Bitcoin returns.
    """
    def __init__(self, input_size):
        super(BitcoinReturnPredictor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x)

# ===========================
# 6. Training Function
# ===========================

def train_model(model, train_loader, test_loader, criterion, optimizer, epochs=100, device='cpu'):
    """
    Train the neural network model.

    :param model: PyTorch model to train.
    :param train_loader: DataLoader for training data.
    :param test_loader: DataLoader for testing data.
    :param criterion: Loss function.
    :param optimizer: Optimizer.
    :param epochs: Number of training epochs.
    :param device: Device to run the training on ('cpu' or 'cuda').
    :return: Lists of training and testing losses.
    """
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
    """
    Evaluate the trained model on the test set.

    :param model: Trained PyTorch model.
    :param X_test: Scaled testing features.
    :param y_test: Testing targets.
    :param scaler: Fitted StandardScaler object.
    :param device: Device to run the evaluation on ('cpu' or 'cuda').
    :return: Predicted values and actual values.
    """
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
    """
    Plot training and testing loss over epochs.

    :param train_losses: List of training losses.
    :param test_losses: List of testing losses.
    """
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
    """
    Plot actual vs predicted returns.

    :param actuals: Actual return values.
    :param predictions: Predicted return values.
    """
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
    """
    Save the trained model's state dictionary.

    :param model: Trained PyTorch model.
    :param path: File path to save the model.
    """
    torch.save(model.state_dict(), path)
    print(f'Model saved to {path}')

def load_model(input_size, path='bitcoin_return_predictor.pth', device='cpu'):
    """
    Load the model's state dictionary.

    :param input_size: Number of input features.
    :param path: File path from where to load the model.
    :param device: Device to map the model to ('cpu' or 'cuda').
    :return: Loaded PyTorch model.
    """
    model = BitcoinReturnPredictor(input_size)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    print(f'Model loaded from {path}')
    return model

# ===========================
# 10. Making Future Predictions
# ===========================

def predict_next_return(model, recent_returns, scaler, device='cpu'):
    """
    Predict the next day's return given recent returns.

    :param model: Trained PyTorch model.
    :param recent_returns: List or array of recent returns (length should match lag_days).
    :param scaler: Fitted StandardScaler object.
    :param device: Device to run the prediction on ('cpu' or 'cuda').
    :return: Predicted return.
    """
    if len(recent_returns) != model.network[0].in_features:
        raise ValueError(f"Expected {model.network[0].in_features} recent returns, got {len(recent_returns)}")
    
    # Convert to numpy array and reshape
    recent_returns = np.array(recent_returns).reshape(1, -1)
    
    # Scale the features
    recent_returns_scaled = scaler.transform(recent_returns)
    
    # Convert to tensor
    input_tensor = torch.tensor(recent_returns_scaled, dtype=torch.float32).to(device)
    
    # Make prediction
    model.eval()
    with torch.no_grad():
        prediction = model(input_tensor).item()
    
    return prediction

# ===========================
# 11. Main Execution Flow
# ===========================

def main():
    # Parameters
    TICKER = 'BTC-USD'
    START_DATE = '2018-01-01'
    END_DATE = '2023-12-31'
    LAG_DAYS = 30
    TEST_SIZE = 0.2
    BATCH_SIZE = 264
    EPOCHS = 100
    LEARNING_RATE = 0.001
    MODEL_PATH = 'bitcoin_return_predictor.pth'
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f'Using device: {DEVICE}')
    
    # 1. Fetch Data
    print("Fetching Bitcoin data...")
    data = fetch_bitcoin_data(ticker=TICKER, start_date=START_DATE, end_date=END_DATE)
    print("Data fetched successfully.\n")
    
    # 2. Preprocess Data
    print("Preprocessing data...")
    X, y, feature_columns = preprocess_data(data, lag_days=LAG_DAYS)
    print(f'Features shape: {X.shape}')
    print(f'Target shape: {y.shape}\n')
    
    # 3. Split and Scale
    print("Splitting and scaling data...")
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = split_and_scale(X, y, test_size=TEST_SIZE)
    print(f'Training features shape: {X_train_scaled.shape}')
    print(f'Testing features shape: {X_test_scaled.shape}\n')
    
    # 4. Create DataLoaders
    print("Creating DataLoaders...")
    train_loader, test_loader = create_data_loaders(X_train_scaled, X_test_scaled, y_train, y_test, batch_size=BATCH_SIZE)
    print("DataLoaders created.\n")
    
    # 5. Initialize Model
    input_size = X_train_scaled.shape[1]
    model = BitcoinReturnPredictor(input_size)
    print(f'Model initialized with input size: {input_size}\n')
    
    # 6. Define Loss and Optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 7. Train the Model
    print("Starting training...")
    train_losses, test_losses = train_model(
        model, train_loader, test_loader, criterion, optimizer, epochs=EPOCHS, device=DEVICE
    )
    print("Training completed.\n")
    
    # 8. Plot Losses
    print("Plotting training and test losses...")
    plot_losses(train_losses, test_losses)
    
    # 9. Evaluate the Model
    print("Evaluating the model...")
    predictions, actuals = evaluate_model(model, X_test_scaled, y_test, scaler, device=DEVICE)
    
    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    
    print(f'\nEvaluation Metrics:')
    print(f'Mean Absolute Error (MAE): {mae:.6f}')
    print(f'Root Mean Squared Error (RMSE): {rmse:.6f}\n')
    
    # 10. Plot Predictions vs Actuals
    print("Plotting predictions vs actuals...")
    plot_predictions(actuals, predictions)
    
    # 11. Save the Model
    print("Saving the model...")
    save_model(model, path=MODEL_PATH)
    
    # 12. Example: Load the Model and Make a Prediction
    print("\nLoading the model and making an example prediction...")
    loaded_model = load_model(input_size, path=MODEL_PATH, device=DEVICE)
    
    # Get the latest lag_days returns from the original data
    last_returns = data['Return'].values[-LAG_DAYS:]
    
    # Predict the next return
    next_return = predict_next_return(loaded_model, last_returns, scaler, device=DEVICE)
    print(f'Predicted Next Return: {next_return:.6f}')

if __name__ == "__main__":
    main()

