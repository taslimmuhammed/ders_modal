import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import Dataset, DataLoader
import json
import os
import pickle
from datetime import datetime
from collections import deque
from sklearn.model_selection import train_test_split

class DeFiTransactionGenerator:
    def __init__(self, seed=42):
        np.random.seed(seed)

    def generate_contract_behavior(self):
        """Generate different types of contract behaviors"""
        behaviors = {
            1: {
                'amount_range': (1, 50),
                'gas_base': 21000,
                'gas_variance': 5000,
                'action_range': (1, 2),
                'burst_probability': 0.1,
                'pattern': 'frequent_small'
            },
            2: {
                'amount_range': (100, 1000),
                'gas_base': 50000,
                'gas_variance': 30000,
                'action_range': (1, 4),
                'burst_probability': 0.2,
                'pattern': 'dex_like'
            },
            3: {
                'amount_range': (1000, 10000),
                'gas_base': 100000,
                'gas_variance': 50000,
                'action_range': (2, 5),
                'burst_probability': 0.05,
                'pattern': 'lending'
            },
            4: {
                'amount_range': (10, 5000),
                'gas_base': 80000,
                'gas_variance': 40000,
                'action_range': (1, 3),
                'burst_probability': 0.15,
                'pattern': 'variable'
            },
            5: {
                'amount_range': (50, 2000),
                'gas_base': 150000,
                'gas_variance': 70000,
                'action_range': (3, 8),
                'burst_probability': 0.3,
                'pattern': 'high_frequency'
            }
        }
        return behaviors

    def generate_dataset(self, n_samples=10000, n_contracts=5):
        """Generate complete dataset with normal and anomalous transactions"""
        data = []
        behaviors = self.generate_contract_behavior()

        for contract_id in range(1, n_contracts + 1):
            behavior = behaviors[contract_id]
            n_contract_samples = n_samples // n_contracts
            current_sequence = []

            for _ in range(n_contract_samples):
                if np.random.random() < 0.05:  # Anomaly
                    tx = self.generate_anomaly(behavior)
                else:
                    tx = self.generate_normal_transaction(behavior)

                tx['contract_id'] = contract_id
                data.append(tx)

        df = pd.DataFrame(data)
        print(df.head())
        return df.sample(frac=1).reset_index(drop=True)

    def generate_normal_transaction(self, behavior):
        """Generate normal transaction"""
        amount = np.random.uniform(*behavior['amount_range'])
        gas_cost = behavior['gas_base'] + np.random.normal(0, behavior['gas_variance'])
        gas_cost = max(21000, int(gas_cost))
        actions = np.random.randint(*behavior['action_range'])

        return {
            'amount': amount,
            'gas_cost': gas_cost,
            'actions_count': actions
        }

    def generate_anomaly(self, behavior):
        """Generate anomalous transaction"""
        tx = self.generate_normal_transaction(behavior)
        anomaly_type = np.random.choice(['amount', 'gas', 'both'])

        if anomaly_type in ['amount', 'both']:
            multiplier = np.random.choice([100, 0.0001])
            tx['amount'] *= multiplier

        if anomaly_type in ['gas', 'both']:
            tx['gas_cost'] *= np.random.uniform(5, 10)

        return tx

class ContractAwareDataset(Dataset):
    def __init__(self, data, labels=None, lookback_window=20):
        self.data = torch.FloatTensor(data)
        self.labels = torch.FloatTensor(labels) if labels is not None else None
        self.lookback_window = lookback_window

    def __len__(self):
        return len(self.data) - self.lookback_window

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.lookback_window]
        if self.labels is not None:
            y = self.labels[idx + self.lookback_window - 1]
            return x, y
        return x, torch.zeros(1)  # Default to 0 if no labels

class ContractAwareLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.2):
        super(ContractAwareLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout, bidirectional=True
        )

        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attention_weights = self.attention(lstm_out)
        context = torch.sum(attention_weights * lstm_out, dim=1)
        return self.fc(context)


class EnhancedDeFiMonitor:
    def __init__(self, lookback_window=20, hidden_dim=128, num_layers=3, test_size=0.2):
        self.lookback_window = lookback_window
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.test_size = test_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scalers = {}
        self.model = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None

    def preprocess_data(self, df):
        """Preprocess data and split into train/test sets"""
        features = {}
        # Normalize basic features
        for col in ['amount', 'gas_cost', 'actions_count']:
            if col not in self.scalers:
                self.scalers[col] = MinMaxScaler(feature_range=(0, 1))
                features[col] = self.scalers[col].fit_transform(df[col].values.reshape(-1, 1))
            else:
                features[col] = self.scalers[col].transform(df[col].values.reshape(-1, 1))

        # Create labels (1 for anomalies, 0 for normal transactions)
        amount_mean = df['amount'].mean()
        amount_std = df['amount'].std()
        gas_mean = df['gas_cost'].mean()
        gas_std = df['gas_cost'].std()

        labels = ((abs(df['amount'] - amount_mean) > 3 * amount_std) |
                 (abs(df['gas_cost'] - gas_mean) > 3 * gas_std)).astype(float)

        # Combine features
        X = np.hstack([features[f] for f in features])
        y = labels.values.reshape(-1, 1)

        # Split into train and test sets
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=42
        )

        return self.x_train, self.y_train

    def fit(self, df, epochs=100, batch_size=64, learning_rate=0.001):
        """Train the model with proper train/test split"""
        X_train, y_train = self.preprocess_data(df)

        train_dataset = ContractAwareDataset(X_train, y_train, self.lookback_window)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        input_dim = X_train.shape[1]
        self.model = ContractAwareLSTM(input_dim, self.hidden_dim, self.num_layers)
        self.model.to(self.device)

        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}')

    def evaluate(self):
        """Evaluate the model on test data"""
        if self.model is None or self.x_test is None:
            raise ValueError("Model must be trained first")

        self.model.eval()
        test_dataset = ContractAwareDataset(self.x_test, self.y_test, self.lookback_window)
        test_loader = DataLoader(test_dataset, batch_size=64)

        predictions = []
        actuals = []

        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(self.device)
                outputs = self.model(batch_x)
                predictions.extend(outputs.cpu().numpy())
                actuals.extend(batch_y.numpy())

        return np.array(predictions), np.array(actuals)

class TransactionFeatureExtractor:
    def __init__(self, lookback_window=10):
        self.lookback_window = lookback_window
        self.history = deque(maxlen=lookback_window)

    def calculate_metrics(self, series):
        """Calculate statistical metrics"""
        if len(series) < 2:
            return 0, 0, 0

        mean_val = np.mean(series[:-1])
        std_val = np.std(series[:-1]) if len(series) > 2 else 0
        current_val = series[-1]

        z_score = (current_val - mean_val) / (std_val + 1e-10)
        pct_change = ((current_val - mean_val) / (mean_val + 1e-10)) * 100

        return z_score, pct_change

    def extract_features(self, transaction):
        """Extract features from transaction"""
        self.history.append(transaction)

        amounts = np.array([tx['amount'] for tx in self.history])
        gas_costs = np.array([tx['gas_cost'] for tx in self.history])

        amount_z, amount_pct = self.calculate_metrics(amounts)
        gas_z, gas_pct = self.calculate_metrics(gas_costs)

        risk_score = max(
            min(abs(amount_z) / 3, 1),  # Scale to 0-1
            min(abs(gas_z) / 3, 1),
            min(abs(amount_pct) / 1000, 1),
            min(abs(gas_pct) / 100, 1)
        )

        return {
            'amount_z_score': amount_z,
            'amount_pct_change': amount_pct,
            'gas_z_score': gas_z,
            'gas_pct_change': gas_pct,
            'risk_score': risk_score
        }