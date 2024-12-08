import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from torch.utils.data import Dataset, DataLoader
import json
import os
import pickle
from datetime import datetime
from collections import deque
from sklearn.model_selection import train_test_split

class WeiAwareDeFiGenerator:
    def __init__(self, seed=42):
        np.random.seed(seed)
        
        # Define wei transaction boundaries (1e-18 ETH)
        self.wei_ranges = {
            'amount': {
                'min': 1e-18,  # 1 wei
                'max': 1e-15,  # 1000 wei
                'mean': 1e-16,
                'std': 1e-17
            },
            'gas_cost': {
                'min': 21000,  # Base gas cost
                'max': 25000,
                'mean': 22000,
                'std': 1000
            }
        }
        
        # Define normal transaction boundaries
        self.normal_ranges = {
            'amount': {
                'min': 50,
                'max': 200,
                'mean': 120,
                'std': 30
            },
            'gas_cost': {
                'min': 21000,
                'max': 25000,
                'mean': 22000,
                'std': 1000
            }
        }
        
        # Enhanced anomaly patterns
        self.anomaly_patterns = {
            'wei_amount_spike': {
                'probability': 0.3,
                'multiplier_range': (1e3, 1e6)
            },
            'wei_amount_drop': {
                'probability': 0.3,
                'multiplier_range': (1e-6, 1e-3)
            },
            'gas_amount_mismatch': {
                'probability': 0.3,
                'threshold': 100  # Significant difference between gas and amount
            }
        }
    
    def generate_wei_transaction(self, contract_id=1):
        """Generate a transaction with wei-scale amounts"""
        amount = np.random.lognormal(
            mean=np.log(self.wei_ranges['amount']['mean']),
            sigma=0.5
        )
        amount = np.clip(
            amount,
            self.wei_ranges['amount']['min'],
            self.wei_ranges['amount']['max']
        )
        
        gas_cost = np.clip(
            np.random.normal(
                self.wei_ranges['gas_cost']['mean'],
                self.wei_ranges['gas_cost']['std']
            ),
            self.wei_ranges['gas_cost']['min'],
            self.wei_ranges['gas_cost']['max']
        )
        
        return {
            'amount': amount,
            'gas_cost': gas_cost,
            'contract_id': contract_id,
            'is_anomaly': False,
            'transaction_type': 'wei'
        }
    
    def generate_anomalous_wei_transaction(self, contract_id=1):
        """Generate anomalous wei-scale transaction"""
        tx = self.generate_wei_transaction(contract_id)
        tx['is_anomaly'] = True
        
        anomaly_type = np.random.choice(list(self.anomaly_patterns.keys()))
        
        if anomaly_type == 'wei_amount_spike':
            multiplier = np.random.uniform(*self.anomaly_patterns['wei_amount_spike']['multiplier_range'])
            tx['amount'] *= multiplier
        
        elif anomaly_type == 'wei_amount_drop':
            multiplier = np.random.uniform(*self.anomaly_patterns['wei_amount_drop']['multiplier_range'])
            tx['amount'] *= multiplier
        
        elif anomaly_type == 'gas_amount_mismatch':
            # Create suspicious mismatch between gas cost and transaction amount
            if np.random.random() < 0.5:
                tx['gas_cost'] *= 100  # Unusually high gas for small amount
            else:
                tx['amount'] *= 1e-6  # Unusually small amount for gas cost
        
        return tx

class EnhancedContractAwareLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.2):
        super(EnhancedContractAwareLSTM, self).__init__()
        
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
        
        # Enhanced feature processing layers
        self.feature_layers = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_dim, 64),
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
        
        # Enhanced feature processing
        features = self.feature_layers(context)
        return self.output_layers(features)

class ImprovedDeFiMonitor:
    def __init__(self, lookback_window=20, hidden_dim=128, num_layers=3):
        self.lookback_window = lookback_window
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scalers = {}
        self.model = None
        
        # Separate scalers for wei and normal transactions
        self.wei_scaler = RobustScaler()
        self.normal_scaler = StandardScaler()
    
    def calculate_transaction_metrics(self, df):
        """Calculate additional metrics for anomaly detection"""
        # Calculate gas to amount ratio (log scale for wei transactions)
        df['gas_amount_ratio'] = np.log1p(df['gas_cost']) - np.log1p(df['amount'])
        
        # Calculate rolling statistics
        df['amount_rolling_mean'] = df['amount'].rolling(window=10, min_periods=1).mean()
        df['amount_rolling_std'] = df['amount'].rolling(window=10, min_periods=1).std()
        df['gas_rolling_mean'] = df['gas_cost'].rolling(window=10, min_periods=1).mean()
        
        # Calculate z-scores
        df['amount_zscore'] = (df['amount'] - df['amount_rolling_mean']) / (df['amount_rolling_std'] + 1e-10)
        df['gas_amount_ratio_zscore'] = (df['gas_amount_ratio'] - df['gas_amount_ratio'].mean()) / (df['gas_amount_ratio'].std() + 1e-10)
        
        return df

    def preprocess_data(self, df):
        """Enhanced preprocessing with wei-aware scaling"""
        df = self.calculate_transaction_metrics(df)
        
        features = {}
        # Use log transformation for amount to handle wei-scale values
        df['log_amount'] = np.log1p(df['amount'])
        
        # Scale features separately for wei and normal transactions
        wei_mask = df['amount'] < 1e-10
        
        for col in ['log_amount', 'gas_cost', 'gas_amount_ratio']:
            if col not in self.scalers:
                self.scalers[col] = {}
                self.scalers[col]['wei'] = RobustScaler()
                self.scalers[col]['normal'] = StandardScaler()
            
            # Scale wei transactions
            wei_data = df.loc[wei_mask, col].values.reshape(-1, 1)
            normal_data = df.loc[~wei_mask, col].values.reshape(-1, 1)
            
            if len(wei_data) > 0:
                features[f'{col}_wei'] = self.scalers[col]['wei'].fit_transform(wei_data)
            if len(normal_data) > 0:
                features[f'{col}_normal'] = self.scalers[col]['normal'].fit_transform(normal_data)
        
        # Combine features
        X = np.hstack([np.vstack(features[f]) for f in features])
        
        # Create enhanced labels considering multiple factors
        labels = (
            (abs(df['amount_zscore']) > 3) |  # Unusual amount
            (abs(df['gas_amount_ratio_zscore']) > 3) |  # Unusual gas/amount ratio
            (df['gas_cost'] > df['gas_rolling_mean'] * 3)  # Unusual gas cost
        ).astype(float)
        
        return X, labels.values.reshape(-1, 1)

    def fit(self, df, epochs=100, batch_size=64, learning_rate=0.001):
        """Train the model with enhanced features"""
        X, y = self.preprocess_data(df)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        train_dataset = ContractAwareDataset(X_train, y_train, self.lookback_window)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        input_dim = X_train.shape[1]
        self.model = EnhancedContractAwareLSTM(input_dim, self.hidden_dim, self.num_layers)
        self.model.to(self.device)
        
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Training loop with early stopping
        best_loss = float('inf')
        patience = 5
        no_improve = 0
        
        for epoch in range(epochs):
            self.model.train()
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
            
            avg_loss = total_loss / len(train_loader)
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                no_improve = 0
            else:
                no_improve += 1
            
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')