from model import ContractAwareLSTM
import torch
import numpy as np
import pandas as pd
import pickle
import os
from collections import deque

def load_model(model_path, scalers_path):
    checkpoint = torch.load(model_path,map_location=torch.device('cpu'))
    
    model = ContractAwareLSTM(
        input_dim=3,
        hidden_dim=checkpoint['hidden_dim'],
        num_layers=checkpoint['num_layers']
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    with open(scalers_path, 'rb') as f:
        scalers = pickle.load(f)
    
    return model, scalers

def test_transactions(model, scalers, transactions, lookback_window=20):
    history = deque(maxlen=lookback_window)
    normal_transaction = {
        'amount': 100,
        'gas_cost': 21000,
        'actions_count': 1,
        'contract_id': 1
    }
    
    # Fill history with normal transactions
    for _ in range(lookback_window):
        history.append(normal_transaction.copy())
    
    results = []
    
    for tx in transactions:
        # Add current transaction to history
        history.append(tx)
        
        # Prepare features
        features = {}
        history_list = list(history)
        
        for col in ['amount', 'gas_cost', 'actions_count']:
            values = np.array([t[col] for t in history_list]).reshape(-1, 1)
            features[col] = scalers[col].transform(values)
        
        # Combine features
        X = np.hstack([features[f] for f in features])
        X = torch.FloatTensor(X).unsqueeze(0)  # Add batch dimension
        
        # Get prediction
        with torch.no_grad():
            prediction = model(X)
            anomaly_score = prediction.item()
        
        results.append({
            'transaction': tx,
            'anomaly_score': anomaly_score,
            'is_anomaly': anomaly_score > 0.5  # Using 0.5 as default threshold
        })
    
    return results

def main():
    # Find the latest saved model
    model_dir = "saved_models_v6"
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.pt')]
    latest_model = max(model_files)
    model_path = os.path.join(model_dir, latest_model)
    scalers_path = model_path.replace('.pt', '_scalers.pkl')
    
    model, scalers = load_model(model_path, scalers_path)
    
    transactions = [
        # 9 Normal transactions
        {
            'amount': 100, #1
            'gas_cost': 21000,
            'actions_count': 1,
            'contract_id': 1
        },
        {
            'amount': 120, #2
            'gas_cost': 22000,
            'actions_count': 1,
            'contract_id': 1
        },
        {
            'amount': 95, #3
            'gas_cost': 21500,
            'actions_count': 1,
            'contract_id': 1
        },
        {
            'amount': 150,#4
            'gas_cost': 23000,
            'actions_count': 2,
            'contract_id': 1
        },
        {
            'amount': 130,#5
            'gas_cost': 22500,
            'actions_count': 1,
            'contract_id': 1
        },
        {
            'amount': 110,#6
            'gas_cost': 21800,
            'actions_count': 1,
            'contract_id': 1
        },
        {
            'amount': 140,#7
            'gas_cost': 23500,
            'actions_count': 2,
            'contract_id': 1
        },
        {
            'amount': 125,#8
            'gas_cost': 22000,
            'actions_count': 1,
            'contract_id': 1
        },
        {
            'amount': 115,#9
            'gas_cost': 21500,
            'actions_count': 1,
            'contract_id': 1
        },
        # Anomalous transactions start here
        {
            'amount': 100000000000000000,  #10 Sudden high amount
            'gas_cost': 21000,
            'actions_count': 1,
            'contract_id': 1
        },
        {
            'amount': 100, #11
            'gas_cost': 500000,  # High gas cost
            'actions_count': 1,
            'contract_id': 1
        },
        {
            'amount': 0.001,  #12 Very low amount
            'gas_cost': 21000,
            'actions_count': 1,
            'contract_id': 1
        },
        {
            'amount': 5000, #13
            'gas_cost': 300000,
            'actions_count': 5,  # Multiple actions
            'contract_id': 1
        },
        {
            'amount': 20000,  # Very high amount
            'gas_cost': 600000,  # Very high gas
            'actions_count': 8,  # Many actions
            'contract_id': 1
        },
        {
            'amount': 0.0001,  # Extremely low amount
            'gas_cost': 19000,  # Unusually low gas
            'actions_count': 1,
            'contract_id': 1
        },
        {
            'amount': 8000,
            'gas_cost': 400000,
            'actions_count': 4,
            'contract_id': 2  # Different contract
        }
    ]
    
    results = test_transactions(model, scalers, transactions)
    
    # Print results
    print("\nTransaction Analysis Results:")
    print("=" * 50)
    for i, result in enumerate(results, 1):
        tx = result['transaction']
        print(f"\nTransaction {i}:")
        print(f"Amount: {tx['amount']}")
        print(f"Gas Cost: {tx['gas_cost']}")
        print(f"Actions: {tx['actions_count']}")
        print(f"Contract ID: {tx['contract_id']}")
        print(f"Anomaly Score: {result['anomaly_score']:.4f}")
        print(f"Is Anomaly: {result['is_anomaly']}")
        print("-" * 30)

if __name__ == "__main__":
    main()