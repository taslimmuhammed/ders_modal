from model import ContractAwareLSTM
import torch
import numpy as np
import pandas as pd
import pickle
import os
from collections import deque
from web3.middleware import ExtraDataToPOAMiddleware
from utils import decode_transaction_input, get_modal
from web3 import Web3

# def prepare_sequence(transactions, scalers, lookback_window=7):
#     """Prepare transaction sequence with sliding window"""
#     history = deque(maxlen=lookback_window)
    
#     # Initialize with base transactions
#     for tx in transactions[:lookback_window]:
#         history.append(tx)
    
#     sequences = []
#     for i in range(lookback_window, len(transactions)):
#         # Prepare features for the window
#         window_data = []
#         for tx in history:
#             # Scale each feature using provided scalers
#             scaled_features = []
#             for key in ['amount', 'gas_cost', 'actions_count']:
#                 value = np.array([[tx[key]]])
#                 scaled_value = scalers[key].transform(value)[0][0]
#                 scaled_features.append(scaled_value)
#             window_data.append(scaled_features)
        

#         # Convert to tensor and reshape for LSTM input
#         sequence = torch.FloatTensor(window_data)
#         sequence = sequence.view(1, lookback_window, 3)  # [batch_size, sequence_length, features]
#         sequences.append(sequence)
#         print(sequence)
#         # Update history
#         history.append(transactions[i])
    
#     return sequences

# def analyze_transactions(model, scalers, transactions, base_window=7):
#     """Analyze transactions using the provided model and scalers"""
#     # Prepare sequences for prediction
#     sequences = prepare_sequence(transactions, scalers, base_window)
    
#     # Make predictions
#     results = []
#     model.eval()  # Set model to evaluation mode
    
#     with torch.no_grad():
#         for i, sequence in enumerate(sequences):
#             transaction_idx = i + base_window
#             outputs = model.forward(sequence)  # Call forward method explicitly
            
#             # Handle different possible output formats
#             if isinstance(outputs, tuple):
#                 anomaly_score = outputs[0].item()  # Take first element if tuple
#             else:
#                 anomaly_score = outputs.item()
            
#             results.append({
#                 'transaction_hash': hashes[transaction_idx],
#                 'transaction_data': {
#                     'amount': transactions[transaction_idx]['amount'],
#                     'gas_cost': transactions[transaction_idx]['gas_cost'],
#                     'actions_count': transactions[transaction_idx]['actions_count']
#                 },
#                 'anomaly_score': anomaly_score,
#                 'is_anomaly': anomaly_score > 0.5
#             })
    
#     return results
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
        history.append(tx)
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
def get_transactions(hashes):
    w3 = Web3(Web3.HTTPProvider("https://polygon-amoy.g.alchemy.com/v2/QEgC4Vsyb3fgPm90ENKlwx5X1-edSRT8"))
    w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
    transactions = []
    for tx_hash in hashes:
        tx = w3.eth.get_transaction(tx_hash)
        receipt = w3.eth.get_transaction_receipt(tx.hash)
        transactions.append({
            "amount": float(Web3.from_wei(tx.value, 'wei')),
            'gas_cost': receipt['gasUsed'],
            'actions_count': decode_transaction_input(tx.input),
            "contract_id": 1
        })
    return transactions
# Main execution
hashes = [
    "0x7c18bd9a3ca9fb90f7f4dddb2f7f65740387488ea45772f9958936eadbd42c02",
    "0x0aef2658093b0ba7338f203ca352ed56c90150d5de2c7f26d6984c9a5cdb0a12",
    "0xe322720f4d3cbebf947dd41c5c41087c4573eeabf2283ab99a24e87ff06cbf74",
    "0xe6a462f081e6cc097a73323e252d232a7f4c0a75b43715d2a5cdca28496b9d96",
    "0x47c5143c4e4ff8e8ca43ab8148715f26dbf6c7e3c46ccff73575fe93aa9e5cf8",
    "0xe4c0de59bf6b30c91df3c191526a5cc7be84ebbff1b2676a45a9f8ae9d52dbd4",
    "0x6f0d96cf8352a08b4226970aa2bbf84434390fc993d7e518a98130be3bac2864",
    "0xa70f132b81bfc1816fa76ef5bcc253d5ec472ca026b4f82eba41163f1dd3d897",
    "0x6dbab242f740e83e3e15e333d1dc460abf3f157f4e8cf2793725d26305dccc98",
    "0x068c0c48f604bbd46aff40289df6b0f5a81fe3ce148be1a5cf320829fa1bb52a",
    "0x0d1b9cbe5aacc8c6f6259c3a3503f05e5a14c2de4f4bae8d221bdde7f431234a",
    "0xe1b2aa9331d11dbdbdc1d14ec1f3be181604a6486c6288706aa33b14bf2a4ca6",
]


model, scalers = get_modal("saved_models_v7")
transactions = get_transactions(hashes)
results = test_transactions(model,scalers, transactions, 7)
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