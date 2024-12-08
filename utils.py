import os
import pickle
import torch
from model import ContractAwareLSTM

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

def get_modal(dir:str):
    model_dir = dir
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.pt')]
    latest_model = max(model_files)
    model_path = os.path.join(model_dir, latest_model)
    scalers_path = model_path.replace('.pt', '_scalers.pkl')
    
    # Load model and scalers
    model, scalers = load_model(model_path, scalers_path)
    return model, scalers

def decode_transaction_input(input_data):
    """
    Attempt to decode transaction input data without ABI
    Returns number of likely method calls based on input data structure
    """
    if input_data == '0x':
        return 1  # Simple transfer
        
    try:
        # Get method signature (first 4 bytes after 0x)
        method_sig = input_data[:10]
        # Count potential parameter blocks (each 32 bytes)
        param_blocks = (len(input_data[10:]) // 64)
        return param_blocks + 1  # Add 1 for the method call itself
    except:
        return 1  # Default to 1 if unable to decode
            