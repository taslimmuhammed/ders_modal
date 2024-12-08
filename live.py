from web3 import Web3
from web3.middleware import ExtraDataToPOAMiddleware
from eth_account import Account
import time
from utils import decode_transaction_input, get_modal

# Initialize Web3 and contract
w3 = Web3(Web3.HTTPProvider("https://polygon-amoy.g.alchemy.com/v2/QEgC4Vsyb3fgPm90ENKlwx5X1-edSRT8"))
w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)

# Contract configuration
CONTRACT_ADDRESS = "0x37fE749Cd8823e6d24a0995a66A53e6d9e12D9BD"
PRIVATE_KEY = "YOUR_PRIVATE_KEY"  # Replace with your private key
ADDRESS_TO_MONITOR = "ADDRESS_TO_MONITOR"  # Replace with address to monitor

# Load AI model
model, scalers = get_modal("saved_models_v7")

# Contract ABI for the ImplementAiPause function
CONTRACT_ABI = [
    {
        "inputs": [
            {
                "internalType": "uint256",
                "name": "_contractId",
                "type": "uint256"
            }
        ],
        "name": "ImplementAiPause",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
    }
]

contract = w3.eth.contract(address=CONTRACT_ADDRESS, abi=CONTRACT_ABI)
account = Account.from_key(PRIVATE_KEY)

def get_transaction_details(tx_hash):
    """Get transaction details formatted for the AI model"""
    tx = w3.eth.get_transaction(tx_hash)
    receipt = w3.eth.get_transaction_receipt(tx_hash)
    
    return {
        "amount": float(Web3.from_wei(tx.value, 'wei')),
        'gas_cost': receipt['gasUsed'],
        'actions_count': decode_transaction_input(tx.input),
        "contract_id": 1
    }

def implement_ai_pause(contract_id):
    """Call the ImplementAiPause function"""
    try:
        transaction = contract.functions.ImplementAiPause(contract_id).build_transaction({
            'from': account.address,
            'nonce': w3.eth.get_transaction_count(account.address),
            'gas': 200000,
            'gasPrice': w3.eth.gas_price
        })
        
        signed_txn = w3.eth.account.sign_transaction(transaction, account.key)
        tx_hash = w3.eth.send_raw_transaction(signed_txn.rawTransaction)
        print(f"Implementing AI Pause. Transaction hash: {tx_hash.hex()}")
        return tx_hash
    except Exception as e:
        print(f"Error implementing AI pause: {e}")

def test_single_transaction(tx_details):
    """Test a single transaction using the existing model"""
    try:
        results = test_transactions(model, scalers, [tx_details], lookback_window=7)
        return results[0]['anomaly_score']
    except Exception as e:
        print(f"Error analyzing transaction: {e}")
        return 0.0

def monitor_address():
    """Monitor address for transactions and analyze them"""
    print(f"Starting monitoring of address: {ADDRESS_TO_MONITOR}")
    print(f"Monitoring from address: {account.address}")
    
    last_block = w3.eth.block_number
    print(f"Starting from block: {last_block}")
    
    while True:
        try:
            current_block = w3.eth.block_number
            
            # Check new blocks
            if current_block > last_block:
                print(f"Checking block {last_block + 1}")
                block = w3.eth.get_block(last_block + 1, full_transactions=True)
                
                # Filter transactions for monitored address
                for tx in block.transactions:
                    if tx['to'] and tx['to'].lower() == ADDRESS_TO_MONITOR.lower():
                        print(f"\nFound transaction: {tx['hash'].hex()}")
                        
                        # Analyze transaction
                        tx_details = get_transaction_details(tx['hash'])
                        anomaly_score = test_single_transaction(tx_details)
                        
                        print(f"Transaction details:")
                        print(f"Amount: {tx_details['amount']}")
                        print(f"Gas Cost: {tx_details['gas_cost']}")
                        print(f"Actions: {tx_details['actions_count']}")
                        print(f"Anomaly Score: {anomaly_score:.4f}")
                        
                        # Implement pause if anomaly detected
                        if anomaly_score > 0.5:
                            print("Anomaly detected! Implementing AI Pause...")
                            implement_ai_pause(tx_details['contract_id'])
                
                last_block = current_block
            
            time.sleep(1)  # Wait before checking next block
            
        except Exception as e:
            print(f"Error occurred: {e}")
            time.sleep(1)  # Wait before retrying

if __name__ == "__main__":
    monitor_address()