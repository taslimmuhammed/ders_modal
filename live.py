from web3 import Web3
from web3.middleware import ExtraDataToPOAMiddleware
import json
import time
from eth_account import Account
import os
from typing import List, Dict
from web3.contract import Contract

class ContractMonitor:
    def __init__(
        self, 
        rpc_url: str,
        contract_address: str,
        private_key: str,
        model,
        scalers,
        lookback_window: int = 7
    ):
        self.w3 = Web3(Web3.HTTPProvider(rpc_url))
        self.w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
        
        # Contract setup
        self.contract_address = Web3.to_checksum_address(contract_address)
        self.contract_abi = [
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
        self.contract = self.w3.eth.contract(address=self.contract_address, abi=self.contract_abi)
        
        # Account setup
        self.account = Account.from_key(private_key)
        print(f"Monitoring from address: {self.account.address}")
        
        # AI model setup
        self.model = model
        self.scalers = scalers
        self.lookback_window = lookback_window
        self.transaction_history = []
    
    def get_transaction_details(self, tx_hash: str) -> Dict:
        """Get transaction details and format them for the AI model"""
        tx = self.w3.eth.get_transaction(tx_hash)
        receipt = self.w3.eth.get_transaction_receipt(tx_hash)
        
        return {
            "amount": float(Web3.from_wei(tx.value, 'wei')),
            'gas_cost': receipt['gasUsed'],
            'actions_count': len(receipt['logs']),  # Using logs as a proxy for actions
            "contract_id": 1  # Adjust based on your needs
        }

    def analyze_transaction(self, transaction: Dict) -> float:
        """Analyze a single transaction using the AI model"""
        self.transaction_history.append(transaction)
        if len(self.transaction_history) > self.lookback_window:
            self.transaction_history.pop(0)
            
        # Only analyze if we have enough history
        if len(self.transaction_history) == self.lookback_window:
            results = test_transactions(
                self.model,
                self.scalers,
                [transaction],
                self.lookback_window
            )
            return results[0]['anomaly_score']
        return 0.0

    def implement_ai_pause(self, contract_id: int):
        """Call the ImplementAiPause function on the contract"""
        transaction = self.contract.functions.ImplementAiPause(contract_id).build_transaction({
            'from': self.account.address,
            'nonce': self.w3.eth.get_transaction_count(self.account.address),
            'gas': 200000,
            'gasPrice': self.w3.eth.gas_price
        })
        
        signed_txn = self.w3.eth.account.sign_transaction(transaction, self.account.key)
        tx_hash = self.w3.eth.send_raw_transaction(signed_txn.rawTransaction)
        print(f"Implementing AI Pause. Transaction hash: {tx_hash.hex()}")
        return tx_hash

    def monitor_address(self, address_to_monitor: str, anomaly_threshold: float = 0.5):
        """Monitor an address for transactions and analyze them"""
        address_to_monitor = Web3.to_checksum_address(address_to_monitor)
        last_block = self.w3.eth.block_number
        
        print(f"Starting monitoring of address: {address_to_monitor}")
        print(f"Current block: {last_block}")
        
        while True:
            try:
                current_block = self.w3.eth.block_number
                
                # Check new blocks
                for block_number in range(last_block + 1, current_block + 1):
                    block = self.w3.eth.get_block(block_number, full_transactions=True)
                    
                    # Filter transactions for monitored address
                    relevant_txs = [
                        tx for tx in block.transactions 
                        if tx['to'] and tx['to'].lower() == address_to_monitor.lower()
                    ]
                    
                    for tx in relevant_txs:
                        tx_details = self.get_transaction_details(tx['hash'])
                        anomaly_score = self.analyze_transaction(tx_details)
                        
                        print(f"\nAnalyzing transaction: {tx['hash'].hex()}")
                        print(f"Anomaly score: {anomaly_score:.4f}")
                        
                        # Implement pause if anomaly detected
                        if anomaly_score > anomaly_threshold:
                            print("Anomaly detected! Implementing AI Pause...")
                            self.implement_ai_pause(tx_details['contract_id'])
                
                last_block = current_block
                time.sleep(1)  # Wait for new blocks
                
            except Exception as e:
                print(f"Error occurred: {e}")
                time.sleep(1)  # Wait before retrying

# Usage example
if __name__ == "__main__":
    # Configuration
    RPC_URL = "https://polygon-amoy.g.alchemy.com/v2/YOUR-API-KEY"
    CONTRACT_ADDRESS = "0x37fE749Cd8823e6d24a0995a66A53e6d9e12D9BD"
    PRIVATE_KEY = "your-private-key"  # Owner's private key
    ADDRESS_TO_MONITOR = "address-to-monitor"
    
    # Initialize monitor
    monitor = ContractMonitor(
        rpc_url=RPC_URL,
        contract_address=CONTRACT_ADDRESS,
        private_key=PRIVATE_KEY,
        model=model,  # Your existing AI model
        scalers=scalers  # Your existing scalers
    )
    
    # Start monitoring
    monitor.monitor_address(ADDRESS_TO_MONITOR)