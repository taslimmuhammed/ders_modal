from web3 import Web3
from eth_account import Account
import json
import time
import random
from typing import List, Dict
import pandas as pd
from web3.middleware import ExtraDataToPOAMiddleware
def send_transaction(transaction,w3,account):
    transaction['nonce'] = w3.eth.get_transaction_count(account.address)
    transaction['gas'] = 2000000  # Adjust as needed
                
    # Get current fee data
    latest_block = w3.eth.get_block('latest')
    base_fee = latest_block['baseFeePerGas']
    priority_fee = w3.eth.max_priority_fee
                
    # Set EIP-1559 gas parameters
    transaction['maxPriorityFeePerGas'] = priority_fee
    transaction['maxFeePerGas'] = base_fee * 2 + priority_fee
    # Sign and send the transaction
    signed_txn = w3.eth.account.sign_transaction(transaction, account.key)
    tx_hash = w3.eth.send_raw_transaction(signed_txn.raw_transaction)
    # Wait for transaction receipt
    tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    return tx_receipt

def generate_attack_transaction(account,w3,contract_address  ) -> Dict:
    print(f"Using account: {account.address}")

    with open('TestDeFi.json', 'r') as f:
        contract_abi = json.load(f)

    contract = w3.eth.contract(
        address=w3.to_checksum_address(contract_address),
        abi=contract_abi
    )
    amount = int(random.uniform(500000, 10000000))
    old_price = contract.functions.getPrice().call()
    tx = contract.functions.deposit().build_transaction({
        'from': account.address,
        'value': int(amount)
    })
    receipt = send_transaction(tx,w3, account)



if __name__ == "__main__":
    w3 = Web3(Web3.HTTPProvider('https://polygon-amoy.g.alchemy.com/v2/QEgC4Vsyb3fgPm90ENKlwx5X1-edSRT8'))
    w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
    account = w3.eth.account.from_key("719a7252f1968da232963a591bf58877ba0f869a9387c952ac9cc7eb4295c1e0")
    generate_attack_transaction(account,w3,"0xB003e8931fAA6Be3406F38f60f890638E5059315")
    