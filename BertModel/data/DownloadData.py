import os
import socket

# Configuration
SERVER_IP = "69.30.85.98"  # Replace with real IP
SERVER_USER = "root"
SERVER_PATH = "/root/DeepSNP3"

# Windows paths
windows_source = r"C:\Users\lhfor\AppData\Local\Programs\Python\Python38-32\DataFetch\DeepGene3\Tokenized"
windows_dest = r"C:\Users\lhfor\AppData\Local\Programs\Python\Python38-32\DeepSNP3\BertModel\data\SNPdata"

# File names
negative_file = "negative_snp_sequences_2pct_tokenized.csv"
positive_file = "positive_snp_sequences_2pct_tokenized.csv"

print("=== SCP Commands (Run on Windows Command Prompt) ===")
print("\n1. UPLOAD TO SERVER:")
print(f'scp "{os.path.join(windows_source, negative_file)}" {SERVER_USER}@{SERVER_IP}:{SERVER_PATH}/')
print(f'scp "{os.path.join(windows_source, positive_file)}" {SERVER_USER}@{SERVER_IP}:{SERVER_PATH}/')

print("\n2. DOWNLOAD FROM SERVER:")
print(f'scp {SERVER_USER}@{SERVER_IP}:{SERVER_PATH}/{negative_file} "{windows_dest}\\"')
print(f'scp {SERVER_USER}@{SERVER_IP}:{SERVER_PATH}/{positive_file} "{windows_dest}\\"')

print("\n3. COPY MULTIPLE FILES:")
print(f'scp {SERVER_USER}@{SERVER_IP}:{SERVER_PATH}/*.csv "{windows_dest}\\"')

print("\n=== PowerShell Alternative (Windows) ===")
print(f'scp "{windows_source}\\*.csv" {SERVER_USER}@{SERVER_IP}:{SERVER_PATH}/')
