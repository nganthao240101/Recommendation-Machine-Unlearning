import pickle
import numpy as np
import os

# Test loading pretrain files
print("Testing loading pretrained files...")
with open('../data/ml-1m/user_pretrain.pk', 'rb') as f:
    uidW = pickle.load(f)
print('user_pretrain shape:', uidW.shape)

with open('../data/ml-1m/item_pretrain.pk', 'rb') as f:
    iidW = pickle.load(f)
print('item_pretrain shape:', iidW.shape)

# Check partition files
files = os.listdir('../data/ml-1m/')
pk_files = [f for f in files if '.pk' in f]
print('PK files in data dir:', pk_files)
print("Done!")