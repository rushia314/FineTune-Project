import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)
import numpy as np
import torch
from training_config import Training_Config
# Cấu hình cơ bản
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(project_dir,"data")
block_size = Training_Config.block_size
batch_size = Training_Config.batch_size
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_batch(split):
    batches_dir = os.path.join(data_dir,"train") if split == 'train' else os.path.join(data_dir,"val")
    ids_path = os.path.join(batches_dir, "ids.bin")
    labels_path = os.path.join(batches_dir, "labels.bin")
    ids_data = np.memmap(ids_path, dtype=np.int32, mode='r', offset=12)
    labels_data = np.memmap(labels_path, dtype=np.int32, mode='r', offset=12)
    ix = torch.randint(len(ids_data) - block_size, (batch_size,))

    x = torch.stack([torch.from_numpy((ids_data[i : i + block_size]).astype(np.int64)) for i in ix])
    
    y = torch.stack([torch.from_numpy((labels_data[i : i + block_size]).astype(np.int64)) for i in ix])
    
    x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
    
    return x, y
