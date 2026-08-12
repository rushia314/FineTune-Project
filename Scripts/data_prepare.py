import numpy as np
import torch
import os

# Cấu hình cơ bản
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(project_dir,"data")
block_size = 256 # Chiều dài tối đa của một câu (Context Window)
batch_size = 4  # Số lượng câu ném vào GPU mỗi lần
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_batch(split):
    # Chọn file dựa trên split
    bin_file = 'train.bin' if split == 'train' else 'val.bin'
    file_path = os.path.join(data_dir, bin_file)
    
    #thêm offset=12 để né cái Header, và dtype=np.int32 để không tràn từ vựng Llama
    data = np.memmap(file_path, dtype=np.int32, mode='r', offset=12)
    
    # Sinh ngẫu nhiên batch_size vị trí bắt đầu
    ix = torch.randint(len(data) - block_size, (batch_size,))
    
    # Cắt dữ liệu thành Input (x) và Target (y)
    x = torch.stack([torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix])
    
    y = torch.stack([torch.from_numpy((data[i + 1 : i + 1 + block_size]).astype(np.int64)) for i in ix])
    
    x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
    
    return x, y
