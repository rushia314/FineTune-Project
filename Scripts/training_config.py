import torch
from data_prepare import get_batch
from contextlib import nullcontext
import numpy as np
import math
from dataclasses import dataclass
@dataclass
class Training_Config:
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    max_iters = 2000
    grad_clip = 1.0
    decay_lr = True
    warmup_iters = 0
    lr_decay_iters = max_iters
    learning_rate = 6e-4
    eval_interval = 1
    eval_iters = 20
    eval_only = False
    checkpoint_dir = "E:/LlamaFineTune/Checkpoint"
    gradient_accumulation_steps=1
    log_interval = 1
    always_save_checkpoint = False