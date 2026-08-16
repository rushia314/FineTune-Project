import sys
import os
project_root = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
import torch
from contextlib import nullcontext
import numpy as np
import math
import pickle
from Scripts.Train_Module.training_config import Training_Config
from peft import get_peft_model_state_dict
import gc
import random
import json
from config import Config
# SETTINGS --------------------------------------------------------------------------------------------
dtype = Training_Config.dtype
ptdtype = Training_Config.ptdtype
device = "cuda" if torch.cuda.is_available() else "cpu"
ctx = nullcontext() if device == 'cpu' else torch.amp.autocast(device_type=device, dtype=ptdtype)
max_iters = Training_Config.max_iters
grad_clip = Training_Config.grad_clip
decay_lr = Training_Config.decay_lr
warmup_iters = Training_Config.warmup_iters
lr_decay_iters = Training_Config.lr_decay_iters
learning_rate = Training_Config.learning_rate
min_lr = learning_rate / 10.0

eval_interval = Training_Config.eval_interval
eval_iters = Training_Config.eval_iters
eval_only = Training_Config.eval_only

always_save_checkpoint = Training_Config.always_save_checkpoint
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
# ----------------------------------------------------------------------------------------------------

def get_batch(split):
    data_dir = os.path.join(project_root,"data")
    block_size = Training_Config.block_size
    batch_size = Training_Config.batch_size
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
@torch.no_grad()
def estimate_loss(model, config):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(config.eval_iters)
        for k in range(config.eval_iters):
            X, Y = get_batch(split)
            with ctx:
                
                outputs = model(
                    input_ids=X,
                    labels=Y,
                    attention_mask=None,
                    position_ids=None,
                    past_key_values=None,
                )
            losses[k] = outputs.loss.item()
            del outputs
        out[split] = losses.mean()
        del losses
    model.train()
    
    return out

def get_lr(it):
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)
def get_metadata_examples(tokenizer, n=20):
    examples = []

    while len(examples) < n:
        X, Y = get_batch("train")

        X = X.detach().cpu()

        for ids in X:
            text = tokenizer.decode(
                ids.tolist(),
                skip_special_tokens=False
            )

            examples.append(text)

            if len(examples) >= n:
                break

    return examples
def trainer(model, optimizer: torch.optim.Optimizer,tokenizer, config,iter_num=0,best_val_loss = float('inf')):
    cur_adapter = model.active_adapter
    print(cur_adapter)
    local_iter_num = 0
    if not os.path.exists(os.path.join(Training_Config.checkpoint_dir,cur_adapter)):
        os.makedirs(os.path.join(Training_Config.checkpoint_dir,cur_adapter))
    X, Y = get_batch("train")
    try:
        while True:
            lr = get_lr(iter_num) if decay_lr else learning_rate
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
            
            if iter_num % eval_interval == 0 and eval_only == False:
                losses = estimate_loss(model, config)

                print(
                    f"step {iter_num}: "
                    f"train loss {losses['train']:.4f}, "
                    f"val loss {losses['val']:.4f}"
                )
                checkpoint = {
                            "model": get_peft_model_state_dict(model,adapter_name=cur_adapter),
                            "optimizer": optimizer.state_dict(),
                            "iter_num": iter_num,
                            "best_val_loss": best_val_loss,
                            "config": config,
                            "adapter_name" : cur_adapter,
                            "dataset_name": Config.dataset,
                            }
                
                torch.save(checkpoint, os.path.join(Training_Config.checkpoint_dir,cur_adapter, "latest_ckpt.pt"))
                
                if losses["val"] < best_val_loss or always_save_checkpoint:
                    best_val_loss = losses["val"]

                    if iter_num > 0:
                        print(f"saving checkpoint to {os.path.join(Training_Config.checkpoint_dir,cur_adapter)}")
                        torch.save(checkpoint, os.path.join(Training_Config.checkpoint_dir,cur_adapter, "best_ckpt.pt"))
                del checkpoint
                gc.collect()
                torch.cuda.empty_cache()
            
            if iter_num == 0 and eval_only:
                losses = estimate_loss(model, config)
                print(f"eval_only mode -> train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
                break
            
            optimizer.zero_grad(set_to_none=True)
            
            for micro_step in range(Training_Config.gradient_accumulation_steps):
                with ctx:
                    outputs = model(
                        input_ids=X,
                        labels=Y,
                        attention_mask=None,
                        position_ids=None,
                        past_key_values=None,
                    )
                    loss = outputs.loss
                    loss = loss / Training_Config.gradient_accumulation_steps

                scaler.scale(loss).backward()
                del outputs
                X, Y = get_batch("train")


            
            if grad_clip != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            # Cập nhật trọng số Optimizer
            scaler.step(optimizer)
            scaler.update()

            
            # 4. Logging
            if iter_num % Training_Config.log_interval == 0:
                lossf = loss.item() * Training_Config.gradient_accumulation_steps
                print(f"iter {iter_num}: loss {lossf:.4f}")

            iter_num += 1
            local_iter_num += 1

            if iter_num > max_iters:
                print("Đã đạt số lượng max_iters, kết thúc training!")
                break
    except KeyboardInterrupt:
        losses = estimate_loss(model, config)
        
        print(
            f"step {iter_num}: "
            f"train loss {losses['train']:.4f}, "
            f"val loss {losses['val']:.4f}"
        )
        
        print("\n" + "="*60)
        checkpoint = {
            "model": get_peft_model_state_dict(model,adapter_name=cur_adapter),
            "optimizer": optimizer.state_dict(),
            "iter_num": iter_num,
            "best_val_loss": best_val_loss,
            "config": config,
            "adapter_name" : cur_adapter,
            "dataset_name": Config.dataset,
        }
        
        print(f"saving checkpoint to {os.path.join(Training_Config.checkpoint_dir,cur_adapter)}")
        torch.save(checkpoint, os.path.join(Training_Config.checkpoint_dir,cur_adapter, "latest_ckpt.pt"))
        if losses["val"] < best_val_loss:
            torch.save(checkpoint, os.path.join(Training_Config.checkpoint_dir,cur_adapter, "best_ckpt.pt"))
        del checkpoint
        gc.collect()
        torch.cuda.empty_cache()
        print(f"Đã lưu checkpoint")
        print("="*60 + "\n")