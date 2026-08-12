import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)
import torch
from data_prepare import get_batch
from contextlib import nullcontext
import numpy as np
import math
import pickle
from training_config import Training_Config
from peft import get_peft_model_state_dict

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
                loss = outputs.loss
                logits = outputs.logits
            losses[k] = loss.item()
        out[split] = losses.mean()
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

def trainer(model, optimizer: torch.optim.Optimizer, config,iter_num=0,best_val_loss = float('inf')):
    local_iter_num = 0
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

                if losses["val"] < best_val_loss or always_save_checkpoint:
                    best_val_loss = losses["val"]

                    if iter_num > 0:
                        checkpoint = {
                            "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "iter_num": iter_num,
                            "best_val_loss": best_val_loss,
                            "config": config,
                        }

                        print(f"saving checkpoint to {Training_Config.checkpoint_dir}")
                        os.makedirs(Training_Config.checkpoint_dir, exist_ok=True)
                        torch.save(checkpoint, os.path.join(Training_Config.checkpoint_dir, "ckpt.pt"))

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
                    logits = outputs.logits
                    loss = loss / Training_Config.gradient_accumulation_steps

                X, Y = get_batch("train")

                scaler.scale(loss).backward()

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
        print("\n" + "="*60)
        checkpoint = {
            "model": get_peft_model_state_dict(model),
            "optimizer": optimizer.state_dict(),
            "iter_num": iter_num,
            "best_val_loss": best_val_loss,
            "config": config,
        }
        
        os.makedirs(Training_Config.checkpoint_dir, exist_ok=True)
        torch.save(checkpoint, os.path.join(Training_Config.checkpoint_dir, "ckpt.pt"))
        
        print(f"Đã lưu checkpoint")
        print("="*60 + "\n")