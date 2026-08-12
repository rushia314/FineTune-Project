import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import torch.nn as nn
import bitsandbytes as bnb
import gc
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model,set_peft_model_state_dict,prepare_model_for_kbit_training
from config import Config
from tqdm import tqdm
from Scripts.train import trainer, estimate_loss, get_lr
from Scripts.training_config import Training_Config
import logging
from collections import Counter
import subprocess
from Scripts.data_prepare import get_batch
logging.getLogger("bitsandbytes").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
project_path = os.path.dirname(os.path.abspath(__file__))

lora_config = Config.lora_config
model_using = Config.model_id
model_save_path = rf"{project_path}\{model_using}"
profiles_path = rf"{project_path}\user_profile"
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,  
    bnb_4bit_quant_type="nf4",        
    bnb_4bit_use_double_quant=True,      
)

def print_trainable_parameters(model):
    trainable_params = 0
    all_params = 0
    for params in model.parameters():
        all_params += params.numel()
        if params.requires_grad:
            trainable_params += params.numel()
    print(f"trainable params: {trainable_params} || all params: {all_params}")

class CastOutputToFloat(nn.Sequential):
    def forward(self,x):
        return super().forward(x).to(torch.float32)
    
tokenizer = AutoTokenizer.from_pretrained(model_save_path)
training_from = "resume"
if __name__ == "__main__":
    model = AutoModelForCausalLM.from_pretrained(
                                                model_save_path,
                                                device_map="auto",
                                                quantization_config=quantization_config,
                                                )
    model = prepare_model_for_kbit_training(model)

    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False
    model.lm_head = CastOutputToFloat(model.lm_head)
    model = get_peft_model(model,lora_config)

    print("Model Sẵn sàng!\n")
    print_trainable_parameters(model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Bắt đầu quy trình trên thiết bị: {device}")

    decay_params = []
    nodecay_params = []
    
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim >= 2:
            decay_params.append(p)
        else:
            nodecay_params.append(p)
            
    optim_groups = [
        {'params': decay_params, 'weight_decay': 0.01},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    optimizer = bnb.optim.AdamW8bit(
                                    optim_groups, 
                                    lr=Training_Config.learning_rate, 
                                    betas=(0.9, 0.95),
                                    eps=1e-8
                                    )

    ckpt_path = os.path.join(Training_Config.checkpoint_dir, "ckpt.pt")
    start_iter = 0
    best_loss = float('inf')

    if os.path.exists(ckpt_path) and training_from == "resume":
        print(f"Đang tải checkpoint từ {ckpt_path}...")
        
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        state_dict = checkpoint['model']
        set_peft_model_state_dict(model, state_dict)
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_iter = checkpoint["iter_num"]
        best_loss = checkpoint["best_val_loss"]
        
        del checkpoint
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        print(f"Đã khôi phục thành công! Sẵn sàng train tiếp từ step {start_iter}")
    else:
        print("Không có checkpoint cũ, bắt đầu train mới từ đầu!")

    #Has to warmup a forward first so everything is set up before the huge workload
    #Spent fricking 4 hours to figure this out
    print("WARMUP...")
    model.eval()

    X, Y = get_batch("train")

    with torch.no_grad():
        outputs = model(
            input_ids=X,
            labels=Y,
            attention_mask=None,
            position_ids=None,
            past_key_values=None,
        )

    del outputs, X, Y

    model.train()

    torch.cuda.synchronize()

    print("WARMUP DONE")
    trainer(
        model=model, 
        optimizer=optimizer, 
        config=Training_Config, 
        iter_num=start_iter,       
        best_val_loss=best_loss    
    )