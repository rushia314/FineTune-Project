import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
import bitsandbytes as bnb
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from config import Config
from tqdm import tqdm
from Scripts.train import trainer, estimate_loss, get_lr
from Scripts.training_config import Training_Config
import logging
logging.getLogger("bitsandbytes").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
project_path = os.path.dirname(os.path.abspath(__file__))

lora_config = Config.lora_config
model_using = Config.model_id
model_save_path = rf"{project_path}\{model_using}"
profiles_path = rf"{project_path}\user_profile"
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

if __name__ == "__main__":
    model = AutoModelForCausalLM.from_pretrained(
                                                model_save_path,
                                                device_map="auto",
                                                dtype = torch.float16,
                                                )

    print("Freezing parameters: ")
    for param in tqdm(model.parameters()):
        param.requires_grad = False
        if param.ndim == 1:
            param.data = param.data.to(torch.float32)

    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
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
    optimizer = torch.optim.AdamW(
        optim_groups, 
        lr=Training_Config.learning_rate, 
        betas=(0.9, 0.95),
        eps=1e-8
    )
    if not os.path.exists(Training_Config.checkpoint_dir):
        ckpt_path = os.path.join(Training_Config.checkpoint_dir, "ckpt.pt")

    
    ckpt_path = os.path.join(Training_Config.checkpoint_dir, "ckpt.pt")
    start_iter = 0
    best_loss = float('inf')

    if os.path.exists(ckpt_path):
        print(f"Đang tải checkpoint từ {ckpt_path}...")
        checkpoint = torch.load(ckpt_path, map_location=device,weights_only = False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_iter = checkpoint["iter_num"]
        best_loss = checkpoint["best_val_loss"]
        
        print(f"Đã khôi phục thành công! Sẵn sàng train tiếp từ step {start_iter}")
    else:
        print("Không có checkpoint cũ, bắt đầu train mới từ đầu!")

    trainer(
        model=model, 
        optimizer=optimizer, 
        config=Training_Config, 
        iter_num=start_iter,       
        best_val_loss=best_loss    
    )