import torch
import os
from peft import get_peft_model,set_peft_model_state_dict
from config import Config
from transformers import AutoModelForCausalLM
from Scripts.training_config import Training_Config
cur_dir = os.path.dirname(os.path.abspath(__file__))
model_save_path = os.path.join(cur_dir,Config.model_id)
cur_adapter = "author_guess"
ckpt_path = os.path.join(Training_Config.checkpoint_dir,cur_adapter,"best_ckpt.pt")
if os.path.exists(ckpt_path):
    if not os.path.exists(os.path.join(cur_dir,"adapters")):
        os.makedirs(os.path.join(cur_dir,"adapters"))
    adapter_path = os.path.join(cur_dir,"adapters",cur_adapter)
    model = AutoModelForCausalLM.from_pretrained(
                                                model_save_path,
                                                device_map="auto",
                                                quantization_config=Config.quantization_config,
    )

    model = get_peft_model(model,Config.lora_config,adapter_name = cur_adapter)
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint['model']
    set_peft_model_state_dict(model,state_dict)
    model.save_pretrained(adapter_path)
    print(f"model exported to {adapter_path}")
else:
    print("Adapter checkpoint doesn't exist! Please make sure to train the adapter first")