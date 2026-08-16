import torch
import os
from peft import get_peft_model,set_peft_model_state_dict
from config import Config
from transformers import AutoModelForCausalLM
from Scripts.Train_Module.training_config import Training_Config
from Scripts.Data_Prepare_Module.metadata import generate_metadata,save_metadata
import json
cur_dir = os.path.dirname(os.path.abspath(__file__))
model_save_path = os.path.join(cur_dir,Config.model_id)
cur_adapter = "author_guess"
ckpt_path = os.path.join(Training_Config.checkpoint_dir,cur_adapter,"best_ckpt.pt")
metadata_examples_path = os.path.join(cur_dir,"data","metadata_examples.json")
adapter_path = os.path.join(cur_dir,"adapters",cur_adapter)

if os.path.exists(ckpt_path):
    if not os.path.exists(os.path.join(cur_dir,"adapters")):
        os.makedirs(os.path.join(cur_dir,"adapters"))
    if not os.path.exists(metadata_examples_path):
        print(
            "metadata_examples.json doesn't exist! "
            "Please make sure the adapter was trained with metadata examples."
        )
    adapter_path = os.path.join(cur_dir,"adapters",cur_adapter)
    model = AutoModelForCausalLM.from_pretrained(
                                                model_save_path,
                                                device_map="auto",
                                                quantization_config=Config.quantization_config,
    )

    model = get_peft_model(model,Config.lora_config,adapter_name = cur_adapter)
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint['model']
    set_peft_model_state_dict(model,state_dict,adapter_name = cur_adapter)
    model.save_pretrained(adapter_path)
    with open(metadata_examples_path,"r",encoding="utf-8") as f:
        metadata_examples = json.load(f)
    metadata = generate_metadata(adapter_name=cur_adapter,examples=metadata_examples)
    save_metadata(adapter_path=adapter_path,adapter_name=cur_adapter,metadata=metadata)
    print(f"model exported to {adapter_path}")
else:
    print("Adapter checkpoint doesn't exist! Please make sure to train the adapter first")