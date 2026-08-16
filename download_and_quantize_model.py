import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn as nn
import bitsandbytes as bnb
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig
from config import Config
project_path = os.path.dirname(os.path.abspath(__file__))
model_id = Config.model_id
cache_dir = r"E:\huggingface"
model_type = "llama-3.1-4b"
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

print("START")
model = AutoModelForCausalLM.from_pretrained(model_id,
                                            torch_dtype=torch.float16,
                                            device_map={"": 0},
                                            quantization_config = quantization_config,
                                            low_cpu_mem_usage=True,
                                            offload_state_dict=True,
                                            cache_dir = cache_dir)

print("MODEL LOADED")

tokenizer = AutoTokenizer.from_pretrained(
                                        model_id,
                                        cache_dir=cache_dir,)
print("TOKENIZER LOADED")
#
#model.save_pretrained(
#    rf"{project_path}\{model_type}"
#)
#
#tokenizer.save_pretrained(rf"{project_path}\{model_type}")
#
#