from dataclasses import dataclass
from peft import LoraConfig
from transformers import BitsAndBytesConfig
import torch
@dataclass
class Config:
    model_id = "llama-3.1-8b"
    dataset = "Abirate/english_quotes"
    lora_config = LoraConfig(
                        r = 32,
                        lora_alpha = 64,
                        lora_dropout = 0.05,
                        bias = "none",
                        task_type = "CAUSAL_LM"
                        )
    quantization_config = BitsAndBytesConfig(
                                            load_in_4bit=True,
                                            bnb_4bit_compute_dtype=torch.bfloat16,  
                                            bnb_4bit_quant_type="nf4",        
                                            bnb_4bit_use_double_quant=True,      
                                        )