from dataclasses import dataclass
from peft import LoraConfig

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
    