import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import json
import torch
import torch.nn as nn
import bitsandbytes as bnb
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig
from config import Config
from peft import LoraConfig, get_peft_model,set_peft_model_state_dict

project_path = os.path.dirname(os.path.abspath(__file__))
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)
model_save_path = rf"{project_path}\llama-3.1-8b"
profiles_path = rf"{project_path}\user_profile"
if not os.path.exists(profiles_path):
    print("đang tạo thư mục người dùng...")
    os.makedirs(profiles_path)
print("Đang nạp mô hình vào VRAM...")
model = AutoModelForCausalLM.from_pretrained(
    model_save_path,
    device_map="auto",
    quantization_config = quantization_config,
    dtype = torch.float16,
)


tokenizer = AutoTokenizer.from_pretrained(model_save_path)
model.config.use_cache = True
print("Sẵn sàng!\n")
model = get_peft_model(model,Config.lora_config)
if os.path.exists("Checkpoint/ckpt.pt"):
    checkpoint = torch.load("Checkpoint/ckpt.pt", map_location='cpu', weights_only=False)

    set_peft_model_state_dict(model,checkpoint['model'])
    del checkpoint
profile_name = input("nhập tên profile muốn dùng: \n")
profile_path = os.path.join(profiles_path,profile_name)
HISTORY_FILE = rf"{profile_path}\chat_history.json"

print("Đang kiểm tra trí nhớ...")
if os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        chat_history = json.load(f)
    print("Đã khôi phục trí nhớ từ lần chat trước!\n")
else:
    print("\nprofile không tồn tại! Đang tạo profile mới...")
    os.makedirs(profile_path,exist_ok=True)
    chat_history = [
        {"role": "system", "content": "You are a helpful and concise AI assistant."}
    ]
    print("Khởi tạo trí nhớ mới!\n")

while True:
    user_input = input("\nBạn: ")
    
    if user_input.lower() in ['quit', 'exit', 'thoát']:
        print("Đang thoát và giải phóng VRAM...")
        break
        
    chat_history.append({"role": "user", "content": user_input})

    inputs = tokenizer.apply_chat_template(
        chat_history,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True
    ).to(model.device)

    print("AI:", end=" ", flush=True)

    input_length = inputs['input_ids'].shape[1]
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        pad_token_id=tokenizer.eos_token_id
    )
    
    response_tokens = outputs[0][input_length:]
    ai_response = tokenizer.decode(response_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    
    print(ai_response)

    chat_history.append({"role": "assistant", "content": ai_response})
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(chat_history, f, ensure_ascii=False, indent=4)

        