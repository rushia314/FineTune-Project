import os
import sys
from datasets import load_dataset
from transformers import AutoTokenizer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from data_hf_prepare import smart_data_prep
def clean_text(text):
    if not isinstance(text, str):
        return str(text)

    text = text.replace("â€™", "'")
    text = text.replace("â€œ", '"')
    text = text.replace("â€", '"') 
    text = text.replace("â€“", "-") 
    return text

def chat_format(batch, nums, columns):
    all_messages = []
    col_user = columns[nums[0] - 1]
    col_assistant = columns[nums[1] - 1]
    batch_size = len(batch[col_user])
    
    for i in range(batch_size):
        user_text = clean_text(batch[col_user][i])
        assistant_text = clean_text(batch[col_assistant][i])
        
        messages = [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text}
        ]
        all_messages.append(messages)
        
    return all_messages


@smart_data_prep(format_fn=chat_format, output_folder="data")
def main(inputs = None):
    print(f"Đang tải Tokenizer của {Config.model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(Config.model_id)
    eot_token_id = tokenizer.eos_token_id
    print(inputs)
    print(f"Đang tải dataset {Config.dataset}...")
    dataset = load_dataset(Config.dataset)

    return dataset, tokenizer, eot_token_id, Config.dataset

if __name__ == '__main__':
    main()