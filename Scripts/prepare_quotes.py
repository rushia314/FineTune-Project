import os
import sys
from datasets import load_dataset
from transformers import AutoTokenizer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from data_hf_prepare import smart_data_prep

def custom_format_quotes(batch):
    formatted_texts = []
    for author, quote, tags in zip(batch['author'], batch['quote'], batch['tags']):
        tags_str = ", ".join(tags) if isinstance(tags, list) else str(tags)
        
        text = f"Author: {author} -> Quote: {quote} -> Tags: {tags_str}"
        formatted_texts.append(text)
    
    return formatted_texts

@smart_data_prep(format_fn=custom_format_quotes, output_folder="data")
def main():
    print(f"Đang tải Tokenizer của {Config.model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(Config.model_id)
    eot_token_id = tokenizer.eos_token_id

    print(f"Đang tải dataset {Config.dataset}...")
    dataset = load_dataset(Config.dataset)

    return dataset, tokenizer, eot_token_id, Config.dataset

if __name__ == '__main__':
    main()