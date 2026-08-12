import os
import numpy as np
from datasets import DatasetDict
from tqdm import tqdm

def smart_data_prep(format_fn, output_folder="data"):
    def decorator(func):
        def wrapper():
            dataset, tokenizer, eot_token_id, dataset_name = func()
            
            first_split = list(dataset.keys())[0]
            columns = dataset[first_split].column_names
            print(f"🔍 Danh sách cột thô gốc trong dataset: {columns}")

            if 'validation' not in dataset and 'test' not in dataset:
                print("Dataset chưa có sẵn val, đang tách 10% từ train...")
                split_dataset = dataset['train'].train_test_split(test_size=0.1, seed=42)
                dataset = DatasetDict({
                    'train': split_dataset['train'],
                    'validation': split_dataset['test']
                })

            def process_batch(batch):
                texts = format_fn(batch)
                texts = [str(t) if t is not None else "" for t in texts]
                
                ids = [tokenizer.encode(t, add_special_tokens=False) + [eot_token_id] for t in texts]
                return {'ids': ids, 'len': [len(i) for i in ids]}

            optimal_proc = min(8, max(1, os.cpu_count() - 2))
            print(f"Đang Tokenize dữ liệu với {optimal_proc} luồng...")
            
            tokenized = dataset.map(
                process_batch,
                remove_columns=columns,
                desc="Tokenizing",
                batched=True,
                num_proc=optimal_proc,
                load_from_cache_file=False,
            )

            dataset_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            data_dir = os.path.join(dataset_dir, output_folder)
            os.makedirs(data_dir, exist_ok=True)

            for split, dset in tokenized.items():
                file_split = 'val' if split == 'validation' else split
                filename = os.path.join(data_dir, f'{file_split}.bin')
                
                arr_len = np.sum(dset['len'], dtype=np.uint64)
                print(f"\nGhi {filename} (Tổng cộng {arr_len:,} tokens)...")

                arr = np.memmap(filename, dtype=np.int32, mode='w+', shape=(arr_len,))
                
                batch_size = 1024 * 16
                idx = 0
                for start_idx in tqdm(range(0, len(dset), batch_size), desc=f'Ghi {file_split}'):
                    end_idx = min(start_idx + batch_size, len(dset))
                    batch = dset[start_idx : end_idx] 
                    arr_batch = np.concatenate(batch['ids'])
                    arr[idx : idx + len(arr_batch)] = arr_batch
                    idx += len(arr_batch)
                
                arr.flush()
                print(f"Đã lưu xong {filename}!")

        return wrapper
    return decorator