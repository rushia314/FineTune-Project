import os
import sys
project_path = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)
if project_path not in sys.path:
    sys.path.insert(0, project_path)
import numpy as np
from datasets import DatasetDict
from tqdm import tqdm
import re
def smart_data_prep(format_fn, output_folder="data"):
    def decorator(func):
        def wrapper(*args,**kwargs):
            dataset, tokenizer, eot_token_id, dataset_name = func(*args, **kwargs)
            
            first_split = list(dataset.keys())[0]
            columns = dataset[first_split].column_names
            print("All columns:")
            for k,column_name in enumerate(columns):
                print(f"{k+1}: {column_name}")
            nums = re.findall(r'\d+',input("choose the data format for example 1-2 or 2-1 : "))
            nums = [int(num) for num in  nums]
            if 'validation' not in dataset and 'test' not in dataset:
                print("Dataset chưa có sẵn val, đang tách 10% từ train...")
                split_dataset = dataset['train'].train_test_split(test_size=0.1, seed=42)
                dataset = DatasetDict({
                    'train': split_dataset['train'],
                    'validation': split_dataset['test']
                })
            # ================= METADATA EXAMPLES =================
            metadata_count = min(30, len(dataset["train"]))
            metadata_dataset = dataset["train"].shuffle(seed=42).select(
                range(metadata_count)
            )
            metadata_batch = metadata_dataset[:metadata_count]
            metadata_messages = format_fn(
                metadata_batch,
                nums,
                columns
            )
            metadata_examples = []

            for messages in metadata_messages:
                metadata_examples.append({
                    "input": messages[0]["content"],
                    "output": messages[1]["content"]
                })
            data_dir = os.path.join(project_path, output_folder)
            os.makedirs(data_dir, exist_ok=True)
            metadata_examples_path = os.path.join(
                data_dir,
                "metadata_examples.json"
            )
            with open(
                metadata_examples_path,
                "w",
                encoding="utf-8"
            ) as f:
                import json

                json.dump(
                    metadata_examples,
                    f,
                    ensure_ascii=False,
                    indent=2
                )
            print(
                f"Đã lưu {len(metadata_examples)} metadata examples "
                f"vào {metadata_examples_path}"
            )
            # ======================================================
            def process_batch(batch):
                messages_list = format_fn(batch, nums, columns)
                texts = [tokenizer.apply_chat_template(msgs, tokenize=False) for msgs in messages_list]
                ids_list = [tokenizer.encode(t, add_special_tokens=False) for t in texts]

                assistant_header = "<|start_header_id|>assistant<|end_header_id|>\n\n"
                header_ids = tokenizer.encode(assistant_header, add_special_tokens=False)
                eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")

                header_len = len(header_ids)
                first_header_id = header_ids[0]

                labels_list = []

                for ids in ids_list:
                    labels = [-100] * len(ids)
                    i = 0
                    total_len = len(ids)
                    
                    while i < total_len:
                        if ids[i] == first_header_id and ids[i : i + header_len] == header_ids:
                            i += header_len
                            start_idx = i
                            
                            try:
                                end_idx = ids.index(eot_id, start_idx) + 1 
                            except ValueError:
                                end_idx = total_len
                            
                            labels[start_idx:end_idx] = ids[start_idx:end_idx]
                            
                            i = end_idx 
                        else:
                            i += 1
                            
                    labels_list.append(labels)
                return {'ids': ids_list, 'labels': labels_list, 'len': [len(i) for i in ids_list]}

            
            optimal_proc = min(8, max(1, os.cpu_count() - 2))

            # ================= DEBUG ================================
            
            # 1. Bốc thử 1 dòng đầu tiên của tập Train
            sample_batch = dataset['train'][:1]
            
            sample_msgs = format_fn(sample_batch, nums, columns)[0]
            print("1. DẠNG DICTIONARY GỐC:")
            print(sample_msgs)
            
            sample_text = tokenizer.apply_chat_template(sample_msgs, tokenize=False)
            print("2. DẠNG STRING (SAU KHI ỐP TEMPLATE):")
            print(sample_text)
            
            print("="*50 + "\n")
            
            input("Bấm Enter nếu template trông đã chuẩn để bắt đầu Tokenize...")
            # =========================================================
            print(f"Đang Tokenize dữ liệu với {optimal_proc} luồng...")
            
            tokenized = dataset.map(
                process_batch,
                remove_columns=columns,
                desc="Tokenizing",
                batched=True,
                num_proc=optimal_proc,
                load_from_cache_file=False,
            )

            data_dir = os.path.join(project_path, output_folder)
            os.makedirs(data_dir, exist_ok=True)
            for split, dset in tokenized.items():
                file_split = 'val' if split == 'validation' else split
                os.makedirs(os.path.join(data_dir,file_split),exist_ok = True)
                ids_dir = os.path.join(data_dir,file_split,'ids.bin')
                labels_dir = os.path.join(data_dir,file_split,'labels.bin')
                arr_len = np.sum(dset['len'], dtype=np.uint64)
                print(f"\nGhi {file_split} (Tổng cộng {arr_len:,} tokens)...")

                arr_X = np.memmap(ids_dir, dtype=np.int32, mode='w+', shape=(arr_len,))
                arr_Y = np.memmap(labels_dir, dtype=np.int32, mode='w+', shape=(arr_len,))
                
                batch_size = 1024 * 16
                idx = 0
                for start_idx in tqdm(range(0, len(dset), batch_size), desc=f'Ghi {file_split}'):
                    end_idx = min(start_idx + batch_size, len(dset))
                    batch = dset[start_idx : end_idx] 
                    
                    arr_batch_X = np.concatenate(batch['ids'])
                    arr_batch_Y = np.concatenate(batch['labels'])
                    
                    arr_X[idx : idx + len(arr_batch_X)] = arr_batch_X
                    arr_Y[idx : idx + len(arr_batch_Y)] = arr_batch_Y
                    idx += len(arr_batch_X)
                
                arr_X.flush()
                arr_Y.flush()
                print(f"Đã lưu xong {ids_dir} và {labels_dir}!")

        return wrapper
    return decorator