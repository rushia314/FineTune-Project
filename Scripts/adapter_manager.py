import os 
from peft import PeftModel
import json
import sys

def adapters_list(adapters_path):
    if os.path.exists(adapters_path):
        adapters_list = [name for name in os.listdir(adapters_path) if os.path.isdir(os.path.join(adapters_path, name))]
        return adapters_list
    else:
        print("adapters_path doesn't exist")
        return []
def adapter_exists(adapters_path,adapter):
    if not os.path.exists(adapters_path):
        raise ValueError(f"Thư mục '{adapters_path}' không tồn tại!")
    return os.path.isdir(os.path.join(adapters_path, adapter))

def load_adapters(model,adapters_path,*args,all_adapter = False):
    if not os.path.exists(adapters_path):
        raise ValueError(f"Thư mục '{adapters_path}' không tồn tại!")

    if not all_adapter and not args:
        print("Không có tên Adapter nào được truyền vào,sẽ trả về model gốc!")
        return model

    adapters_list = [name for name in os.listdir(adapters_path) if os.path.isdir(os.path.join(adapters_path, name))]
    if all_adapter:
        if len(adapters_list) > 0:
            first_adapter = adapters_list[0]
            first_adapter_path = os.path.join(adapters_path, first_adapter)
            model = PeftModel.from_pretrained(model, first_adapter_path, adapter_name=first_adapter)
        if len(adapters_list) > 1:
            for adapter in adapters_list[1:]:
                adapter_path = os.path.join(adapters_path, adapter)
                model.load_adapter(adapter_path, adapter_name=adapter)
        print("all adapters loaded!")
        return model
    else:

        valid_adapters = [name for name in args if name in adapters_list]
        invalid_adapters = [name for name in args if name not in adapters_list]
        if invalid_adapters:
            print(f"Các adapter sau không tồn tại và sẽ bị bỏ qua: {invalid_adapters}")

        if not valid_adapters:
            print("Không có adapter hợp lệ nào để load. Trả về Base Model.")
            return model
        first_adapter = valid_adapters[0]

        first_adapter_path = os.path.join(adapters_path, first_adapter)

        model = PeftModel.from_pretrained(model, first_adapter_path, adapter_name=first_adapter)

        if len(valid_adapters) > 1:
            for adapter in valid_adapters[1:]:
                adapter_path = os.path.join(adapters_path, adapter)
                model.load_adapter(adapter_path, adapter_name=adapter)
        print("all choosen adapters succesfully loaded ")
        return model

def get_metadata(registry, adapter_name):
    adapter = registry.get(adapter_name)

    if adapter is None:
        return None

    return adapter["metadata"]
def get_adapter(registry, adapter_name):
    return registry.get(adapter_name)
