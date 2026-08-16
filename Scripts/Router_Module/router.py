import os 
import json
def build_router_text(metadata):
    capabilities = "\n".join(
        f"- {cap}"
        for cap in metadata["capabilities"]
    )

    examples = "\n".join(
        f"- {example}"
        for example in metadata["router_examples"]
    )

    return (
        f"Description:\n"
        f"{metadata['description']}\n\n"
        f"Capabilities:\n"
        f"{capabilities}\n\n"
        f"Example requests:\n"
        f"{examples}"
    )
def build_registry(adapters_path):
    registry = {}

    if not os.path.isdir(adapters_path):
        return registry

    for entry in os.scandir(adapters_path):
        if not entry.is_dir():
            continue

        metadata_path = os.path.join(
            entry.path,
            "metadata.json"
        )

        if not os.path.isfile(metadata_path):
            print(
                f"Bỏ qua adapter '{entry.name}': "
                f"không có metadata.json"
            )
            continue

        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            print(
                f"Không đọc được metadata của '{entry.name}': {e}"
            )
            continue

        registry[entry.name] = {
            "path": entry.path,
            "metadata": metadata,
        }

    return registry
