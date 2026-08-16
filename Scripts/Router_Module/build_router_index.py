import os
import sys
import json
project_root = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
import numpy as np

from router import build_registry, build_router_text

from Scripts.Router_Module.embedder import model

adapters_path = os.path.join(project_root,"adapters")

registry = build_registry(
    adapters_path
)

if not registry:
    raise RuntimeError(
        "Không tìm thấy adapter hợp lệ."
    )


adapter_names = []
router_texts = []

for adapter_name, info in registry.items():
    metadata = info["metadata"]

    text = build_router_text(
        metadata
    )

    adapter_names.append(
        adapter_name
    )

    router_texts.append(
        text
    )


embeddings = model.encode(
    router_texts,
    normalize_embeddings=True
)

print("Adapters:")
for name in adapter_names:
    print("-", name)

print("Embedding shape:")
print(embeddings.shape)
index_dir = os.path.join(project_root,"Scripts","Router_Module","router_index")

os.makedirs(
    index_dir,
    exist_ok=True
)

np.save(
    os.path.join(
        index_dir,
        "embeddings.npy"
    ),
    embeddings
)

with open(
    os.path.join(
        index_dir,
        "adapter_names.json"
    ),
    "w",
    encoding="utf-8"
) as f:
    import json

    json.dump(
        adapter_names,
        f,
        ensure_ascii=False,
        indent=2
    )