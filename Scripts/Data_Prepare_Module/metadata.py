from google import genai
from pydantic import BaseModel, Field
import os
import json

client = genai.Client()
class AdapterMetadata(BaseModel):
    description: str = Field(
        description="What this adapter is specialized for."
    )

    capabilities: list[str] = Field(
        description="Main capabilities of the adapter."
    )

    router_examples: list[str] = Field(
        description="Example user requests that should route to this adapter."
    )


def generate_metadata(
    adapter_name: str,
    examples: list[dict[str, str]]
) -> AdapterMetadata:

    examples_text = "\n\n".join(
        f"Example {i + 1}:\n"
        f"Input: {example['input']}\n"
        f"Output: {example['output']}"
        for i, example in enumerate(examples)
    )

    prompt = f"""
You are generating metadata for a LoRA adapter.

Adapter name:
{adapter_name}

This adapter was trained using examples of the following form:

{examples_text}

Infer what this adapter is specialized for.

Rules:
- Base the description only on the provided input/output examples.
- Do not invent capabilities that are not supported by the examples.
- Identify the underlying task, not the specific facts in the examples.
- router_examples should be realistic user requests that would benefit from this adapter.
- Do not rename the adapter.
- Do not include specific training examples as capabilities unless they represent
  a general capability of the adapter.
- router_examples must be generic and task-oriented.
- Do not copy specific quotes, names, or passages from the training examples.
- Generate diverse paraphrases of the kind of user request this adapter should handle.
Return only the requested structured output.
"""

    response = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": AdapterMetadata,
        },
    )

    if response.parsed is None:
        raise ValueError(
            "Gemini did not return valid structured metadata."
        )

    return response.parsed


def save_metadata(
    adapter_path: str,
    adapter_name: str,
    metadata: AdapterMetadata
):
    os.makedirs(adapter_path, exist_ok=True)

    metadata_path = os.path.join(
        adapter_path,
        "metadata.json"
    )

    data = {
        "name": adapter_name,
        **metadata.model_dump()
    }

    with open(
        metadata_path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=4
        )