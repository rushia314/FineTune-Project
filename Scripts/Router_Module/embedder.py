from sentence_transformers import SentenceTransformer
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

model = SentenceTransformer(MODEL_NAME)


def embed_text(text: str):
    return model.encode(
        text,
        normalize_embeddings=True
    )