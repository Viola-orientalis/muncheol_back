import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from backend.config.config import VSTORE_DIR

idx = VSTORE_DIR / "faiss_index.idx"
meta = VSTORE_DIR / "metadatas.json"

print("Index exists:", idx.exists(), idx)
print("Meta  exists:", meta.exists(), meta)
