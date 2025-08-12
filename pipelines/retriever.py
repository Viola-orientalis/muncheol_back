# Ensure repo root is importable and import backend config paths
import sys, os
from pathlib import Path as _P
ROOT = _P(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from backend.config.config import (
    ROOT_DIR, DATA_DIR, RAW_DIR, CLEANED_DIR, CHUNKS_DIR, VSTORE_DIR, EMBED_MODEL
)

import os, json, numpy as np, faiss
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

VSTORE_INDEX = VSTORE_DIR / "faiss_index.idx"
VSTORE_META  = VSTORE_DIR / "metadatas.json"

def qvec(q:str):
    return client.embeddings.create(model=EMBED_MODEL, input=q).data[0].embedding

if __name__ == "__main__":
    if not VSTORE_INDEX.exists():
        print("[ERR] 인덱스가 없습니다. pipelines/ingest_local.sh 먼저 실행하세요.")
        raise SystemExit(1)
    q = input("질문> ").strip() or "스모크 테스트"
    vec = np.array(qvec(q), dtype="float32").reshape(1,-1)
    index = faiss.read_index(str(VSTORE_INDEX))
    metas = json.load(open(VSTORE_META, encoding="utf-8"))
    D, I = index.search(vec, 5)
    for rank, (d, idx) in enumerate(zip(D[0], I[0]), 1):
        m = metas[idx]
        print(f"{rank}. {m['filename']} ({m['category']}) dist={float(d):.4f} path={m['path']}")
