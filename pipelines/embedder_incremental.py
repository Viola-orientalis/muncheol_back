# Ensure repo root is importable and import backend config paths
import sys, os
from pathlib import Path as _P
ROOT = _P(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from backend.config.config import (
    ROOT_DIR, DATA_DIR, RAW_DIR, CLEANED_DIR, CHUNKS_DIR, VSTORE_DIR, EMBED_MODEL
)

import os, json
import numpy as np, faiss
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv

from pipelines.utils_hash import file_sha1

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

VSTORE_INDEX = VSTORE_DIR / "faiss_index.idx"
VSTORE_META  = VSTORE_DIR / "metadatas.json"

def _retry_embed(texts, model, attempts=5):
    out = []
    i = 0
    while i < len(texts):
        batch = texts[i:i+64]
        for t in range(attempts):
            try:
                res = client.embeddings.create(model=model, input=batch)
                out.extend([d.embedding for d in res.data])
                break
            except Exception:
                if t == attempts - 1:
                    raise
        i += 64
    return np.array(out, dtype="float32")

def _atomic_write(path: Path, content: str):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)

def _atomic_write_index(index, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    faiss.write_index(index, str(tmp))
    os.replace(tmp, path)

def ensure_dirs():
    for d in [VSTORE_DIR]:
        d.mkdir(parents=True, exist_ok=True)

def load_existing():
    if VSTORE_INDEX.exists() and VSTORE_META.exists():
        return faiss.read_index(str(VSTORE_INDEX)), json.loads(VSTORE_META.read_text(encoding="utf-8"))
    return None, []

def collect_chunks():
    items = []
    for cat_dir in CHUNKS_DIR.iterdir():
        if not cat_dir.is_dir(): continue
        for fp in cat_dir.glob("*.txt"):
            items.append((str(fp), cat_dir.name, fp.name, file_sha1(fp)))
    return items

def diff_new_changed(existing_metas, current_items):
    old_pairs = {(m.get("path",""), m.get("sha1","")) for m in existing_metas}
    return [(p,c,f,s) for (p,c,f,s) in current_items if (p,s) not in old_pairs]

def embed_batch(texts, batch=64):
    out = []
    for i in tqdm(range(0, len(texts), batch)):
        chunk = texts[i:i+batch]
        vecs = _retry_embed(chunk, EMBED_MODEL)
        out.extend(list(vecs))
    return np.array(out, dtype="float32")

def run():
    ensure_dirs()
    index, metas = load_existing()
    current = collect_chunks()
    targets = diff_new_changed(metas, current)

    if index is None:
        print("🔰 최초 인덱스 생성...")
        texts, new_metas = [], []
        for p, cat, fn, sha1 in current:
            texts.append(Path(p).read_text(encoding="utf-8", errors="ignore"))
            new_metas.append({"category":cat, "filename":fn, "path":str(Path(p).resolve()), "rel_path":str(Path(p).resolve().relative_to(ROOT_DIR).as_posix()), "sha1":sha1})
        if not texts:
            print("[warn] 청크가 없습니다. text_splitter를 먼저 실행하세요.")
            return
        vecs = embed_batch(texts)
        dim = vecs.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(vecs)
        metas = new_metas
    else:
        if not targets:
            print("변경/신규 청크 없음. 인덱스 유지.")
        else:
            print(f"➕ 신규/변경 청크 {len(targets)}개 추가 중...")
            texts, add_metas = [], []
            for p, cat, fn, sha1 in targets:
                texts.append(Path(p).read_text(encoding="utf-8", errors="ignore"))
                add_metas.append({"category":cat, "filename":fn, "path":str(Path(p).resolve()), "rel_path":str(Path(p).resolve().relative_to(ROOT_DIR).as_posix()), "sha1":sha1})
            vecs = embed_batch(texts)
            index.add(vecs)
            metas.extend(add_metas)

    _atomic_write_index(index, VSTORE_INDEX)
    _atomic_write(VSTORE_META, json.dumps(metas, ensure_ascii=False, indent=2))
    print(f"✅ 저장 완료 → {VSTORE_INDEX}")

if __name__ == "__main__":
    run()
