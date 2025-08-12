from pathlib import Path
import json
import numpy as np
import faiss

from backend.config.config import VSTORE_DIR, ROOT_DIR
from backend.utils.logger import get_logger

logger = get_logger("rag_service")

INDEX_PATH = VSTORE_DIR / "faiss_index.idx"
META_PATH  = VSTORE_DIR / "metadatas.json"

_index = None
_metas = None
_last_mtime = (0, 0)

def resolve_path_for_meta(meta: dict) -> Path:
    p = Path(meta.get("path", ""))
    if p.exists():
        return p
    rel = meta.get("rel_path")
    if rel:
        rp = ROOT_DIR / rel
        if rp.exists():
            return rp
        try:
            rp2 = (ROOT_DIR / Path(rel))
            if rp2.exists():
                return rp2
        except Exception:
            pass
    return p

def _mtime_pair():
    idx_m = INDEX_PATH.stat().st_mtime if INDEX_PATH.exists() else 0
    meta_m = META_PATH.stat().st_mtime if META_PATH.exists() else 0
    return (idx_m, meta_m)

def _ensure_index():
    if not INDEX_PATH.exists() or not META_PATH.exists():
        raise FileNotFoundError(
            f"FAISS 인덱스가 없습니다. 파이프라인을 먼저 실행하세요: pipelines/ingest_local.sh\n경로: {INDEX_PATH}"
        )

def _reload_if_changed():
    global _index, _metas, _last_mtime
    cur = _mtime_pair()
    if _index is None or cur != _last_mtime:
        _ensure_index()
        logger.info("Loading vector index and metadatas...")
        _index = faiss.read_index(str(INDEX_PATH))
        _metas = json.loads(META_PATH.read_text(encoding='utf-8'))
        _last_mtime = cur

def load_index():
    _reload_if_changed()
    return _index, _metas

def search(vec: np.ndarray, k: int = 5):
    idx, metas = load_index()
    D, I = idx.search(vec.astype('float32').reshape(1, -1), k)
    out = []
    for d, i in zip(D[0], I[0]):
        if 0 <= i < len(metas):
            m = metas[i].copy()
            m['score'] = float(d)
            out.append(m)
    return out
