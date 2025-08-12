# backend/services/rag_service.py
from pathlib import Path
import json
import numpy as np
import faiss

from backend.config.config import VSTORE_DIR, ROOT_DIR, CHUNKS_DIR
from backend.utils.logger import get_logger

logger = get_logger("rag_service")

INDEX_PATH = VSTORE_DIR / "faiss_index.idx"
META_PATH  = VSTORE_DIR / "metadatas.json"

_index = None
_metas = None
_last_mtime = (0, 0)

def resolve_path_for_meta(meta: dict) -> Path:
    """
    경로 복원 규칙:
      1) meta['path']를 우선 사용하되, 백슬래시(\) → 슬래시(/) 표준화 후 절대/상대 모두 검사
      2) rel_path가 있으면 ROOT_DIR 기준으로 검사
      3) 표준 위치 fallback:
         - data/chunks/<category>/<filename>
         - output/chunks/<category>/<filename> (레거시)
      4) raw path가 'output/chunks'를 가리키면 'data/chunks'로 치환 시도
    """
    raw_path = (meta.get("path") or "").strip()
    rel = (meta.get("rel_path") or "").strip()
    category = meta.get("category") or ""
    filename = meta.get("filename") or ""

    def _norm(p: str) -> Path:
        if not p:
            return Path("")
        p = p.replace("\\", "/").strip()
        return Path(p)

    # 1) meta.path 시도 (절대 → 상대(ROOT_DIR) 순)
    p = _norm(raw_path)
    if p.is_absolute() and p.exists():
        return p
    if str(p):
        cand = (ROOT_DIR / p).resolve()
        if cand.exists():
            return cand

    # 2) rel_path 시도
    rp = _norm(rel)
    if str(rp):
        if rp.is_absolute() and rp.exists():
            return rp
        cand = (ROOT_DIR / rp).resolve()
        if cand.exists():
            return cand

    # 3) 표준 fallback: data/chunks, output/chunks
    if category and filename:
        cand = (CHUNKS_DIR / category / filename).resolve()
        if cand.exists():
            return cand
        cand2 = (ROOT_DIR / "output" / "chunks" / category / filename).resolve()
        if cand2.exists():
            return cand2

    # 4) 레거시 치환: output/chunks → data/chunks
    if "output/chunks" in raw_path or "output\\chunks" in raw_path:
        replaced = raw_path.replace("\\", "/").replace("output/chunks", "data/chunks")
        cand = (ROOT_DIR / _norm(replaced)).resolve()
        if cand.exists():
            return cand

    # 마지막: 그래도 못 찾으면 ROOT_DIR 기준으로 반환(존재 X일 수 있음)
    return (ROOT_DIR / p) if str(p) else ROOT_DIR

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
