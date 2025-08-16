from pathlib import Path
import os

# ── 프로젝트 루트 (P_backend/)
ROOT_DIR = Path(__file__).resolve().parents[2]

# ── 데이터 경로
DATA_DIR    = ROOT_DIR / "data"
RAW_DIR     = DATA_DIR / "raw"
CLEANED_DIR = DATA_DIR / "cleaned"
CHUNKS_DIR  = DATA_DIR / "chunks"

# ── 벡터스토어 경로
VSTORE_DIR   = ROOT_DIR / "vectorstore" / "dev" / "current"
VSTORE_INDEX = VSTORE_DIR / "faiss_index.idx"
VSTORE_META  = VSTORE_DIR / "metadatas.json"

# ── 모델 이름 (환경변수로 오버라이드 가능)
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL  = os.getenv("CHAT_MODEL", "gpt-4o")

# ── Flask / CORS / 로깅
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*")  # prod에선 콤마로 구분
LOG_LEVEL       = os.getenv("LOG_LEVEL", "INFO")

def ensure_dirs():
    """파이프라인에서 필요 폴더들을 미리 생성"""
    for d in [RAW_DIR, CLEANED_DIR, CHUNKS_DIR, VSTORE_DIR]:
        d.mkdir(parents=True, exist_ok=True)
