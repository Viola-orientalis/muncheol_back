# Ensure repo root is importable and import backend config paths
import sys, os
from pathlib import Path as _P
ROOT = _P(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from backend.config.config import (
    ROOT_DIR, DATA_DIR, RAW_DIR, CLEANED_DIR, CHUNKS_DIR, VSTORE_DIR, EMBED_MODEL
)

from pathlib import Path

MAX_CHARS = 1200

def ensure_dirs():
    for d in [CHUNKS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

def split_text(txt: str, max_chars=MAX_CHARS):
    parts, buf, acc = [], [], 0
    for line in txt.splitlines(keepends=True):
        if not line.strip():
            line = "\n"
        buf.append(line)
        acc += len(line)
        if acc >= max_chars:
            parts.append("".join(buf).strip()); buf, acc = [], 0
    if buf:
        parts.append("".join(buf).strip())
    return [p for p in parts if p]

def run():
    ensure_dirs()
    for cat_dir in CLEANED_DIR.iterdir():
        if not cat_dir.is_dir():
            continue
        out_cat = CHUNKS_DIR / cat_dir.name
        out_cat.mkdir(parents=True, exist_ok=True)
        for fp in cat_dir.glob("*.txt"):
            chunks = split_text(fp.read_text(encoding="utf-8", errors="ignore"))
            for i, ch in enumerate(chunks):
                (out_cat / f"{fp.name}_chunk_{i}.txt").write_text(ch, encoding="utf-8")
    print("✅ chunks 생성:", CHUNKS_DIR)

if __name__ == "__main__":
    print("✂️  text_splitter start")
    run()
