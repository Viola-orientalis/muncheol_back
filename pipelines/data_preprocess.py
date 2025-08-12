# Ensure repo root is importable and import backend config paths
import sys, os
from pathlib import Path as _P
ROOT = _P(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from backend.config.config import (
    ROOT_DIR, DATA_DIR, RAW_DIR, CLEANED_DIR, CHUNKS_DIR, VSTORE_DIR, EMBED_MODEL
)

import re, json
from pathlib import Path

def ensure_dirs():
    for d in [RAW_DIR, CLEANED_DIR]:
        d.mkdir(parents=True, exist_ok=True)

def clean_text(s: str) -> str:
    s = s.replace("\r", "")
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def extract_text_from_any(path: Path) -> str:
    if path.suffix.lower() in [".txt", ".md"]:
        return path.read_text(encoding="utf-8", errors="ignore")
    elif path.suffix.lower() in [".json", ".csv"]:
        try:
            return json.dumps(json.loads(path.read_text(encoding="utf-8", errors="ignore")),
                              ensure_ascii=False, indent=2)
        except Exception:
            return path.read_text(encoding="utf-8", errors="ignore")
    else:
        return path.read_text(encoding="utf-8", errors="ignore")

def run():
    ensure_dirs()
    cats = [p for p in RAW_DIR.iterdir() if p.is_dir()]
    if not cats:
        print(f"[warn] {RAW_DIR} 아래에 카테고리 폴더를 만들어 원본을 넣어주세요.")
        return
    for cat_dir in cats:
        out_dir = CLEANED_DIR / cat_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        for fp in cat_dir.rglob("*"):
            if not fp.is_file():
                continue
            try:
                txt = extract_text_from_any(fp)
                (out_dir / f"{fp.stem}.txt").write_text(clean_text(txt), encoding="utf-8")
            except Exception as e:
                print("[skip]", fp, e)
    print("✅ cleaned 생성:", CLEANED_DIR)

if __name__ == "__main__":
    print("📂 data_preprocess start")
    run()
