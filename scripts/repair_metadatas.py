import json
import os
from pathlib import Path

VSTORE_META = Path("vectorstore/dev/current/metadatas.json")
CHUNKS_DIR = Path("data/chunks")

def find_matching_file(filename):
    # 카테고리 하위 폴더 포함 모든 청크 파일 검색
    for path in CHUNKS_DIR.rglob("*"):
        if path.is_file() and path.name == filename:
            return path
    return None

def main():
    if not VSTORE_META.exists():
        print(f"[ERR] 메타데이터 파일 없음: {VSTORE_META}")
        return

    with open(VSTORE_META, "r", encoding="utf-8") as f:
        metas = json.load(f)

    fixed = 0
    for m in metas:
        orig_path = m.get("path") or m.get("rel_path")
        filename = Path(orig_path).name if orig_path else None
        if not filename:
            continue

        chunk_path = find_matching_file(filename)
        if chunk_path:
            m["path"] = str(chunk_path.resolve())
            m["rel_path"] = str(chunk_path.relative_to(Path.cwd()))
            fixed += 1
        else:
            print(f"[WARN] 매칭 실패: {filename}")

    with open(VSTORE_META, "w", encoding="utf-8") as f:
        json.dump(metas, f, ensure_ascii=False, indent=2)

    print(f"[OK] 총 {fixed}개 경로 수정 완료")

if __name__ == "__main__":
    main()
