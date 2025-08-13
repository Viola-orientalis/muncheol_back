import os, json, re, zipfile
from pathlib import Path
from config import RAW_DIR, CLEANED_DIR, ensure_dirs

# zip 파일명에 포함된 키워드로 카테고리 자동 판별
CATEGORY_KEYWORDS = {
    "판결문": ["판결문"],
    "결정례": ["결정례"],
    "해석례": ["해석례"],
    "법령":   ["법령"],
}

# 이미 폴더/파일이 있으면 덮어쓸지 여부 (환경변수로 제어)
FORCE_UNZIP = os.getenv("FORCE_UNZIP", "0") == "1"

def _detect_category(name: str) -> str:
    n = name.lower()
    for cat, kws in CATEGORY_KEYWORDS.items():
        for kw in kws:
            if kw.lower() in n:
                return cat
    return "기타"

def _normalize_zip_filenames():
    """
    확장자 앞 공백 같은 이상한 zip 이름 정리:
      'VS_결정례 .zip' -> 'VS_결정례.zip'
    """
    for z in RAW_DIR.glob("*.zip"):
        fixed = RAW_DIR / z.name.replace(" .zip", ".zip")
        if fixed != z:
            try:
                z.rename(fixed)
            except Exception:
                pass

def _auto_unzip_grouped():
    """
    data/raw/*.zip → zip 이름으로 카테고리 판별 후
    data/raw/<카테고리>/ 로 자동 해제.
    이미 대상 폴더에 내용물이 있으면 기본은 스킵(FORCE_UNZIP=1이면 덮어씀).
    """
    _normalize_zip_filenames()

    count = 0
    for z in sorted(RAW_DIR.glob("*.zip")):
        base = z.stem
        cat = _detect_category(base)
        out_dir = RAW_DIR / cat
        out_dir.mkdir(parents=True, exist_ok=True)

        # 이미 풀려 있으면 스킵(강제 모드 제외)
        if not FORCE_UNZIP and any(out_dir.iterdir()):
            # 카테고리 폴더 안에 파일이 하나라도 있으면 스킵
            continue

        try:
            with zipfile.ZipFile(z, "r") as zip_ref:
                zip_ref.extractall(out_dir)
            print(f"✅ unzip: {z.name} -> {out_dir}")
            count += 1
        except Exception as e:
            print(f"[warn] unzip 실패: {z.name} ({e})")
    if count == 0:
        print("[info] 자동 해제 대상 zip 없음 또는 이미 해제됨.")

def _clean_text(s: str) -> str:
    s = s.replace("\r", "")
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def _extract_text_from_any(path: Path) -> str:
    # 텍스트/마크다운/JSON/CSV 단순화 추출
    if path.suffix.lower() in [".txt", ".md"]:
        return path.read_text(encoding="utf-8", errors="ignore")
    elif path.suffix.lower() in [".json", ".csv"]:
        try:
            return json.dumps(
                json.loads(path.read_text(encoding="utf-8", errors="ignore")),
                ensure_ascii=False, indent=2
            )
        except Exception:
            return path.read_text(encoding="utf-8", errors="ignore")
    else:
        # 기타 확장자는 일단 텍스트로 시도
        return path.read_text(encoding="utf-8", errors="ignore")

def run():
    ensure_dirs()

    # 0) ZIP 자동 해제 (카테고리 매핑)
    _auto_unzip_grouped()

    # 1) 카테고리 디렉터리 스캔
    cats = [p for p in RAW_DIR.iterdir() if p.is_dir()]
    if not cats:
        print(f"[warn] {RAW_DIR} 아래에 카테고리 폴더를 만들어 원본을 넣어주세요.")
        return

    # 2) 파일별 정제 텍스트 생성 → data/cleaned/<카테고리>/*.txt
    for cat_dir in cats:
        out_dir = CLEANED_DIR / cat_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        for fp in cat_dir.rglob("*"):
            if not fp.is_file():
                continue
            try:
                txt = _extract_text_from_any(fp)
                (out_dir / f"{fp.stem}.txt").write_text(_clean_text(txt), encoding="utf-8")
            except Exception as e:
                print("[skip]", fp, e)

    print("✅ cleaned 생성:", CLEANED_DIR)

if __name__ == "__main__":
    print("📂 data_preprocess start")
    run()

