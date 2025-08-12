# Ensure repo root is importable and import backend config paths
import sys, os
from pathlib import Path as _P
ROOT = _P(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from backend.config.config import (
    ROOT_DIR, DATA_DIR, RAW_DIR, CLEANED_DIR, CHUNKS_DIR, VSTORE_DIR, EMBED_MODEL
)

import os, json, time, re
import numpy as np, faiss
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

LLM_MODEL   = "gpt-4o"
TOPK_EMBED  = 40
MAX_DOCS    = 12
MAX_CHARS   = 1600
ALPHA       = 0.65
SAFE_MODE   = True

def embed(x:str):
    return client.embeddings.create(model=EMBED_MODEL, input=x).data[0].embedding

def load_store():
    idx = faiss.read_index(str(VSTORE_DIR / "faiss_index.idx"))
    metas = json.loads((VSTORE_DIR / "metadatas.json").read_text(encoding="utf-8"))
    return idx, metas

def faiss_search(qv, index, k=TOPK_EMBED):
    D, I = index.search(np.array(qv, dtype="float32").reshape(1,-1), k)
    return D[0], I[0]

STOPWORDS = set(["은","는","이","가","을","를","에","에서","으로","과","와","및","the","a","an","of","in","on","for","to","and","or","is","are"])

def extract_keywords(q:str, min_len=2, max_kw=8):
    toks = re.split(r"\s+", re.sub(r"[^\w가-힣 ]+"," ", q))
    seen, out = set(), []
    for t in toks:
        tl = t.lower().strip()
        if not tl or len(tl)<min_len or tl in STOPWORDS: continue
        if tl not in seen:
            out.append(t); seen.add(tl)
        if len(out)>=max_kw: break
    return out

def kw_score(text:str, kws):
    if not kws: return 0.0
    s = text.lower()
    return sum(len(re.findall(re.escape(k.lower()), s)) for k in kws)

def minmax(a):
    if not a: return []
    mn, mx = min(a), max(a)
    if mx - mn < 1e-9: return [0.0]*len(a)
    return [(x-mn)/(mx-mn) for x in a]

NUM_PAT = re.compile(r"([0-9][0-9,\.]*\s*(원|만원|천원|백만원)|과태료|벌금|부과|별표|제\d+조|제\d+항|제\d+호)")
def numeric_snippet(txt:str, max_chars=MAX_CHARS):
    lines = re.split(r"(?<=[\.\!\?]\s|\n)", txt)
    hits = [ln for ln in lines if NUM_PAT.search(ln)]
    return ("\n".join(hits)[:max_chars]) if hits else txt[:max_chars]

def highlight(txt:str, kws):
    if not kws: return txt
    def repl(m): return f"**{m.group(0)}**"
    pat = re.compile("|".join([re.escape(k) for k in kws]), re.IGNORECASE)
    return pat.sub(repl, txt)

def pick_hybrid_best(I, D, metas, question, max_docs=MAX_DOCS):
    kws = extract_keywords(question)
    # 파일당 best chunk
    best = {}
    for idx, dist in zip(I, D):
        if idx >= len(metas): continue
        fn = metas[idx]["filename"]
        if fn not in best or dist < best[fn][1]:
            best[fn] = (idx, float(dist))
    items, emb_raw, kw_raw = [], [], []
    for fn, (idx, dist) in best.items():
        m = metas[idx]
        try: content = Path(m["path"]).read_text(encoding="utf-8", errors="ignore")
        except: content = ""
        emb_raw.append(-dist); kw_raw.append(kw_score(content, kws))
        items.append({"meta":m,"content":content,"dist":dist})
    emb, kw = minmax(emb_raw), minmax(kw_raw)
    scored = []
    for i, it in enumerate(items):
        s = ALPHA*emb[i] + (1-ALPHA)*kw[i]
        has = any(k.lower() in it["content"].lower() for k in kws) if kws else True
        s += 0.05 if has else -0.05
        scored.append((s, it))
    scored.sort(key=lambda x:x[0], reverse=True)
    out = []
    for rank, (score, it) in enumerate(scored[:max_docs], start=1):
        out.append((rank, it["meta"], it["content"], it["dist"], score))
    return out, kws

def has_kw(selected, kws):
    if not kws: return True
    pat = re.compile("|".join([re.escape(k) for k in kws]), re.IGNORECASE)
    return any(pat.search(c) for _,_,c,_,_ in selected)

def build_context(selected, kws):
    ctxs = []
    for rank, meta, content, dist, score in selected:
        snip = highlight(numeric_snippet(content), kws)
        ctxs.append(f"[{rank}] {meta['filename']} ({meta['category']}) (dist={dist:.4f},hyb={score:.3f})\n{snip}")
    return "\n\n---\n\n".join(ctxs)

def build_messages(history, question, ctx, kws):
    recent = history[-6:]
    sys = {"role":"system","content":
           "당신은 한국 법률 도메인 RAG 비서입니다. "
           "제공된 컨텍스트 밖의 내용은 답하지 말고 '근거 부족'이라고 말하세요. "
           "가능하면 금액·수치·조항을 우선 정리하고, 답변 끝에 [1], [2] 같은 근거 번호를 표기하세요. "
           "컨텍스트에 있는 질문 키워드는 반드시 본문에 포함하세요: " + ", ".join(kws)}
    user = {"role":"user","content":
            "이전 대화:\n" + "\n".join([f"- {m['role']}: {m['content']}" for m in recent]) +
            "\n\n현재 질문:\n" + question +
            "\n\n[컨텍스트]\n" + (ctx if ctx else "(없음)") +
            "\n\n지침: 위 컨텍스트와 대화 이력을 바탕으로 정확히 답하세요."}
    return [sys, user]

def main():
    idx = VSTORE_DIR / "faiss_index.idx"
    if not idx.exists():
        print("[ERR] 인덱스 없음. pipelines/ingest_local.sh 먼저 실행.")
        return
    index, metas = load_store()
    history, sess = [], f"sessions/chat_{int(time.time())}.jsonl"
    os.makedirs("sessions", exist_ok=True)

    print("대화형 RAG 시작(빈 줄 종료)")
    while True:
        q = input("\n❓ 질문> ").strip()
        if not q: print("종료"); break
        hint = " " + history[-1]["content"][:400] if history else ""
        qv = embed(q + hint)
        D, I = faiss_search(qv, index, TOPK_EMBED)
        selected, kws = pick_hybrid_best(I, D, metas, q, MAX_DOCS)
        if SAFE_MODE and not has_kw(selected, kws):
            print("\n🧠 답변:\n내부 근거가 부족합니다. 데이터 동기화 후 다시 시도해주세요.")
            continue
        ctx = build_context(selected, kws)
        msgs = build_messages(history, q, ctx, kws)
        resp = client.chat.completions.create(model=LLM_MODEL, messages=msgs, temperature=0.2)
        ans = resp.choices[0].message.content.strip()
        print("\n🧠 답변:\n", ans)
        history += [{"role":"user","content":q},{"role":"assistant","content":ans}]
        Path(sess).write_text("\n".join([json.dumps(x, ensure_ascii=False) for x in history]), encoding="utf-8")

if __name__ == "__main__":
    main()
