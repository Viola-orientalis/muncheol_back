from flask import Blueprint, request, jsonify
import numpy as np

from backend.services.llm_service import embed, chat
from backend.services.rag_service import search as rag_search, resolve_path_for_meta

bp = Blueprint("chat", __name__)

@bp.route("/api/ask-rag", methods=["POST"])
def ask_rag():
    data = request.get_json(force=True) or {}
    question = (data.get("question") or "").strip()
    top_k = int(data.get("top_k", 5))
    if not question:
        return jsonify({"ok": False, "error": "question required"}), 400

    # 1) embed + rag search
    qv = np.array(embed(question), dtype="float32")
    hits = rag_search(qv, k=top_k)

    # 2) build context
    ctxs = []
    for i, h in enumerate(hits, 1):
        p = resolve_path_for_meta(h)
        try:
            with open(p, "r", encoding="utf-8") as f:
                snippet = f.read()[:1600]
        except Exception:
            snippet = ""
        ctxs.append(f"[{i}] {h['filename']} ({h['category']})\n{snippet}")

    # 3) chat
    prompt = (
        "다음 컨텍스트를 근거로 질문에 답하세요. 확실치 않으면 '근거 부족'이라고 말하세요.\n\n"
        + ("\n\n---\n\n".join(ctxs) if ctxs else "(컨텍스트 없음)") +
        f"\n\n질문: {question}"
    )
    answer = chat([{"role":"user","content": prompt}])

    return jsonify({"ok": True, "answer": answer, "sources": hits})
