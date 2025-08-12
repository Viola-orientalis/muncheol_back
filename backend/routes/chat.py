from flask import Blueprint, request, jsonify
import numpy as np

from backend.services.llm_service import embed, chat
from backend.services.rag_service import search as rag_search, resolve_path_for_meta

bp = Blueprint("chat", __name__)

@bp.route("/api/ask-rag", methods=["POST"])
def ask_rag():
    data = request.get_json(silent=True) or {}
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

    # 컨텍스트 블록 문자열 생성 (백슬래시 문제 방지)
    context_block = "\n\n---\n\n".join(ctxs) if ctxs else "(컨텍스트 없음)"

    # 3) chat
    prompt = (
        "당신은 대한민국 법률 전문 챗봇입니다.\n"
        "아래 제공된 '컨텍스트'는 판례·법령·유권해석 등의 신뢰할 수 있는 자료입니다.\n"
        "반드시 다음 규칙을 지켜서 답변하세요.\n\n"
        "1. 답변은 **반드시 컨텍스트를 최우선 근거**로 작성합니다.\n"
        "2. 컨텍스트에 있는 내용은 문장 끝에 [번호] 형태로 표시합니다. "
        "번호는 컨텍스트 블록 순서에 대응합니다.\n"
        "3. 컨텍스트에 해당 근거가 전혀 없으면, '근거 부족'이라고 명확히 밝히고 "
        "추가로 일반적인 법률 지식을 참고하여 답할 수 있습니다.\n"
        "4. 금액, 형량, 조문 번호 등은 컨텍스트에 있으면 반드시 그대로 기재합니다.\n"
        "5. 추측이나 오해를 줄 수 있는 모호한 표현은 사용하지 않습니다.\n"
        "6. 최종 답변은 간결하지만 법률 문서 수준의 정확성을 유지합니다.\n\n"
        "-----\n"
        + context_block + "\n"
        "-----\n"
        f"질문: {question}"
    )
    answer = chat([{"role": "user", "content": prompt}])

    return jsonify({"ok": True, "answer": answer, "sources": hits})

@bp.route("/api/ask", methods=["POST", "OPTIONS"])
def ask_alias():
    if request.method == "OPTIONS":
        return ("", 204)
    return ask_rag()
