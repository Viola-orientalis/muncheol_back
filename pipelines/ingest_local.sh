#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

echo "[1/4] data_preprocess.py"
python -m pipelines.data_preprocess

echo "[2/4] text_splitter.py"
python -m pipelines.text_splitter

echo "[3/4] embedder_incremental.py"
python -m pipelines.embedder_incremental

echo "[4/4] retriever.py (스모크)"
python -m pipelines.retriever <<< "스모크 테스트"

echo "✅ vectorstore/dev/current 업데이트 완료"
