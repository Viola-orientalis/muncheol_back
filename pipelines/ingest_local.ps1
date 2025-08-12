Set-Location (Split-Path $MyInvocation.MyCommand.Path) | Out-Null
Set-Location ..

Write-Host "[1/4] data_preprocess.py"
python -m pipelines.data_preprocess

Write-Host "[2/4] text_splitter.py"
python -m pipelines.text_splitter

Write-Host "[3/4] embedder_incremental.py"
python -m pipelines.embedder_incremental

Write-Host "[4/4] retriever.py (스모크)"
"스모크 테스트" | python -m pipelines.retriever

Write-Host "✅ vectorstore/dev/current 업데이트 완료"
