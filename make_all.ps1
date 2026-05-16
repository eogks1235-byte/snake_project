# 풀 파이프라인 — 학습부터 모든 결과물까지 (~45분)
# 1) PPO 학습 (15분)
# 2) mp4 4개 (1분 진화 + 1분 battle + 1분 compare_video + compare 비교 영상)
# 3) PNG 차트 (학습곡선 + compare 막대)
# 4) 통계 표 (Cohen's d, p-value, CI)

$ErrorActionPreference = "Stop"

Write-Host "`n=== STEP 1/5: PPO 학습 (~15분) ===" -ForegroundColor Cyan
python -m neural.ppo_world --steps 5000000 --live-plot

Write-Host "`n=== STEP 2/5: 영상 1 - 진화 baseline ===" -ForegroundColor Cyan
python -m neural.main --record --headless --auto-exit-ticks 3600

Write-Host "`n=== STEP 3/5: 영상 2 - Battle ===" -ForegroundColor Cyan
python -m neural.battle --headless --ticks 3600

Write-Host "`n=== STEP 4/5: 영상 3 - Compare 나란히 ===" -ForegroundColor Cyan
python -m neural.compare_video --headless --ticks 3600

Write-Host "`n=== STEP 5/5: 통계 비교 + PNG 차트 + 추가 mp4 (~25분) ===" -ForegroundColor Cyan
python -m neural.ppo_world --compare --n-seeds 30 --ticks 1500 --plot --video

Write-Host "`n=== 전체 완료 ===" -ForegroundColor Green
Get-ChildItem "recordings\neural" | Format-Table Name, Length, LastWriteTime
