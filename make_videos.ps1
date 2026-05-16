# 3개 영상 한 번에 생성 — 모두 1분 (3600 frames @ 60fps)
# 사전 조건: recordings/neural/ppo_world_policy.pt 필요
#   (없으면: python -m neural.ppo_world --steps 5000000)

$ErrorActionPreference = "Stop"

Write-Host "`n=== 영상 1/3: 진화 baseline ===" -ForegroundColor Cyan
python -m neural.main --record --headless --auto-exit-ticks 3600

Write-Host "`n=== 영상 2/3: Battle (진화 + PPO 혼합) ===" -ForegroundColor Cyan
python -m neural.battle --headless --ticks 3600

Write-Host "`n=== 영상 3/3: Evolution vs PPO 나란히 ===" -ForegroundColor Cyan
python -m neural.compare_video --headless --ticks 3600

Write-Host "`n=== 완료 ===" -ForegroundColor Green
Write-Host "결과: recordings/neural/"
Get-ChildItem "recordings\neural\*.mp4" | Sort-Object LastWriteTime -Descending |
    Select-Object -First 5 | Format-Table Name, Length, LastWriteTime
