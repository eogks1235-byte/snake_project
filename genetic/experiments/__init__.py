"""연구 모드 — 헤드리스 시뮬레이션 + 파라미터 스윕 + 통계 분석.

4개 실험:
  A: trait vs policy 적응 속도 비교 (genetic vs neural)
  B: 계절 주기 길이 → 성선택 ornament 진화
  C: 이주 비용 → 종 분화(biome 간 trait 분리) 임계
  D: 음식 비대칭 → Hawk 비율 ESS 이동

사용:
  python -m genetic.experiments              # 4개 다 실행
  python -m genetic.experiments --only b     # B만
  python -m genetic.experiments --seeds 30   # 시드 30개로
"""
