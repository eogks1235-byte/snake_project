"""경기 평점 시스템 — Phase 2 프레임워크.

설계 원칙:
  • 가중치(RATING_WEIGHTS)는 모듈 상수 — 한 곳만 수정하면 전체 튜닝
  • PlayerMatchStats: 한 선수의 한 경기 통계 컨테이너 (모든 카운터)
  • match_engine 은 이벤트마다 stats.shots_total += 1 같이 카운터만 증가
  • compute_rating() 이 매치 종료 시점에 한 번 호출되어 가중치 합산

통합 순서 (점진 구현):
  Step 1) match_engine.py: self.stats: dict = {(team_idx, name): PlayerMatchStats}
         + _spawn_players() 후 모든 선수에 빈 stats 생성
  Step 2) 이벤트 hook 지점에 카운터 증가 코드 삽입 (아래 HOOK_POINTS 참고)
  Step 3) _finalize_ratings 를 rating.compute_rating 호출로 교체
  Step 4) result.player_match_stats = self.stats 도 결과에 보존 (분석/UI용)
"""
from dataclasses import dataclass, field
from typing import Optional


# ──────────────────────────────────────────────────────────────────
# 가중치 — 모든 튜닝은 여기서. base 6.0 에 가산, 마지막에 [4.0, 10.0] clamp.
# 한 줄당 하나의 결정 — 주석으로 의도 명시.
# ──────────────────────────────────────────────────────────────────
RATING_WEIGHTS: dict = {
    # ── 골 / 어시 / 자책 (Phase 1 유지) ──
    'goal_outfield':        1.40,    # 필드 플레이어 골
    'goal_gk':              2.50,    # GK 골 (코너 어택 등 극히 희귀)
    'assist':               0.80,
    'own_goal':            -1.50,

    # ── 슛 (시도 자체에 미세 보너스) ──
    'shot_off_target':      0.02,    # 빗나간 슛
    'shot_on_target':       0.10,    # 유효슛
    'shot_blocked':         0.02,    # 막힌 슛 (블락)

    # ── 패스 / 빌드업 ──
    'key_pass':             0.15,    # 어시 안 됐지만 슈팅으로 연결된 패스
    'pass_completed':       0.004,   # 누적치 — 80패스 = +0.32
    'pass_missed':         -0.005,   # 빼앗긴 패스

    # ── 드리블 ──
    'dribble_won':          0.06,
    'dribble_failed':      -0.04,

    # ── 수비 (outfielder) ──
    'tackle_won':           0.12,
    'tackle_lost':         -0.08,    # 태클 시도 실패 → 파울 위험
    'interception':         0.08,    # 패스 라인 차단
    'clearance':            0.04,    # 박스 안 클리어링
    'foul_committed':      -0.10,    # 파울 (옐로 별도)
    'foul_won':             0.04,    # 파울 얻어냄 (세트피스 유발)

    # ── GK ──
    'save_routine':         0.20,    # 일반 세이브
    'save_difficult':       0.35,    # 결정적 세이브 (1대1 / xG 높은 슛)
    'pk_save':              1.50,    # PK 세이브 (정규/연장)
    'pk_shootout_save':     0.80,    # PK 승부차기 세이브 (라운드당)
    'goal_conceded_gk':    -0.20,

    # ── 카드 ──
    'yellow':              -0.40,
    'red':                 -1.50,

    # ── 종료 보정 ──
    'clean_sheet_gk':       1.00,
    'clean_sheet_def':      0.30,    # DEF 도 클린시트 보상 (작게)
    'sub_bonus_max':        0.30,    # 후반 교체 최대 보너스 (잔여 30분 풀)
    'short_minutes_cap':    0.80,    # ≤30분 출전: (rating-6.0)*0.80
    'clamp_lo':             4.0,
    'clamp_hi':            10.0,
}


# ──────────────────────────────────────────────────────────────────
# 경기당 선수 통계 컨테이너
# ──────────────────────────────────────────────────────────────────
@dataclass
class PlayerMatchStats:
    """한 선수 × 한 경기 통계. match_engine 이 이벤트마다 카운터 증가."""
    # 식별
    team_idx: int = -1
    name: str = ''
    role: str = ''            # GK / DEF / MID / FWD

    # 출전
    is_starter: bool = True
    sub_in_minute: int = 0
    sub_out_minute: int = 0   # 0 이면 끝까지 뛴 것
    on_pitch_end: bool = True
    minutes: int = 0          # finalize 시 계산

    # 골 / 어시
    goals: int = 0
    assists: int = 0
    own_goals: int = 0
    key_passes: int = 0

    # 슛 (총합 = on_target + off_target + blocked)
    shots_total: int = 0
    shots_on_target: int = 0
    shots_blocked: int = 0

    # 패스 / 빌드업
    passes_completed: int = 0
    passes_missed: int = 0

    # 드리블
    dribbles_won: int = 0
    dribbles_failed: int = 0

    # 수비
    tackles_won: int = 0
    tackles_lost: int = 0
    interceptions: int = 0
    clearances: int = 0
    fouls_committed: int = 0
    fouls_won: int = 0

    # GK
    saves_routine: int = 0
    saves_difficult: int = 0
    pk_saves: int = 0
    pk_shootout_saves: int = 0
    goals_conceded: int = 0
    clean_sheet: bool = False

    # 카드
    yellows: int = 0
    reds: int = 0


# ──────────────────────────────────────────────────────────────────
# 평점 산출 — 매치 종료 시 1회 호출
# ──────────────────────────────────────────────────────────────────
def compute_rating(stats: PlayerMatchStats, end_minute: int,
                    w: dict = None) -> float:
    """누적 통계 → 0~10 평점.
    end_minute: 정규/ET 포함 실제 경기 종료 분 (PK 제외)."""
    if w is None:
        w = RATING_WEIGHTS

    r = 6.0

    # 골 / 어시 / 자책
    r += stats.goals * (w['goal_gk'] if stats.role == 'GK'
                        else w['goal_outfield'])
    r += stats.assists * w['assist']
    r += stats.own_goals * w['own_goal']

    # 슛 — 시도/유효/막힘 분리
    shots_off = max(0, stats.shots_total
                     - stats.shots_on_target - stats.shots_blocked)
    r += shots_off * w['shot_off_target']
    r += stats.shots_on_target * w['shot_on_target']
    r += stats.shots_blocked * w['shot_blocked']

    # 패스 / 드리블
    r += stats.key_passes * w['key_pass']
    r += stats.passes_completed * w['pass_completed']
    r += stats.passes_missed * w['pass_missed']
    r += stats.dribbles_won * w['dribble_won']
    r += stats.dribbles_failed * w['dribble_failed']

    # 수비
    r += stats.tackles_won * w['tackle_won']
    r += stats.tackles_lost * w['tackle_lost']
    r += stats.interceptions * w['interception']
    r += stats.clearances * w['clearance']
    r += stats.fouls_committed * w['foul_committed']
    r += stats.fouls_won * w['foul_won']

    # GK
    r += stats.saves_routine * w['save_routine']
    r += stats.saves_difficult * w['save_difficult']
    r += stats.pk_saves * w['pk_save']
    r += stats.pk_shootout_saves * w['pk_shootout_save']
    r += stats.goals_conceded * w['goal_conceded_gk']

    # 카드
    r += stats.yellows * w['yellow']
    r += stats.reds * w['red']

    # 종료 보정 — 클린시트
    if stats.clean_sheet:
        if stats.role == 'GK':
            r += w['clean_sheet_gk']
        elif stats.role == 'DEF':
            r += w['clean_sheet_def']

    # 후반 sub 보너스 — 들어와서 시간 적게 뛰는 거 보정
    if not stats.is_starter and stats.sub_in_minute >= 45:
        remaining = max(0, end_minute - stats.sub_in_minute)
        r += w['sub_bonus_max'] * min(30, remaining) / 30

    # 출전 시간 짧으면 변동 폭 축소 (6.0 기준 ±20%)
    if stats.minutes <= 30:
        r = 6.0 + (r - 6.0) * w['short_minutes_cap']

    return max(w['clamp_lo'], min(w['clamp_hi'], r))


def compute_live_rating(stats: PlayerMatchStats, cur_minute: int,
                         w: dict = None) -> float:
    """실시간 라이브 평점 — 경기 중 표시용. finalize 미적용 버전.
    종료 보정(clean_sheet, sub_bonus, minutes_cap) 빼고 누적만."""
    if w is None:
        w = RATING_WEIGHTS

    r = 6.0
    r += stats.goals * (w['goal_gk'] if stats.role == 'GK'
                        else w['goal_outfield'])
    r += stats.assists * w['assist']
    r += stats.own_goals * w['own_goal']
    shots_off = max(0, stats.shots_total
                     - stats.shots_on_target - stats.shots_blocked)
    r += shots_off * w['shot_off_target']
    r += stats.shots_on_target * w['shot_on_target']
    r += stats.shots_blocked * w['shot_blocked']
    r += stats.key_passes * w['key_pass']
    r += stats.passes_completed * w['pass_completed']
    r += stats.passes_missed * w['pass_missed']
    r += stats.dribbles_won * w['dribble_won']
    r += stats.dribbles_failed * w['dribble_failed']
    r += stats.tackles_won * w['tackle_won']
    r += stats.tackles_lost * w['tackle_lost']
    r += stats.interceptions * w['interception']
    r += stats.clearances * w['clearance']
    r += stats.fouls_committed * w['foul_committed']
    r += stats.fouls_won * w['foul_won']
    r += stats.saves_routine * w['save_routine']
    r += stats.saves_difficult * w['save_difficult']
    r += stats.pk_saves * w['pk_save']
    r += stats.goals_conceded * w['goal_conceded_gk']
    r += stats.yellows * w['yellow']
    r += stats.reds * w['red']
    return max(w['clamp_lo'], min(w['clamp_hi'], r))


# ──────────────────────────────────────────────────────────────────
# match_engine 통합 HOOK 가이드 — 이벤트 발생 위치별 카운터 증가
# ──────────────────────────────────────────────────────────────────
HOOK_POINTS = """
match_engine.py 통합 지점 — 각 hook 에 stats 카운터 증가 1줄씩 삽입.

Match.__init__ 끝:
    self.stats: dict = {}    # (team_idx, name) → PlayerMatchStats
    for p in self.players:
        if p.name:
            self.stats[(p.team_idx, p.name)] = PlayerMatchStats(
                team_idx=p.team_idx, name=p.name, role=p.role,
                is_starter=p.is_starter, sub_in_minute=p.sub_in_minute,
            )

_score() — 골/어시/자책/실점:
    [scorer]      stats.goals += 1
    [assist]      stats.assists += 1
    [own_goal]    stats.own_goals += 1
    [conceding GK] stats.goals_conceded += 1

_step_shot_flight() — 슛 시도:
    [shooter]     stats.shots_total += 1
                  if will_be_goal:    stats.shots_on_target += 1
                  elif on_target_save: stats.shots_on_target += 1
                  elif blocked_for_corner: stats.shots_blocked += 1
    [shot_state.gk] (세이브 시):
                  if save_text == 'BIG SAVE': stats.saves_difficult += 1
                  else: stats.saves_routine += 1

_substitute_one() — 교체:
    [out_p stats] sub_out_minute = self.minute, on_pitch_end = False
    [in_p stats]  sub_in_minute = self.minute

새로 만들 hook 지점들 (기존 코드 없음, 추가 필요):

_try_pass() / _step_pass_flight():
    [passer]      성공 시 passes_completed += 1
                  실패 시 passes_missed += 1
                  성공 + 수신자가 다음 슛/골 → key_passes += 1

_check_pressure() 또는 carrier 교체 시:
    [defender]    압박 성공 + 탈취 → tackles_won += 1
                  압박 실패 + 파울 → tackles_lost += 1, fouls_committed += 1
    [carrier]     탈취당하면 dribbles_failed += 1
                  돌파 성공 → dribbles_won += 1

_check_foul():
    [fouled]      fouls_won += 1
    [fouler]      fouls_committed += 1
                  옐로 트리거 시 yellows += 1 (drumbeat: 누적 옐로 2개=레드)
                  레드 시 reds += 1

PK 승부차기 (_tick_pk):
    [GK]          PK 막으면 pk_shootout_saves += 1

_finalize_ratings 전면 교체:
    end_min = self.minute
    for p in self.players:
        key = (p.team_idx, p.name)
        if key not in self.stats: continue
        s = self.stats[key]

        # 분 시간 채움
        start_min = s.sub_in_minute if not s.is_starter else 0
        stop_min = s.sub_out_minute if s.sub_out_minute > 0 else end_min
        s.minutes = max(0, stop_min - start_min)
        s.on_pitch_end = (s.sub_out_minute == 0 and (s.is_starter or s.sub_in_minute > 0))

        # 클린시트 (출전 끝까지 가있고 팀 실점 0)
        conceded = (self.result.away_goals if p.team_idx == 0
                    else self.result.home_goals)
        s.clean_sheet = (conceded == 0 and s.minutes >= 60)

        self.player_ratings[key] = compute_rating(s, end_min)
"""
