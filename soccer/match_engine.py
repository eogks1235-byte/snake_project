"""상태 기반 매치 엔진.

매 tick 모든 선수 위치를 시뮬레이션하고, carrier의 실제 위치/슛 레인/
GK 거리/수비수 차단을 평가해 슛/골 이벤트를 결정한다. 세트피스는 수비
압박에서 자연 발생 (파울 → FK/PK, 막힌 슛 → 코너).

시뮬 시간:
  TICKS_PER_HALF * 2 = 한 경기 (전반/후반)
  매 게임분 = SIM_TICKS_PER_MIN tick
"""
import math
import random
from dataclasses import dataclass, field
from typing import Optional, List

from .teams import Team
from .formations import (
    positions_home, positions_away, matchup_mult,
    DEFENSIVE_MAP, AGGRESSIVE_MAP, DESPERATE_MAP, LOCKDOWN_MAP, PARKING_BUS,
)


TICKS_PER_HALF = 45                  # 한 경기 90 게임분
SIM_TICKS_PER_MIN = 13               # 1 게임분 = 13 시뮬 tick → 1170 tick = 39s @ 30fps

# ── 상태 기반 슛 파라미터 ────────────────────────────────
SHOT_PROB_PER_TICK_BASE = 0.55       # carrier가 양질의 슛 기회일 때 per-tick 시도 확률
SHOT_ZONE_X_RATIO = 0.50             # 어택킹 써드 진입 x 비율 (이후 슛 가능)
ONE_V_ONE_GK_DIST = 14.0             # GK까지 이 거리 안 → "1대1" 보너스
BLOCKER_LANE_HALF_Y = 4.5            # 슛 라인 y-반경 (블로커 카운트용)
SHOT_FLIGHT_TICKS = 16               # 슛 비행 ticks
GOAL_K = 12.0                        # 골 확률 시그모이드 기울기
GOAL_P_SCALE = 0.68                  # 골 확률 전역 스케일 (캘리브레이션용)
ON_TARGET_RATE_BASE = 0.32           # 슛 중 유효슛 기본 비율 (실력 보정 추가)

# 파울 / 세트피스 발생 파라미터
FOUL_PROB_PER_TICK_BASE = 0.018      # 수비 압박 시 tick당 파울 확률
PEN_BOX_X_RATIO = 0.78               # 페널티 박스 x 시작 (공격팀 기준)
PEN_BOX_Y_RANGE = (0.20, 0.80)       # 페널티 박스 y 범위
PRESSURE_RADIUS = 3.2                # 수비수 압박 반경
CORNER_ON_BLOCK_PROB = 0.72          # 막힌 슛이 코너로 이어질 확률
SETPIECE_SETUP_TICKS = 22            # 코너/FK 셋업 ticks (그 후 슛 비행)
PENALTY_SETUP_TICKS = 30             # PK 셋업 ticks

# ── 2026 북중미 월드컵 매치 조건 ─────────────────────────
ALTITUDE_BOOST = {                   # 고도 → 슛 확률 배율 (멕시코시티 효과)
    'low':  1.00,
    'mid':  1.06,
    'high': 1.18,                    # 멕시코시티 ~2240m
}
HOT_LATE_PENALTY = 0.86              # 더위 + 후반 + heat_native 아닌 팀: 수비 ×0.86
ALT_LATE_PENALTY = 0.92              # HIGH-ALT + 후반 + altitude_native 아닌 팀: 수비 ×0.92
HOT_LATE_MIN = 60                    # "후반 시작" 분 (이 이후 적응 효과)
LAST_ROUND_ATK_BOOST = 1.12          # 예선 마지막 라운드 양 팀 attack ↑ (3위 진출 다툼)

# 공/캐리어 안정화 — 영상 품질: 부드럽고 떨림 없게
CARRIER_TTL_RANGE = (34, 72)         # 캐리어 더 오래 유지 → 공 흐름 끊김 ↓
PASS_PROB_PER_TICK = 0.018           # 패스 빈도 ↓ → 공이 덜 튐
BALL_FRICTION = 0.965                # 마찰 ↑↑ → 부드러운 글라이드
BALL_FOLLOW_K = 0.065                # carrier 끌림 ↓ → 점착 느낌 ↓, 떨림 제거
BALL_STEP_RATIO = 0.42               # ball position step (낮을수록 슬로우)

# ── 시네마틱 phase (게임분 비례) ───────────────────────────
GOAL_HOLD_TICKS = 60                 # 골 직후 공이 골대 안에 머무는 시간 (~2초 @ 30fps)
KICKOFF_RESET_TICKS = 28             # 일반 골 → 공/선수가 중앙으로 리셋되는 시간
SETPIECE_DISPERSE_TICKS = 90         # 세트피스 골 → 박스에 모인 선수들이 흩어지는 시간 (~3초)
KICKOFF_DURATION = GOAL_HOLD_TICKS + KICKOFF_RESET_TICKS  # 일반 골 기본 kickoff 시간
SHOT_FLIGHT_TICKS = 16               # 슛 비행 — 좀 더 길게 + 더 강한 가속감

# 선수 lerp / 호흡 모션 ── 랜덤 jitter 대신 부드러운 sin
PLAYER_LERP = 0.045                  # 선수 위치 추적 (낮을수록 부드러움)
BUILDUP_PLAYER_LERP = 0.032          # buildup은 더 느긋하게
BREATH_AMPLITUDE = 0.10              # sin 호흡 크기 (px) — 영상에서 거의 안 보임
BREATH_FREQ = 0.045                  # sin 주기 (낮을수록 천천히 호흡)

# 역할별 home 구속 강도 (1.0=고정, 0.0=공만 따라감) — 포메이션 유지력 ↑
HOME_PULL = {
    'GK':  0.92,    # 골키퍼는 거의 고정
    'DEF': 0.72,    # 수비 라인 유지 강화
    'MID': 0.52,    # 미드필더 — 포메이션 형태 유지
    'FWD': 0.40,    # 공격수도 포지션 인지하며 움직임
}
# Wing 선수일수록 y는 더 강하게 home 유지 → 좌우 폭 활용 ↑
# 단, 너무 크면 와이드 선수가 라인을 타고 "고정"되어 보이므로 작게
HOME_PULL_Y_BOOST = 0.08
HOME_PULL_Y_MAX = 0.70      # py 상한 — 와이드 선수도 ball에 따라 좌우 ↕ 활동

# 전술적 다양성 — 같은 role 그룹이 똑같이 움직이지 않게
FULLBACK_Y_DIST_THRESHOLD_RATIO = 0.25     # |home_y - 중앙| / 절반 > 0.25 → 풀백
OVERLAP_PUSH_RATIO = 0.52                   # 풀백 오버래핑 시 전진 비율
TACTICAL_DRIFT_AMP_X = 2.2                  # 전술 드리프트 x amplitude (px) — 더 잔잔하게
TACTICAL_DRIFT_AMP_Y = 1.1                  # y amplitude
TACTICAL_DRIFT_PERIOD_X = 300               # tick — 주기 길게 → 천천히 흐름
TACTICAL_DRIFT_PERIOD_Y = 400
POSITION_SWAP_PERIOD = 460                  # tick — 포지션 스왑 트리거 주기 (덜 자주)
POSITION_SWAP_PROB = 0.30                   # 트리거 시 실제 스왑 확률 (덜 자주)

# ── 교체 + 스태미너 ──────────────────────────────────────
SUB_WINDOW_START_MIN = 60                   # 60분부터 교체 가능
SUB_WINDOW_END_MIN = 88                     # 88분 교체 마감 (FIFA 룰)
MAX_SUBS_PER_TEAM = 5                       # FIFA 5명 (post-COVID)
EXTRA_TIME_BONUS_SUB = 1                    # 연장전 1명 추가 교체 허용
EXTRA_TIME_DURATION = 30                    # 연장전 30분
ET_WINNER_STAMINA = 90                      # ET 승리팀 다음 매치 시작 stamina
SUB_PROB_PER_GAME_MIN = 0.20                # 5명 다 사용하도록 빈도 ↑
FATIGUE_START_MIN = 75                      # 75분부터 선발 페이스 다운
FATIGUE_PER_MIN = 0.010                     # 1%/분 → 90분에 0.85 (선발)
SUB_FRESH_BOOST = 1.05                      # 후보 들어온 직후 1.05배
SUB_FRESH_DURATION_MIN = 15                 # 신선 효과 유지 분

# ── 경기장 친숙도 (재방문 보너스) ────────────────────────
VENUE_FAMILIAR_ATK_MULT = 1.04              # 같은 경기장 재방문 시 공격 ×1.04
VENUE_FAMILIAR_DEF_MULT = 1.03              # 수비 ×1.03

# ── 추가시간 / 자책골 / 어시스트 / PK ─────────────────────
STOPPAGE_RANGE = (3, 9)                     # 90 + N분 종료 (N ∈ [3, 9])
OWN_GOAL_CHANCE = 0.022                     # 골당 자책골 확률 (실제 ~2%)
ASSIST_CHANCE = 0.70                        # 골당 어시스트 부여 확률
ASSIST_ROLE_W = {'MID': 0.50, 'FWD': 0.32, 'DEF': 0.16, 'GK': 0.02}


def _ease_in_out_sine(t: float) -> float:
    return -(math.cos(math.pi * max(0.0, min(1.0, t))) - 1) / 2


def _color_clash(c1: tuple, c2: tuple, threshold: int = 90) -> bool:
    """두 색이 너무 비슷하면 True (각 채널 차이 합 < threshold)."""
    d = abs(c1[0] - c2[0]) + abs(c1[1] - c2[1]) + abs(c1[2] - c2[2])
    return d < threshold


def _apply_style(tag: str, atk: float, df: float) -> tuple:
    """style_tag → (atk, def, mid_bonus) 보정."""
    if tag == 'possession':
        return atk, df, 1.15
    if tag == 'counter':
        return atk * 1.10, df, 1.0
    if tag == 'parking-bus':
        return atk, df * 1.20, 1.0
    return atk, df, 1.0


# ──────────────────────────────────────────────────────────────
# 결과 모델
# ──────────────────────────────────────────────────────────────

@dataclass
class MatchEvent:
    minute: int
    kind: str        # 'goal' | 'shot' | 'kickoff' | 'half' | 'fulltime' | 'pk'
    team_idx: int    # 0 또는 1 (해당 없으면 -1)
    text: str = ''


@dataclass
class MatchResult:
    home: Team
    away: Team
    home_goals: int = 0
    away_goals: int = 0
    home_pk: int = 0
    away_pk: int = 0
    went_to_pk: bool = False
    went_to_et: bool = False              # 연장전 진입 여부
    decided_in_et: bool = False           # 연장전에서 승부 결정
    events: List[MatchEvent] = field(default_factory=list)
    # 선수별 통계 (Tournament awards 계산용)
    player_goals: dict = field(default_factory=dict)   # (team_idx, name) → goals
    player_assists: dict = field(default_factory=dict) # (team_idx, name) → assists
    own_goals: int = 0                                  # 매치 내 자책골 개수
    appearances: list = field(default_factory=list)    # [(team_idx, name, role, rating, is_star)]

    @property
    def winner(self) -> Optional[Team]:
        if self.went_to_pk:
            return self.home if self.home_pk > self.away_pk else self.away
        if self.home_goals > self.away_goals:
            return self.home
        if self.away_goals > self.home_goals:
            return self.away
        return None  # 무승부

    @property
    def loser(self) -> Optional[Team]:
        w = self.winner
        if w is None:
            return None
        return self.away if w is self.home else self.home

    def score_str(self) -> str:
        base = f'{self.home_goals}-{self.away_goals}'
        if self.went_to_pk:
            base += f' (PK {self.home_pk}-{self.away_pk})'
        return base


# ──────────────────────────────────────────────────────────────
# 시각화용 점 (선수)
# ──────────────────────────────────────────────────────────────

@dataclass
class Player:
    team_idx: int      # 0 = home, 1 = away
    role: str          # GK / DEF / MID / FWD
    x: float
    y: float
    home_x: float      # 포메이션상 홈 위치
    home_y: float
    is_fullback: bool = False     # 외곽 DEF (오버래핑 가능)
    drift_phase_x: float = 0.0     # 전술 드리프트 phase
    drift_phase_y: float = 0.0
    name: str = ''                 # 선수명 (스쿼드 매핑)
    rating: int = 70               # 능력치
    is_star: bool = False          # ★ 스타
    pk_taker: bool = False         # PK 전담
    is_starter: bool = True        # 선발 (vs 후보)
    on_pitch: bool = True          # 현재 출전 중
    sub_in_minute: int = 0         # 후보의 경우 들어온 분 (선발은 0)
    subbed_off: bool = False       # 한 번 교체 나감 → 다시 못 들어옴 (FIFA 룰)

    @property
    def can_carry(self) -> bool:
        return self.role in ('MID', 'FWD')


# ──────────────────────────────────────────────────────────────
# 매치 엔진
# ──────────────────────────────────────────────────────────────

class Match:
    """한 경기를 tick 단위로 진행. tick()을 매 프레임 호출."""

    PITCH_W = 80.0
    PITCH_H = 50.0

    def __init__(self, home: Team, away: Team, knockout: bool,
                 rng: random.Random, precompute: bool = False,
                 altitude: str = 'low', hot: bool = False,
                 last_round_push: bool = False,
                 home_familiar: bool = False,
                 away_familiar: bool = False,
                 home_starting_stamina: int = 100,
                 away_starting_stamina: int = 100):
        self.home = home
        self.away = away
        self.knockout = knockout       # True면 무승부 시 PK
        self.rng = rng                 # 게임 결정용 (점유/슛/골)
        # 시각 jitter용 별도 RNG — 게임 결정 RNG를 오염시키지 않음.
        # 같은 base seed에서 fast/full tick 결과가 일치하도록 분리.
        self.visual_rng = random.Random(rng.random() * 2_000_000_000)

        # 2026 매치 조건
        self.altitude = altitude       # 'low' / 'mid' / 'high'
        self.is_hot = hot              # 더운 경기장 (후반 수비 감쇄)
        self.last_round_push = last_round_push  # 예선 마지막 라운드 양 팀 attack ↑

        self.minute = 0                # 0~90
        self.tick_count = 0
        self.result = MatchResult(home=home, away=away)

        # 키트 충돌 — 양 팀 색이 너무 비슷하면 어웨이는 secondary로
        self.away_use_secondary = _color_clash(home.color, away.color)

        # 친숙도/조건 저장 (재계산 시 사용)
        self.home_familiar = home_familiar
        self.away_familiar = away_familiar

        # 매치별 포메이션 결정트리 — 상대 강약 + 스타일 고려
        home_form = self._decide_initial_formation(home, away)
        away_form = self._decide_initial_formation(away, home)
        # 강팀이 약팀+버스주차 만나면 한 번 더 공격적으로 (양방향 조정)
        if (home.data.fifa_points - away.data.fifa_points > 100
                and away_form in PARKING_BUS):
            home_form = AGGRESSIVE_MAP.get(home.formation, home_form)
        if (away.data.fifa_points - home.data.fifa_points > 100
                and home_form in PARKING_BUS):
            away_form = AGGRESSIVE_MAP.get(away.formation, away_form)
        self.home_formation_used = home_form
        self.away_formation_used = away_form
        self.home_used_secondary = (home_form != home.formation)
        self.away_used_secondary = (away_form != away.formation)

        # 효과적 style — 포메이션 종류로 자동 결정
        self.home_style_effective = self._style_for_formation(home, home_form)
        self.away_style_effective = self._style_for_formation(away, away_form)

        # 전술 변경 추적
        self.tactical_changes = [0, 0]      # 팀별 도중 변경 횟수
        # 리액티브 전술 — 상대가 바꾸면 N분 후 카운터 검사
        self.pending_counter_minute = [0, 0]   # 0=비활성, >0=해당 분에 강제 검사
        # 골 직후 즉시 검사 방지 중복 락 (같은 분 중복 트리거 방지)
        self._last_reactive_minute = -1
        self._last_tactical_minute = -1     # 같은 분에 중복 트리거 방지

        # 능력치 (style + 고도 + last_round + 매치업 + 친숙도 통합) — 계산
        self._recompute_strengths()

        # wc_titles 보너스 — 골 확률 곱셈 (정적, 변경 안 됨)
        self.h_goal_bonus = 1.0 + home.wc_titles * 0.020
        self.a_goal_bonus = 1.0 + away.wc_titles * 0.020

        # 시각화 선수 + 공
        self.players: List[Player] = []
        self._spawn_players()
        self.ball_x = self.PITCH_W / 2
        self.ball_y = self.PITCH_H / 2
        self.ball_vx = 0.0
        self.ball_vy = 0.0
        self.possession_idx = 0 if self.rng.random() < self.home_possession else 1
        self.goal_flash = 0           # 골 직후 화면 강조 카운트
        self.goal_flash_team = -1

        # 안정 carrier — 매 tick 무작위 선택이 아니라 일정 시간 유지
        self.carrier_idx: Optional[int] = None
        self.carrier_ttl: int = 0

        # 패스 in-flight 상태 — 공이 목표 지점으로 비행 중일 때
        # 수신자는 그 지점으로 스프린트해서 받음 (lead-pass)
        self.pass_in_flight: bool = False
        self.pass_target_idx: Optional[int] = None
        self.pass_dest_x: float = 0.0          # 공이 도착할 좌표 (lead-pass)
        self.pass_dest_y: float = 0.0
        self.pass_start_x: float = 0.0
        self.pass_start_y: float = 0.0
        self.pass_total_ticks: int = 0
        self.pass_tick: int = 0

        # 매치 통계 (실시간 누적)
        self.team_shots = [0, 0]            # 슛 시도 횟수
        self.team_on_target = [0, 0]        # 유효슛
        self.team_xG = [0.0, 0.0]            # 누적 기대 득점
        self.possession_ticks = [0, 0]       # 점유 tick 수 (점유율 계산용)
        self.player_goals: dict = {}         # (team_idx, name) → 골 수
        self.player_assists: dict = {}       # (team_idx, name) → 어시스트 수
        self.own_goals = 0                   # 매치 내 자책골
        self.subs_made = [0, 0]              # 팀별 교체 수 (0~5)
        self._last_min_subbed = -1           # 같은 분에 중복 트리거 방지

        # 추가시간 — self.rng 사용 (게임 결정 RNG, fast/full 일관성)
        self.regulation_end_minute = 90 + self.rng.randint(*STOPPAGE_RANGE)

        # 연장전
        self.in_extra_time = False
        self.extra_time_end_minute = 0  # ET 진입 시 확정
        self.max_subs_for_team = MAX_SUBS_PER_TEAM   # ET 시 +1로 변경

        # ET 시작 stamina (Tournament.team_stamina_carry 에서 전달)
        self.home_starting_stamina = home_starting_stamina
        self.away_starting_stamina = away_starting_stamina

        # PK 5인 키커 명단 (PK 진입 시 결정됨)
        self.home_pk_kickers: list = []
        self.away_pk_kickers: list = []

        # 상태 머신: normal / shot_in_flight / setpiece_shot / kickoff
        self.phase = 'normal'
        self.phase_start_tick = 0
        self.visual_possession_idx = self.possession_idx

        # 전·후반 진영 교체 — False면 team 0이 +x 방향 공격
        self.half_swap: bool = False

        # 진행 중인 슛 상태 (shot_in_flight 동안만 유효)
        # {origin_x, origin_y, target_y, will_be_goal, goal_type, blocked_for_corner,
        #  team_idx, scorer_name, assist_name, save_text}
        self.shot_state: Optional[dict] = None
        # 진행 중인 세트피스 (setpiece_shot 동안만 유효)
        # {kind, sp_x, sp_y, team_idx, will_be_goal, scorer_name, assist_name}
        self.setpiece_state: Optional[dict] = None

        # 매치 내 발생한 골 로그 (실시간) — setpiece_counts 등 stats용
        # {minute, tick, team_idx, type, scorer, assist}
        self.goals_log: list = []
        # kickoff 페이즈에서 참조할 마지막 골 type ('open_play' / 'corner' / 'free_kick' / 'penalty')
        self.last_goal_type: str = 'open_play'
        self.last_goal_team: int = -1

        self.finished = False
        self.in_pk = False
        self.pk_round = 0
        self.events: List[MatchEvent] = self.result.events
        self.events.append(MatchEvent(0, 'kickoff', -1, 'KICK-OFF'))

    # ── 시각화 셋업 ──────────────────────────────────────────

    def _pick_formation(self, my_team: Team, opp_team: Team) -> str:
        """호환용 — _decide_initial_formation 으로 위임."""
        return self._decide_initial_formation(my_team, opp_team)

    def _decide_initial_formation(self, my_team: Team, opp_team: Team) -> str:
        """매치 시작 시 포메이션 결정.
           - 50pt 이상 약함 → team.secondary_formation (없으면 DEFENSIVE_MAP fallback)
           - 100pt 이상 강함 + 상대 parking-bus 스타일 → AGGRESSIVE
           - 그 외 → primary
        """
        diff = my_team.data.fifa_points - opp_team.data.fifa_points
        if diff < -50:
            # team.secondary_formation을 우선 사용 (팀별 커스텀 가능)
            sec = getattr(my_team, 'secondary_formation', '') or ''
            if sec and sec != my_team.formation:
                return sec
            return DEFENSIVE_MAP.get(my_team.formation, my_team.formation)
        if diff > 100 and opp_team.style_tag == 'parking-bus':
            return AGGRESSIVE_MAP.get(my_team.formation, my_team.formation)
        return my_team.formation

    def _style_for_formation(self, team: Team, used_form: str) -> str:
        """효과적 style — 사용 포메이션이 primary면 원래 style, 아니면 변형 style."""
        if used_form == team.formation:
            return team.style_tag
        if used_form in PARKING_BUS:
            return 'counter'                 # 5-3-2 / 5-4-1 → counter style
        if used_form in {'3-4-3', '4-2-4'}:
            return 'possession'              # 공격형 변형 → 점유 위주
        if used_form == '4-1-4-1':
            return 'parking-bus'             # 락다운 — DM shield + 깊은 블록
        return team.style_tag

    def _recompute_strengths(self):
        """현재 포메이션/스타일/조건 기반으로 strength 재계산.
        init + 도중 포메이션 변경 시 양쪽에서 호출."""
        home, away = self.home, self.away
        h_atk = home.attack + 0.5 * home.midfield
        a_atk = away.attack + 0.5 * away.midfield
        h_def = home.defense + 0.5 * home.midfield + 0.3 * home.keeper
        a_def = away.defense + 0.5 * away.midfield + 0.3 * away.keeper

        # style — 효과적 style 사용 (포메이션 따라 자동)
        h_atk, h_def, h_mid_bonus = _apply_style(self.home_style_effective, h_atk, h_def)
        a_atk, a_def, a_mid_bonus = _apply_style(self.away_style_effective, a_atk, a_def)

        # 고도
        alt_mult = ALTITUDE_BOOST.get(self.altitude, 1.0)
        h_atk *= alt_mult
        a_atk *= alt_mult

        # 예선 마지막 라운드
        if self.last_round_push:
            h_atk *= LAST_ROUND_ATK_BOOST
            a_atk *= LAST_ROUND_ATK_BOOST

        # 포메이션 상성
        h_atk *= matchup_mult(self.home_formation_used, self.away_formation_used)
        a_atk *= matchup_mult(self.away_formation_used, self.home_formation_used)

        # 친숙도
        if self.home_familiar:
            h_atk *= VENUE_FAMILIAR_ATK_MULT
            h_def *= VENUE_FAMILIAR_DEF_MULT
        if self.away_familiar:
            a_atk *= VENUE_FAMILIAR_ATK_MULT
            a_def *= VENUE_FAMILIAR_DEF_MULT

        self.h_attack_strength = h_atk
        self.a_attack_strength = a_atk
        self.h_defense_strength = h_def
        self.a_defense_strength = a_def

        # 점유율 — possession 스타일은 미드 가중치 ↑
        h_poss = home.midfield * h_mid_bonus + 0.5 * home.attack
        a_poss = away.midfield * a_mid_bonus + 0.5 * away.attack
        self.home_possession = h_poss / (h_poss + a_poss)

    # ── 도중 전술 변경 ────────────────────────────────────

    def _check_tactical_change(self):
        """전술 변경 트리거:
        - 정기: 50/70/80분
        - 리액티브: pending_counter_minute (상대 포메이션 변경 후 N분 후 카운터)
        """
        scheduled = self.minute in (50, 70, 80)
        # 카운터가 예약된 팀
        counter_teams = [i for i in (0, 1)
                          if self.pending_counter_minute[i] != 0
                          and self.minute >= self.pending_counter_minute[i]]
        if not scheduled and not counter_teams:
            return

        # 정기 검사 — 중복 방지
        if scheduled and self.minute != self._last_tactical_minute:
            self._last_tactical_minute = self.minute
            for team_idx in (0, 1):
                if self.tactical_changes[team_idx] >= 3:    # 팀당 최대 3회 (확장)
                    continue
                self._maybe_tactical_change(team_idx)

        # 카운터 검사 — 상대가 바꾼 뒤 N분 후
        for team_idx in counter_teams:
            self.pending_counter_minute[team_idx] = 0
            if self.tactical_changes[team_idx] >= 3:
                continue
            self._maybe_tactical_change(team_idx, reactive=True)

    def _check_post_goal_tactical(self, scoring_team_idx: int):
        """골 직후 즉시 검사 — 지고 있는 팀이 가장 강하게 반응."""
        if self.minute == self._last_reactive_minute:
            return
        self._last_reactive_minute = self.minute
        # 양 팀 다 검사 — 지고 있는 팀은 공격적, 이기는 팀은 잠그기
        for team_idx in (0, 1):
            if self.tactical_changes[team_idx] >= 3:
                continue
            self._maybe_tactical_change(team_idx, reactive=True)

    def _maybe_tactical_change(self, team_idx: int, reactive: bool = False):
        team = self.home if team_idx == 0 else self.away
        opp = self.away if team_idx == 0 else self.home
        if team_idx == 0:
            score_diff = self.result.home_goals - self.result.away_goals
        else:
            score_diff = self.result.away_goals - self.result.home_goals
        cur_form = (self.home_formation_used if team_idx == 0
                     else self.away_formation_used)
        opp_form = (self.away_formation_used if team_idx == 0
                     else self.home_formation_used)
        pts_diff = team.data.fifa_points - opp.data.fifa_points

        new_form = None

        # ── 리액티브 (골 직후 / 상대 변경 후 카운터): 분 무관 ──
        if reactive:
            # 지고 있음 → 공격적 (강팀이면 더 극단)
            if score_diff <= -1:
                if pts_diff > 80 or self.minute >= 70 or score_diff <= -2:
                    new_form = DESPERATE_MAP.get(team.formation)
                else:
                    new_form = AGGRESSIVE_MAP.get(team.formation)
            # 이기고 있음 + 후반 → 잠그기
            elif score_diff >= 1 and self.minute >= 60:
                new_form = LOCKDOWN_MAP.get(team.formation)
            # 동점 + 상대 parking → 뚫기
            elif score_diff == 0 and opp_form in PARKING_BUS and pts_diff > 50:
                new_form = AGGRESSIVE_MAP.get(team.formation)
            # 동점 + 상대 공격적 + 우리 약함 → 수비 강화
            elif (score_diff == 0
                   and opp_form in ('4-3-3', '3-4-3', '4-2-4')
                   and pts_diff < -50):
                new_form = DEFENSIVE_MAP.get(team.formation)
        else:
            # ── 정기 검사 (50/70/80분) ──
            # 80분 + 지고 있음 → DESPERATE (all-out)
            if self.minute >= 80 and score_diff <= -1:
                new_form = DESPERATE_MAP.get(team.formation)
            # 70분 + 지고 있음 → AGGRESSIVE
            elif self.minute >= 70 and score_diff <= -1:
                new_form = AGGRESSIVE_MAP.get(team.formation)
            # 70분 + 큰 리드(2+) → LOCKDOWN
            elif self.minute >= 70 and score_diff >= 2:
                new_form = LOCKDOWN_MAP.get(team.formation)
            # 50분 + 동점 + 우리 강 + 상대 parking → AGGRESSIVE 강제
            elif self.minute >= 50 and score_diff == 0:
                if pts_diff > 100 and opp_form in PARKING_BUS:
                    new_form = AGGRESSIVE_MAP.get(team.formation)
                elif pts_diff < -80 and cur_form not in PARKING_BUS:
                    new_form = DEFENSIVE_MAP.get(team.formation)
            # 50분 + 1점 차로 이기는 중 → LOCKDOWN
            elif self.minute >= 50 and score_diff == 1:
                new_form = LOCKDOWN_MAP.get(team.formation)

        if new_form and new_form != cur_form:
            self._change_formation_for_team(team_idx, new_form)
            # 상대팀 카운터 예약 — 3~5분 후 검사
            opp_idx = 1 - team_idx
            if self.tactical_changes[opp_idx] < 3:
                self.pending_counter_minute[opp_idx] = self.minute + self.visual_rng.randint(3, 5)
            self.tactical_changes[team_idx] += 1

    def _change_formation_for_team(self, team_idx: int, new_form: str):
        """on-pitch 11명을 새 포메이션 슬롯에 greedy 재배치."""
        pos_fn = positions_home if team_idx == 0 else positions_away
        new_slots = pos_fn(new_form)
        slot_data = [
            {'role': r, 'home_x': fx * self.PITCH_W,
             'home_y': fy * self.PITCH_H}
            for r, fx, fy in new_slots
        ]
        on_pitch = [p for p in self.players
                     if p.team_idx == team_idx and p.on_pitch]
        if len(on_pitch) != len(slot_data):
            return  # 비정상 — 무시

        # Greedy 매칭: same-role 우선, 그 다음 거리 최소
        used_pids = set()
        for slot in slot_data:
            same_role = [p for p in on_pitch
                         if id(p) not in used_pids
                         and p.role == slot['role']]
            cands = same_role if same_role else [
                p for p in on_pitch if id(p) not in used_pids
            ]
            if not cands:
                continue
            best = min(cands,
                        key=lambda p: ((p.home_x - slot['home_x']) ** 2
                                        + (p.home_y - slot['home_y']) ** 2))
            used_pids.add(id(best))
            best.home_x = slot['home_x']
            best.home_y = slot['home_y']
            best.role = slot['role']
            center_y = self.PITCH_H * 0.5
            fb_thr = self.PITCH_H * 0.5 * FULLBACK_Y_DIST_THRESHOLD_RATIO
            best.is_fullback = (best.role == 'DEF'
                                 and abs(best.home_y - center_y) > fb_thr)

        # 포메이션 + style 갱신
        team = self.home if team_idx == 0 else self.away
        if team_idx == 0:
            self.home_formation_used = new_form
            self.home_used_secondary = (new_form != team.formation)
            self.home_style_effective = self._style_for_formation(team, new_form)
        else:
            self.away_formation_used = new_form
            self.away_used_secondary = (new_form != team.formation)
            self.away_style_effective = self._style_for_formation(team, new_form)

        # 능력치 재계산
        self._recompute_strengths()

        # 이벤트
        self.events.append(MatchEvent(
            self.minute, 'tactical', team_idx,
            f'TACTIC ({team.code}) {new_form}'
        ))

    def _spawn_players(self):
        """팀별 포메이션 — 홈은 좌→우, 어웨이는 우→좌.
           각 선수에 fullback 플래그 + 드리프트 phase + 스쿼드 매칭."""
        center_y = self.PITCH_H * 0.5
        fb_threshold = self.PITCH_H * 0.5 * FULLBACK_Y_DIST_THRESHOLD_RATIO

        # 팀별 스쿼드를 role별 풀로 분류 (rating 높은 순)
        def role_pools(team) -> dict:
            pools = {'GK': [], 'DEF': [], 'MID': [], 'FWD': []}
            for p in team.squad:
                if p.role in pools:
                    pools[p.role].append(p)
            for role in pools:
                pools[role].sort(key=lambda p: -p.rating)
            return pools

        h_pools = role_pools(self.home)
        a_pools = role_pools(self.away)
        h_used = {role: 0 for role in h_pools}
        a_used = {role: 0 for role in a_pools}

        def next_pdata(team_idx: int, role: str):
            pools = h_pools if team_idx == 0 else a_pools
            used = h_used if team_idx == 0 else a_used
            pool = pools.get(role, [])
            if used[role] < len(pool):
                p = pool[used[role]]
                used[role] += 1
                return p
            return None

        for team_idx, formation, positions_fn in [
            (0, self.home_formation_used, positions_home),
            (1, self.away_formation_used, positions_away),
        ]:
            # 선발 11명 — 포메이션 슬롯에 배치
            for role, fx, fy in positions_fn(formation):
                x = fx * self.PITCH_W
                y = fy * self.PITCH_H
                is_fb = (role == 'DEF' and abs(y - center_y) > fb_threshold)
                pdata = next_pdata(team_idx, role)
                self.players.append(Player(
                    team_idx=team_idx, role=role, x=x, y=y,
                    home_x=x, home_y=y,
                    is_fullback=is_fb,
                    drift_phase_x=self.visual_rng.random() * math.tau,
                    drift_phase_y=self.visual_rng.random() * math.tau,
                    name=pdata.name if pdata else '',
                    rating=pdata.rating if pdata else 70,
                    is_star=pdata.is_star if pdata else False,
                    pk_taker=pdata.pk_taker if pdata else False,
                    is_starter=True, on_pitch=True,
                ))
            # 후보 5명 — 벤치 (off-pitch). 남은 풀에서 가져옴
            pools = h_pools if team_idx == 0 else a_pools
            used = h_used if team_idx == 0 else a_used
            for role in ('GK', 'DEF', 'MID', 'FWD'):
                pool = pools.get(role, [])
                while used[role] < len(pool):
                    pdata = pool[used[role]]
                    used[role] += 1
                    self.players.append(Player(
                        team_idx=team_idx, role=role,
                        x=-100, y=-100,        # 화면 밖
                        home_x=-100, home_y=-100,
                        is_fullback=False,
                        drift_phase_x=0.0, drift_phase_y=0.0,
                        name=pdata.name, rating=pdata.rating,
                        is_star=pdata.is_star, pk_taker=pdata.pk_taker,
                        is_starter=False, on_pitch=False,
                    ))

    # ── 메인 루프 ───────────────────────────────────────────

    def tick(self):
        """시각화 tick — 모든 시뮬레이션 + 화면용 보간."""
        if self.finished:
            return
        if self.in_pk:
            self._tick_pk()
            return
        self._sim_step(fast=False)

    def tick_fast(self):
        """헤드리스 tick — 시각용 jitter/smoothing 만 생략, 결정 로직 동일."""
        if self.finished:
            return
        if self.in_pk:
            self._tick_pk()
            return
        self._sim_step(fast=True)

    def _sim_step(self, fast: bool):
        """한 tick — tick/tick_fast 공통 본체.
        fast=True는 호환용 시그너처일 뿐, 모든 시뮬레이션 로직은 동일."""
        self.tick_count += 1
        new_minute = (self.tick_count % SIM_TICKS_PER_MIN == 0)
        if new_minute:
            self.minute += 1
            # 후반 시작 — 진영 교체 (1회만, 연장전엔 다시 안 함)
            if self.minute == 45 and not self.half_swap and not self.in_extra_time:
                self._swap_halves()
            self._check_substitution()
            self._check_tactical_change()

        # 주기적 포지션 스왑 — fast/visual 양쪽 동일 RNG 경로 위해 항상 실행
        if (self.tick_count % POSITION_SWAP_PERIOD == 0
                and self.visual_rng.random() < POSITION_SWAP_PROB):
            self._maybe_swap_positions()

        # 점유 전환 — fast/visual 모두 동일 RNG 경로
        if self.rng.random() < 0.020:
            new_poss = 0 if self.rng.random() < self.home_possession else 1
            if new_poss != self.possession_idx:
                self.possession_idx = new_poss
                if self.phase == 'normal':
                    self.carrier_idx = None
                    self.carrier_ttl = 0
                    self.pass_in_flight = False
                    self.pass_target_idx = None

        # Phase 갱신 (shot_in_flight 완료, kickoff 종료 등)
        self._update_phase()

        # 시각 점유 — buildup이 없으므로 phase 기반 단순화
        if self.phase == 'kickoff' and self.last_goal_team >= 0:
            self.visual_possession_idx = 1 - self.last_goal_team
        else:
            self.visual_possession_idx = self.possession_idx

        # Phase별 본체
        if self.phase == 'shot_in_flight':
            self._step_shot_flight(fast)
        elif self.phase == 'setpiece_shot':
            self._step_setpiece_shot(fast)
        elif self.phase == 'kickoff':
            self._step_kickoff(fast)
        else:
            self._step_normal(fast)

        if self.goal_flash > 0:
            self.goal_flash -= 1

        # 점유 카운트 — visual_possession_idx로 통일 (양 모드 동일)
        self.possession_ticks[self.visual_possession_idx] += 1

        # 정규/연장 종료
        if self.tick_count % SIM_TICKS_PER_MIN == 0:
            if self.in_extra_time and self.minute >= self.extra_time_end_minute:
                self._finish_extra_time()
            elif (not self.in_extra_time
                    and self.minute >= self.regulation_end_minute):
                self._finish_regulation()

    # ── Phase 머신 ──────────────────────────────────────────

    def _update_phase(self):
        """kickoff 페이즈 종료 체크 (shot_in_flight/setpiece_shot 종료는
        _step_shot_flight/_step_setpiece_shot 내부에서 처리)."""
        if self.phase == 'kickoff':
            elapsed = self.tick_count - self.phase_start_tick
            if elapsed >= self._kickoff_duration_actual():
                self.phase = 'normal'
                self.carrier_idx = None
                self.carrier_ttl = 0
                self.pass_in_flight = False
                self.pass_target_idx = None

    # ── Visual: NORMAL ──────────────────────────────────────

    def _player_pulls(self, p: 'Player') -> tuple:
        """선수의 (home_pull_x, home_pull_y) — wing일수록 y 더 강하게 home 고정."""
        base = HOME_PULL.get(p.role, 0.4)
        y_dist = abs(p.home_y - self.PITCH_H * 0.5) / (self.PITCH_H * 0.5)
        py = min(HOME_PULL_Y_MAX, base + HOME_PULL_Y_BOOST * y_dist)
        return base, py

    def _offside_line(self, attacking_team: int) -> float:
        """attacking_team의 공격 방향 기준, 상대 2번째 디펜더 x = offside line."""
        opps = [p.x for p in self.players
                 if p.team_idx != attacking_team and p.on_pitch]
        attacks_pos = self._attacks_positive(attacking_team)
        if len(opps) < 2:
            return self.PITCH_W * (0.82 if attacks_pos else 0.18)
        if attacks_pos:
            opps.sort(reverse=True)   # GK가 [0], 마지막 DEF가 [1]
        else:
            opps.sort()
        return opps[1]

    def _apply_offside_cap(self, p: 'Player', tx: float,
                            home_off: float, away_off: float) -> float:
        """FWD의 target x를 오프사이드 라인 안쪽으로 클램프."""
        if p.role != 'FWD':
            return tx
        line = home_off if p.team_idx == 0 else away_off
        if self._attacks_positive(p.team_idx):
            return min(tx, line - 0.5)
        return max(tx, line + 0.5)

    def _enforce_offside_position(self, p: 'Player',
                                    home_off: float, away_off: float):
        """FWD 실제 위치도 오프사이드 안쪽으로 hard cap."""
        if p.role != 'FWD':
            return
        line = home_off if p.team_idx == 0 else away_off
        if self._attacks_positive(p.team_idx):
            p.x = min(p.x, line)
        else:
            p.x = max(p.x, line)

    def _step_normal(self, fast: bool):
        """일반 플레이 — carrier 관리, 패스, 슛 트리거, 파울 체크."""
        atk_pos = self._attacks_positive(self.visual_possession_idx)
        target_x = (self.PITCH_W * 0.95 if atk_pos
                    else self.PITCH_W * 0.05)

        if self.pass_in_flight:
            self._advance_pass()
            self.ball_x = max(1, min(self.PITCH_W - 1, self.ball_x))
            self.ball_y = max(1, min(self.PITCH_H - 1, self.ball_y))
            carrier = self._current_carrier()
            self.carrier_ttl -= 1
        else:
            carrier = self._current_carrier()
            if carrier is None:
                carrier = self._pick_new_carrier()

            if carrier is not None:
                dx = carrier.x - self.ball_x
                dy = carrier.y - self.ball_y
                dist = math.hypot(dx, dy)
                near_factor = min(1.0, dist / 8.0)
                follow = BALL_FOLLOW_K * (0.35 + 0.65 * near_factor)
                self.ball_vx = self.ball_vx * BALL_FRICTION + dx * follow
                self.ball_vy = self.ball_vy * BALL_FRICTION + dy * follow
            else:
                self.ball_vx *= BALL_FRICTION
                self.ball_vy *= BALL_FRICTION
            self.ball_x += self.ball_vx * BALL_STEP_RATIO
            self.ball_y += self.ball_vy * BALL_STEP_RATIO
            self.ball_x = max(1, min(self.PITCH_W - 1, self.ball_x))
            self.ball_y = max(1, min(self.PITCH_H - 1, self.ball_y))

            self.carrier_ttl -= 1
            if (carrier is not None
                    and self.visual_rng.random() < PASS_PROB_PER_TICK):
                self._try_pass()

        # 2-phase 선수 갱신: 비FWD → offside 라인 → FWD
        tc = self.tick_count

        if self.pass_in_flight and self.carrier_idx is not None:
            ref_ball_x = self.players[self.carrier_idx].x
            ref_ball_y = self.players[self.carrier_idx].y
        else:
            ref_ball_x = self.ball_x
            ref_ball_y = self.ball_y

        # breath/drift는 양쪽 모두 적용 — 위치가 게임 결정에 영향 주므로 일치 필요
        breath = BREATH_AMPLITUDE
        breath_y = BREATH_AMPLITUDE * 0.7
        drift_amp_x = TACTICAL_DRIFT_AMP_X
        drift_amp_y = TACTICAL_DRIFT_AMP_Y

        def _move_player(i: int, p: Player, off_cap: bool,
                          home_off: float, away_off: float):
            if self.pass_in_flight and i == self.pass_target_idx:
                tx = self.pass_dest_x
                ty = self.pass_dest_y
                sprint_k = max(PLAYER_LERP * 2.4, 0.11)
                p.x += (tx - p.x) * sprint_k
                p.y += (ty - p.y) * sprint_k
                p.x = max(0.5, min(self.PITCH_W - 0.5, p.x))
                p.y = max(0.5, min(self.PITCH_H - 0.5, p.y))
                if off_cap:
                    self._enforce_offside_position(p, home_off, away_off)
                return

            pull_x, pull_y = self._player_pulls(p)
            drift_x = math.sin(tc / TACTICAL_DRIFT_PERIOD_X
                                + p.drift_phase_x) * drift_amp_x
            d_y = math.sin(tc / TACTICAL_DRIFT_PERIOD_Y
                            + p.drift_phase_y) * drift_amp_y
            eff_home_x = p.home_x + drift_x
            eff_home_y = p.home_y + d_y
            tx = eff_home_x * pull_x + ref_ball_x * (1.0 - pull_x)
            ty = eff_home_y * pull_y + ref_ball_y * (1.0 - pull_y)
            if p.is_fullback and p.team_idx == self.visual_possession_idx:
                if atk_pos:
                    tx = tx + (self.PITCH_W * 0.62 - tx) * OVERLAP_PUSH_RATIO
                else:
                    tx = tx + (self.PITCH_W * 0.38 - tx) * OVERLAP_PUSH_RATIO
            if p.team_idx == self.visual_possession_idx and p.role == 'FWD':
                tx = tx * 0.5 + target_x * 0.5
            if off_cap and i != self.carrier_idx:
                tx = self._apply_offside_cap(p, tx, home_off, away_off)
            bx = math.sin((tc + i * 7.3) * BREATH_FREQ) * breath
            by = math.cos((tc + i * 5.1) * BREATH_FREQ * 0.93) * breath_y
            p.x += (tx - p.x) * PLAYER_LERP + bx
            p.y += (ty - p.y) * PLAYER_LERP + by
            p.x = max(0.5, min(self.PITCH_W - 0.5, p.x))
            p.y = max(0.5, min(self.PITCH_H - 0.5, p.y))
            if off_cap and i != self.carrier_idx:
                self._enforce_offside_position(p, home_off, away_off)

        for i, p in enumerate(self.players):
            if not p.on_pitch or p.role == 'FWD':
                continue
            _move_player(i, p, False, 0, 0)

        home_off = self._offside_line(0)
        away_off = self._offside_line(1)

        for i, p in enumerate(self.players):
            if not p.on_pitch or p.role != 'FWD':
                continue
            _move_player(i, p, True, home_off, away_off)

        # 슛 + 파울 트리거 — carrier가 있고 패스 중이 아닐 때만
        # 박스 안일 땐 파울 먼저 (PK 현실적 비율 위해), 그 외엔 슛 먼저
        if carrier is not None and not self.pass_in_flight:
            in_box = self._is_in_penalty_box(carrier.x, carrier.y,
                                              carrier.team_idx)
            if in_box:
                self._check_foul_opportunity(carrier)
                if self.phase != 'normal':
                    return
            if self._try_shoot(carrier):
                return
            if not in_box:
                self._check_foul_opportunity(carrier)

    # ── Visual: KICKOFF (골 직후) ───────────────────────────

    def _kickoff_duration_actual(self) -> int:
        """방금 골이 코너/FK였으면 해산 시간 더 길게."""
        if self.last_goal_type in ('corner', 'free_kick'):
            return GOAL_HOLD_TICKS + SETPIECE_DISPERSE_TICKS
        return KICKOFF_DURATION

    def _kickoff_target_pos(self, p: Player) -> tuple:
        # 공격 방향과 반대 (자기 진영)으로 후퇴
        attacks_pos = self._attacks_positive(p.team_idx)
        if attacks_pos:
            # 자기 진영 = -x 쪽
            if p.role == 'FWD':
                tx = min(p.home_x, self.PITCH_W * 0.42)
            elif p.role == 'MID':
                tx = min(p.home_x, self.PITCH_W * 0.45)
            else:
                tx = min(p.home_x, self.PITCH_W * 0.48)
        else:
            # 자기 진영 = +x 쪽
            if p.role == 'FWD':
                tx = max(p.home_x, self.PITCH_W * 0.58)
            elif p.role == 'MID':
                tx = max(p.home_x, self.PITCH_W * 0.55)
            else:
                tx = max(p.home_x, self.PITCH_W * 0.52)
        return tx, p.home_y

    def _step_kickoff(self, fast: bool):
        elapsed = self.tick_count - self.phase_start_tick
        cx = self.PITCH_W / 2
        cy = self.PITCH_H / 2

        if elapsed < GOAL_HOLD_TICKS:
            # 공은 득점팀이 공격하던 골대 안에 머무름 (전·후반 반영)
            if self.last_goal_team >= 0:
                if self._attacks_positive(self.last_goal_team):
                    hold_x = self.PITCH_W * 0.985
                else:
                    hold_x = self.PITCH_W * 0.015
            else:
                hold_x = self.ball_x
            jitter_phase = elapsed * 0.7
            jitter = max(0.0, 1.0 - elapsed / 20.0)
            self.ball_x += (hold_x - self.ball_x) * 0.18
            self.ball_y += (cy + math.sin(jitter_phase) * 0.6 * jitter
                            - self.ball_y) * 0.12
            self.ball_vx = 0
            self.ball_vy = 0
            for p in self.players:
                if not p.on_pitch:
                    continue
                tx, ty = self._kickoff_target_pos(p)
                lerp_k = 0.04 if p.team_idx == self.last_goal_team else 0.07
                p.x += (tx - p.x) * lerp_k
                p.y += (ty - p.y) * lerp_k
        else:
            reset_elapsed = elapsed - GOAL_HOLD_TICKS
            is_sp = self.last_goal_type in ('corner', 'free_kick')
            reset_total = SETPIECE_DISPERSE_TICKS if is_sp else KICKOFF_RESET_TICKS
            player_lerp = 0.045 if is_sp else 0.12
            progress = min(1.0, reset_elapsed / reset_total)
            eased = _ease_in_out_sine(progress)
            self.ball_vx *= 0.82
            self.ball_vy *= 0.82
            pull = 0.10 + eased * 0.12
            self.ball_x += (cx - self.ball_x) * pull
            self.ball_y += (cy - self.ball_y) * pull
            for p in self.players:
                if not p.on_pitch:
                    continue
                tx, ty = self._kickoff_target_pos(p)
                p.x += (tx - p.x) * player_lerp
                p.y += (ty - p.y) * player_lerp

        self.carrier_idx = None
        self.carrier_ttl = 0

    # ── SHOT: 상태 기반 슛 결정/비행 ─────────────────────────

    def _attacks_positive(self, team_idx: int) -> bool:
        """True면 해당 팀이 +x 방향(오른쪽) 골대를 공격 중.
        half_swap=False (전반)에서 team 0이 +x. 후반엔 반대."""
        return (team_idx == 0) != self.half_swap

    def _opp_goal_x(self, team_idx: int) -> float:
        """team_idx 가 공격하는 골대 x. 전·후반에 따라 자동 반전."""
        return self.PITCH_W if self._attacks_positive(team_idx) else 0.0

    def _own_goal_x(self, team_idx: int) -> float:
        """team_idx 의 자기 골대 x (수비하는 골)."""
        return 0.0 if self._attacks_positive(team_idx) else self.PITCH_W

    def _swap_halves(self):
        """전반→후반 진영 교체 — 모든 위치/공/패스/세트피스를 +x 미러링."""
        W = self.PITCH_W
        for p in self.players:
            p.x = W - p.x
            p.home_x = W - p.home_x
        self.ball_x = W - self.ball_x
        self.ball_vx = -self.ball_vx
        if self.pass_in_flight:
            self.pass_dest_x = W - self.pass_dest_x
            self.pass_start_x = W - self.pass_start_x
        if self.shot_state is not None:
            self.shot_state['origin_x'] = W - self.shot_state['origin_x']
        if self.setpiece_state is not None:
            self.setpiece_state['sp_x'] = W - self.setpiece_state['sp_x']
        self.half_swap = True
        # 후반 시작 알림 + kickoff 페이즈로 잠시 정렬 (셀레브 없이 짧게)
        self.events.append(MatchEvent(self.minute, 'half', -1, 'HALF TIME'))
        self.phase = 'kickoff'
        self.phase_start_tick = self.tick_count
        self.last_goal_team = -1   # 중립 — 셀레브레이션 없이 빠른 리셋
        self.last_goal_type = 'open_play'
        self.carrier_idx = None
        self.carrier_ttl = 0
        self.pass_in_flight = False
        self.pass_target_idx = None

    def _find_opp_keeper(self, team_idx: int) -> Optional[Player]:
        opp = 1 - team_idx
        for p in self.players:
            if p.team_idx == opp and p.role == 'GK' and p.on_pitch:
                return p
        return None

    def _count_blockers_in_lane(self, carrier: Player) -> int:
        """carrier와 골대 사이 슛 레인에 있는 상대 (GK 제외) 카운트."""
        team = carrier.team_idx
        attacks_pos = self._attacks_positive(team)
        goal_x = self._opp_goal_x(team)
        cnt = 0
        for p in self.players:
            if (not p.on_pitch or p.team_idx == team
                    or p.role == 'GK'):
                continue
            # carrier보다 골대에 더 가까운 (앞에 있는) 선수만
            if attacks_pos and p.x <= carrier.x:
                continue
            if not attacks_pos and p.x >= carrier.x:
                continue
            # x 레인 안 (carrier ~ 골대 사이) — 골대 너머는 제외
            if attacks_pos and p.x > goal_x:
                continue
            if not attacks_pos and p.x < goal_x:
                continue
            if abs(p.y - carrier.y) < BLOCKER_LANE_HALF_Y:
                cnt += 1
        return cnt

    def _shot_quality(self, carrier: Player) -> float:
        """0~1 슛 기회 품질. 거리/각도/blockers/GK 거리 종합."""
        team = carrier.team_idx
        attacks_pos = self._attacks_positive(team)
        # 어택킹 써드 진입 여부
        if attacks_pos and carrier.x < self.PITCH_W * SHOT_ZONE_X_RATIO:
            return 0.0
        if not attacks_pos and carrier.x > self.PITCH_W * (1.0 - SHOT_ZONE_X_RATIO):
            return 0.0
        goal_x = self._opp_goal_x(team)
        cy = self.PITCH_H / 2
        dx = abs(goal_x - carrier.x)
        dy = abs(carrier.y - cy)
        dist = math.hypot(dx, dy)
        max_dist = self.PITCH_W * 0.45
        # 가까울수록 quality ↑ (0~1)
        dist_q = max(0.0, 1.0 - dist / max_dist)
        # 각도 — 정중앙일수록 ↑
        angle_q = max(0.0, 1.0 - dy / (self.PITCH_H * 0.5))
        # Blockers — 많을수록 quality ↓
        blockers = self._count_blockers_in_lane(carrier)
        blocker_q = max(0.0, 1.0 - blockers * 0.35)
        # GK 거리 — 1대1 보너스
        gk = self._find_opp_keeper(team)
        gk_q = 0.0
        if gk is not None:
            gk_dist = math.hypot(gk.x - carrier.x, gk.y - carrier.y)
            if blockers == 0 and gk_dist < ONE_V_ONE_GK_DIST:
                gk_q = 0.5  # 1대1 큰 보너스
        # 종합
        q = dist_q * 0.55 + angle_q * 0.20 + blocker_q * 0.25 + gk_q
        return max(0.0, min(1.0, q))

    def _try_shoot(self, carrier: Player) -> bool:
        """carrier가 슛 가치 있을 때 per-tick 확률로 슛 시도. 시도 시 True."""
        if carrier.role not in ('MID', 'FWD'):
            return False
        if self.phase != 'normal':
            return False
        quality = self._shot_quality(carrier)
        if quality <= 0.0:
            return False
        # 슈터 능력 보정 — rating 75 = 1.0, 90 = 1.3, 60 = 0.7
        skill_mult = max(0.55, carrier.rating / 75.0)
        atk = self.home if carrier.team_idx == 0 else self.away
        atk_factor = max(0.7, atk.attack / 75.0)
        # 1분당 ~SIM_TICKS_PER_MIN 회 평가되므로 per-tick 확률은 작아야 함
        p_shoot = SHOT_PROB_PER_TICK_BASE * quality * skill_mult * atk_factor
        if self.rng.random() > p_shoot:
            return False
        self._start_shot_flight(carrier, quality)
        return True

    def _start_shot_flight(self, carrier: Player, quality: float):
        """슛 결과 미리 결정 + 비행 phase 진입."""
        team = carrier.team_idx
        atk = self.home if team == 0 else self.away
        df = self.away if team == 0 else self.home
        df_late_mult = (self._late_match_def_mult(self.away)
                         if team == 0 else self._late_match_def_mult(self.home))
        # 골 확률: 슈터 능력 + 슛 quality vs (수비 + GK)
        df_def = df.defense * df_late_mult
        shooter_skill = carrier.rating
        # quality 높을수록 골 확률 ↑
        edge = ((shooter_skill - 0.5 * df_def - 0.5 * df.keeper) / 28.0
                + (quality - 0.4) * 1.1)
        goal_p = 1.0 / (1.0 + math.exp(-edge * 1.4))
        # quality가 매우 낮으면 (먼 슛) 골 확률도 캡
        goal_p *= 0.20 + 0.65 * quality  # 0.2 ~ 0.85 스케일
        goal_p *= GOAL_P_SCALE
        # 1대1 추가 보너스
        gk = self._find_opp_keeper(team)
        if gk is not None and self._count_blockers_in_lane(carrier) == 0:
            gk_dist = math.hypot(gk.x - carrier.x, gk.y - carrier.y)
            if gk_dist < ONE_V_ONE_GK_DIST:
                goal_p = min(0.92, goal_p * 1.65)
        # wc_titles 보너스
        bonus = self.h_goal_bonus if team == 0 else self.a_goal_bonus
        goal_p *= bonus
        goal_p = max(0.01, min(0.95, goal_p))

        # 통계 — 슛 + xG
        self.team_shots[team] += 1
        self.team_xG[team] += goal_p

        will_be_goal = self.rng.random() < goal_p
        # 막혔을 때 유효슛 비율 (visual_rng — 게임 결정 RNG 보존)
        if not will_be_goal:
            on_target_p = ON_TARGET_RATE_BASE + (atk.attack - 70) * 0.005
            if self.visual_rng.random() < on_target_p:
                self.team_on_target[team] += 1
        else:
            self.team_on_target[team] += 1

        # 막힌 슛 중 일부는 코너로 (blockers > 0 일 때만)
        blocked_for_corner = False
        if not will_be_goal and self._count_blockers_in_lane(carrier) > 0:
            if self.rng.random() < CORNER_ON_BLOCK_PROB:
                blocked_for_corner = True

        # 비행 상태 셋업
        cy = self.PITCH_H / 2
        target_y = cy + (carrier.y - cy) * 0.30  # 살짝 구석으로
        scorer_name = carrier.name if will_be_goal else ''
        assist_name = ''
        if will_be_goal:
            scorer_name = self._pick_scorer_for_open_play(team, carrier)
            assist_name = self._pick_assister(team, scorer_name)

        self.shot_state = {
            'origin_x': carrier.x,
            'origin_y': carrier.y,
            'target_y': target_y,
            'will_be_goal': will_be_goal,
            'blocked_for_corner': blocked_for_corner,
            'team_idx': team,
            'scorer_name': scorer_name,
            'assist_name': assist_name,
            'is_setpiece': False,
            'goal_type': 'open_play',
        }
        self.phase = 'shot_in_flight'
        self.phase_start_tick = self.tick_count
        self.pass_in_flight = False
        self.pass_target_idx = None

    def _pick_scorer_for_open_play(self, team_idx: int, shooter: Player) -> str:
        """오픈플레이 스코러 — carrier 본인이 70% 확률, 나머지 30%는 팀 가중치."""
        if shooter.name and self.visual_rng.random() < 0.70:
            return shooter.name
        return self._pick_scorer(team_idx, 'open_play')

    def _step_shot_flight(self, fast: bool):
        """슛 비행 — SHOT_FLIGHT_TICKS 동안 공이 골대로. 종료 시 결과 발동."""
        if self.shot_state is None:
            self.phase = 'normal'
            return
        elapsed = self.tick_count - self.phase_start_tick
        s = self.shot_state
        team = s['team_idx']
        goal_x = (self.PITCH_W * 1.01 if self._attacks_positive(team)
                   else self.PITCH_W * -0.01)
        progress = min(1.0, elapsed / SHOT_FLIGHT_TICKS)
        eased = 1 - (1 - progress) ** 2
        self.ball_x = s['origin_x'] + (goal_x - s['origin_x']) * eased
        self.ball_y = s['origin_y'] + (s['target_y'] - s['origin_y']) * eased
        self.ball_vx = 0
        self.ball_vy = 0

        # 다른 선수들은 평소대로 움직임 (단순한 lerp)
        self._step_players_basic(fast)

        if elapsed >= SHOT_FLIGHT_TICKS:
            self._resolve_shot()

    def _resolve_shot(self):
        """비행 종료 — 골 / 세이브 / 코너 처리."""
        if self.shot_state is None:
            self.phase = 'normal'
            return
        s = self.shot_state
        team = s['team_idx']
        team_obj = self.home if team == 0 else self.away

        if s['will_be_goal']:
            self._score(team, goal_type=s.get('goal_type', 'open_play'),
                         scorer_name=s.get('scorer_name', ''),
                         assist_name=s.get('assist_name', ''))
            # _score 가 kickoff phase 로 전환
        else:
            self.events.append(MatchEvent(
                self.minute, 'shot', team,
                f'{team_obj.code} shot, saved'))
            if s['blocked_for_corner']:
                # 막힌 슛 → 코너 세트피스로 전환
                self._start_setpiece(team, 'corner')
            else:
                # 다시 normal로
                self.phase = 'normal'
                self.shot_state = None
                self.carrier_idx = None
                self.carrier_ttl = 0
                self.possession_idx = 1 - team

    def _step_players_basic(self, fast: bool):
        """슛/세트피스 비행 중 단순한 선수 위치 갱신 (자신의 home으로 lerp)."""
        tc = self.tick_count
        breath = BREATH_AMPLITUDE * 0.4
        for i, p in enumerate(self.players):
            if not p.on_pitch:
                continue
            tx = p.home_x * 0.5 + self.ball_x * 0.5
            ty = p.home_y * 0.5 + self.ball_y * 0.5
            bx = math.sin((tc + i * 7.3) * BREATH_FREQ) * breath
            by = math.cos((tc + i * 5.1) * BREATH_FREQ * 0.93) * breath * 0.75
            p.x += (tx - p.x) * PLAYER_LERP + bx
            p.y += (ty - p.y) * PLAYER_LERP + by
            p.x = max(0.5, min(self.PITCH_W - 0.5, p.x))
            p.y = max(0.5, min(self.PITCH_H - 0.5, p.y))

    # ── FOUL → set piece (corner / FK / PK) ─────────────────

    def _is_in_penalty_box(self, x: float, y: float, attacking_team: int) -> bool:
        """좌표가 attacking_team의 공격 박스 안인지."""
        in_y = (PEN_BOX_Y_RANGE[0] * self.PITCH_H <= y
                <= PEN_BOX_Y_RANGE[1] * self.PITCH_H)
        if not in_y:
            return False
        if self._attacks_positive(attacking_team):
            return x >= self.PITCH_W * PEN_BOX_X_RATIO
        return x <= self.PITCH_W * (1.0 - PEN_BOX_X_RATIO)

    def _nearest_opp_pressure(self, carrier: Player) -> float:
        """carrier에게 가장 가까운 상대 선수까지 거리. 압박 지표."""
        team = carrier.team_idx
        best = 1e9
        for p in self.players:
            if not p.on_pitch or p.team_idx == team or p.role == 'GK':
                continue
            d = math.hypot(p.x - carrier.x, p.y - carrier.y)
            if d < best:
                best = d
        return best

    def _check_foul_opportunity(self, carrier: Player):
        """수비수가 압박 반경 안이고 carrier가 위협 위치면 파울 가능성."""
        if self.phase != 'normal':
            return
        team = carrier.team_idx
        attacks_pos = self._attacks_positive(team)
        if attacks_pos and carrier.x < self.PITCH_W * 0.45:
            return
        if not attacks_pos and carrier.x > self.PITCH_W * 0.55:
            return
        dist = self._nearest_opp_pressure(carrier)
        if dist > PRESSURE_RADIUS:
            return
        proximity = max(0.0, 1.0 - dist / PRESSURE_RADIUS)
        in_box = self._is_in_penalty_box(carrier.x, carrier.y, team)
        # 박스 안은 파울 확률 3배 (PK 현실적 비율 위해)
        box_mult = 3.0 if in_box else 1.0
        p_foul = FOUL_PROB_PER_TICK_BASE * (0.5 + proximity) * box_mult
        if self.rng.random() > p_foul:
            return
        if in_box:
            self._start_setpiece(team, 'penalty')
        else:
            self._start_setpiece(team, 'free_kick')

    def _start_setpiece(self, team_idx: int, kind: str):
        """세트피스 진입 — 셋업 + 슛 비행을 합친 phase."""
        # 세트피스 위치 — 공격 방향에 따라
        attacks_pos = self._attacks_positive(team_idx)
        if kind == 'corner':
            sp_x = self.PITCH_W * 0.99 if attacks_pos else self.PITCH_W * 0.01
            sp_y = (self.PITCH_H * 0.04
                    if self.visual_rng.random() < 0.5
                    else self.PITCH_H * 0.96)
        elif kind == 'free_kick':
            offset = self.visual_rng.uniform(0.18, 0.32)
            sp_x = (self.PITCH_W * (1.0 - offset) if attacks_pos
                    else self.PITCH_W * offset)
            sp_y = self.PITCH_H * self.visual_rng.uniform(0.30, 0.70)
        elif kind == 'penalty':
            sp_x = self.PITCH_W * 0.85 if attacks_pos else self.PITCH_W * 0.15
            sp_y = self.PITCH_H * 0.50
        else:
            return

        # 골 확률 평가 — kind에 따라
        atk = self.home if team_idx == 0 else self.away
        df = self.away if team_idx == 0 else self.home
        df_late_mult = (self._late_match_def_mult(self.away)
                         if team_idx == 0
                         else self._late_match_def_mult(self.home))
        if kind == 'penalty':
            edge = (atk.attack - 0.6 * df.keeper) / 22.0
            goal_p = 0.55 + 0.40 / (1.0 + math.exp(-edge))   # ~0.55~0.95
        elif kind == 'corner':
            edge = (atk.attack - 0.5 * df.defense * df_late_mult - 0.4 * df.keeper) / 30.0
            goal_p = 1.0 / (1.0 + math.exp(-edge * 1.4)) * 0.18
        else:  # free_kick
            edge = (atk.attack - 0.5 * df.defense * df_late_mult - 0.5 * df.keeper) / 30.0
            goal_p = 1.0 / (1.0 + math.exp(-edge * 1.4)) * 0.13
        bonus = self.h_goal_bonus if team_idx == 0 else self.a_goal_bonus
        goal_p = min(0.95, goal_p * bonus)

        will_be_goal = self.rng.random() < goal_p
        self.team_shots[team_idx] += 1
        self.team_xG[team_idx] += goal_p
        if will_be_goal:
            self.team_on_target[team_idx] += 1
        else:
            on_target_p = ON_TARGET_RATE_BASE + (atk.attack - 70) * 0.005
            if self.visual_rng.random() < on_target_p:
                self.team_on_target[team_idx] += 1

        scorer_name = self._pick_scorer(team_idx, kind) if will_be_goal else ''
        assist_name = self._pick_assister(team_idx, scorer_name) if will_be_goal else ''

        self.setpiece_state = {
            'kind': kind,
            'sp_x': sp_x,
            'sp_y': sp_y,
            'team_idx': team_idx,
            'will_be_goal': will_be_goal,
            'goal_type': kind,
            'scorer_name': scorer_name,
            'assist_name': assist_name,
        }
        self.phase = 'setpiece_shot'
        self.phase_start_tick = self.tick_count
        self.carrier_idx = None
        self.carrier_ttl = 0
        self.pass_in_flight = False
        self.pass_target_idx = None
        # 이벤트 로그
        self.events.append(MatchEvent(
            self.minute,
            'pk' if kind == 'penalty' else 'setpiece',
            team_idx,
            f'{atk.code} {kind.replace("_", " ").upper()}'))

    def _step_setpiece_shot(self, fast: bool):
        """세트피스 — 셋업 ticks 동안 선수들 박스로 모이고, 끝부분 슛 비행."""
        if self.setpiece_state is None:
            self.phase = 'normal'
            return
        sp = self.setpiece_state
        kind = sp['kind']
        team = sp['team_idx']
        sp_x = sp['sp_x']
        sp_y = sp['sp_y']
        setup_ticks = (PENALTY_SETUP_TICKS if kind == 'penalty'
                        else SETPIECE_SETUP_TICKS)
        total_ticks = setup_ticks + SHOT_FLIGHT_TICKS
        elapsed = self.tick_count - self.phase_start_tick

        # 공 위치
        if elapsed < setup_ticks:
            self.ball_x += (sp_x - self.ball_x) * 0.14
            self.ball_y += (sp_y - self.ball_y) * 0.14
            self.ball_vx *= 0.82
            self.ball_vy *= 0.82
        else:
            shot_elapsed = elapsed - setup_ticks
            progress = min(1.0, shot_elapsed / SHOT_FLIGHT_TICKS)
            eased = 1 - (1 - progress) ** 2
            attacks_pos = self._attacks_positive(team)
            goal_x = (self.PITCH_W * 1.01 if attacks_pos
                       else self.PITCH_W * -0.01)
            goal_y = self.PITCH_H * 0.50
            self.ball_x = sp_x + (goal_x - sp_x) * eased
            self.ball_y = sp_y + (goal_y - sp_y) * eased
            if kind == 'free_kick':
                curve_dir = 1 if attacks_pos else -1
                self.ball_x += math.sin(progress * math.pi) * curve_dir * 0.8
            self.ball_vx = 0
            self.ball_vy = 0

        # 선수 배치 — kind 별
        if kind == 'penalty':
            self._move_setpiece_penalty(team, fast)
        else:
            self._move_setpiece_box(team, sp_x, sp_y, fast)

        if elapsed >= total_ticks:
            self._resolve_setpiece()

    def _move_setpiece_penalty(self, team: int, fast: bool):
        attacks_pos = self._attacks_positive(team)
        edge_x = self.PITCH_W * 0.70 if attacks_pos else self.PITCH_W * 0.30
        for i, p in enumerate(self.players):
            if not p.on_pitch:
                continue
            if p.role == 'GK':
                # 자기 골 라인 (자기 공격 방향과 반대)
                tx = (self.PITCH_W * 0.04
                      if self._attacks_positive(p.team_idx)
                      else self.PITCH_W * 0.96)
                ty = self.PITCH_H * 0.50
            else:
                tx = edge_x + (i % 5 - 2) * 1.2
                ty = p.home_y * 0.5 + self.PITCH_H * 0.50 * 0.5
            p.x += (tx - p.x) * 0.07
            p.y += (ty - p.y) * 0.07
            p.x = max(0.5, min(self.PITCH_W - 0.5, p.x))
            p.y = max(0.5, min(self.PITCH_H - 0.5, p.y))

    def _move_setpiece_box(self, team: int, sp_x: float, sp_y: float, fast: bool):
        attacks_pos = self._attacks_positive(team)
        box_x = self.PITCH_W * 0.85 if attacks_pos else self.PITCH_W * 0.15
        box_y = self.PITCH_H * 0.50
        tc = self.tick_count
        breath = BREATH_AMPLITUDE * 0.4
        for i, p in enumerate(self.players):
            if not p.on_pitch:
                continue
            if p.role == 'GK':
                tx = (self.PITCH_W * 0.04
                      if self._attacks_positive(p.team_idx)
                      else self.PITCH_W * 0.96)
                ty = self.PITCH_H * 0.50
            else:
                tx = box_x * 0.7 + p.home_x * 0.3
                ty = box_y * 0.55 + p.home_y * 0.45
            bx = math.sin((tc + i * 7.3) * BREATH_FREQ) * breath
            by = math.cos((tc + i * 5.1) * BREATH_FREQ * 0.93) * breath * 0.75
            p.x += (tx - p.x) * BUILDUP_PLAYER_LERP + bx
            p.y += (ty - p.y) * BUILDUP_PLAYER_LERP + by
            p.x = max(0.5, min(self.PITCH_W - 0.5, p.x))
            p.y = max(0.5, min(self.PITCH_H - 0.5, p.y))

    def _resolve_setpiece(self):
        if self.setpiece_state is None:
            self.phase = 'normal'
            return
        sp = self.setpiece_state
        team = sp['team_idx']
        team_obj = self.home if team == 0 else self.away
        if sp['will_be_goal']:
            self._score(team, goal_type=sp['goal_type'],
                         scorer_name=sp.get('scorer_name', ''),
                         assist_name=sp.get('assist_name', ''))
        else:
            self.events.append(MatchEvent(
                self.minute, 'shot', team,
                f'{team_obj.code} {sp["kind"]} saved'))
            self.phase = 'normal'
            self.setpiece_state = None
            self.possession_idx = 1 - team
            self.carrier_idx = None
            self.carrier_ttl = 0

    # ── Carrier 헬퍼 (visual_possession_idx 사용) ──────────

    def _current_carrier(self) -> Optional[Player]:
        if self.carrier_idx is None or self.carrier_ttl <= 0:
            return None
        c = self.players[self.carrier_idx]
        if c.team_idx != self.visual_possession_idx:
            return None
        return c

    def _pick_new_carrier(self) -> Optional[Player]:
        candidates = [(i, p) for i, p in enumerate(self.players)
                      if p.team_idx == self.visual_possession_idx
                      and p.on_pitch and p.can_carry]
        if not candidates:
            self.carrier_idx = None
            return None
        # 골대 방향 + 능력치 가중치
        atk_pos = self._attacks_positive(self.visual_possession_idx)
        weights = []
        for _, p in candidates:
            pos_w = (max(0.1, p.x / self.PITCH_W) if atk_pos
                      else max(0.1, 1.0 - p.x / self.PITCH_W))
            rating_w = max(0.6, p.rating / 75.0)
            star_w = 1.6 if p.is_star else 1.0
            weights.append(pos_w * rating_w * star_w)
        idx_in_cands = self._weighted_pick(weights)
        self.carrier_idx, _ = candidates[idx_in_cands]
        self.carrier_ttl = self.visual_rng.randint(*CARRIER_TTL_RANGE)
        return self.players[self.carrier_idx]

    def _try_pass(self):
        """현재 carrier에서 같은 팀 동료에게 lead-pass.
        공은 동료의 전방 공간으로 날아가고, 수신자는 그 공간으로 뛰어든다.
        도착 시점에 carrier 전환."""
        if self.carrier_idx is None or self.pass_in_flight:
            return
        cur = self.players[self.carrier_idx]
        candidates = [(i, p) for i, p in enumerate(self.players)
                      if p.team_idx == self.visual_possession_idx
                      and p.on_pitch and p.can_carry
                      and i != self.carrier_idx]
        if not candidates:
            return
        # 골대 더 가까운 동료 + 너무 멀지 않은 동료 우선
        atk_pos = self._attacks_positive(self.visual_possession_idx)
        def _weight(p):
            # 전방성 (0~1)
            if atk_pos:
                fwd = max(0.1, (p.x - cur.x + self.PITCH_W) / (2 * self.PITCH_W))
            else:
                fwd = max(0.1, (cur.x - p.x + self.PITCH_W) / (2 * self.PITCH_W))
            # 거리 패널티 (너무 먼 long ball 줄임)
            dist = math.hypot(p.x - cur.x, p.y - cur.y)
            dist_factor = max(0.15, 1.0 - dist / (self.PITCH_W * 0.6))
            return fwd * dist_factor
        weights = [_weight(p) for _, p in candidates]
        idx_in_cands = self._weighted_pick(weights)
        target_idx, target = candidates[idx_in_cands]

        # Lead-pass: 동료의 전방 공간(골대 방향)으로 공을 보낸다
        dist = math.hypot(target.x - cur.x, target.y - cur.y)
        lead_amt = min(14.0, max(4.0, dist * 0.18))
        lead_x = lead_amt if atk_pos else -lead_amt
        # FWD 수신자면 y도 살짝 와이드하게 (사이드로 뛰어듦)
        lead_y = 0.0
        if target.role == 'FWD':
            # 골대까지 y 거리 줄이는 방향으로 lead (중앙 정렬)
            cy = self.PITCH_H * 0.5
            lead_y = (cy - target.y) * 0.12

        dest_x = max(2.0, min(self.PITCH_W - 2.0, target.x + lead_x))
        dest_y = max(2.0, min(self.PITCH_H - 2.0, target.y + lead_y))

        # 비행 시간 — 공이 cur 위치에서 dest까지 가는 데 걸리는 tick
        # 잔디 위를 굴러가는 느낌 → 좀 더 길게 + 짧은 패스도 최소 시간 확보
        flight_dist = math.hypot(dest_x - self.ball_x, dest_y - self.ball_y)
        flight_ticks = int(max(12, min(26, flight_dist / 3.8)))

        self.pass_in_flight = True
        self.pass_target_idx = target_idx
        self.pass_start_x = self.ball_x
        self.pass_start_y = self.ball_y
        self.pass_dest_x = dest_x
        self.pass_dest_y = dest_y
        self.pass_total_ticks = flight_ticks
        self.pass_tick = 0
        # 패스 진행 동안 carrier는 송신자 상태 유지하지만 공은 따라가지 않게
        # carrier_ttl을 비행 시간만큼 늘려서 picker가 새로 안 뽑게
        self.carrier_ttl = max(self.carrier_ttl, flight_ticks + 4)
        # ball velocity 초기화 — 비행은 위치 보간으로 표현
        self.ball_vx = 0.0
        self.ball_vy = 0.0

    def _advance_pass(self) -> bool:
        """비행 중인 공을 한 tick 진행. 도착 시 carrier 전환, True 반환.
        탑다운 뷰 — 공은 잔디 위를 굴러가는 느낌. 호(arc) 없음."""
        if not self.pass_in_flight:
            return False
        self.pass_tick += 1
        t = min(1.0, self.pass_tick / max(1, self.pass_total_ticks))
        # ease_out_quad — 잔디 마찰 감속 (cubic보다 더 부드럽고 자연스러움)
        eased = 1 - (1 - t) ** 2
        prev_x, prev_y = self.ball_x, self.ball_y
        self.ball_x = self.pass_start_x + (self.pass_dest_x - self.pass_start_x) * eased
        self.ball_y = self.pass_start_y + (self.pass_dest_y - self.pass_start_y) * eased
        # 잔여 속도 기록 (도착 후 자연스러운 감속용)
        self.ball_vx = (self.ball_x - prev_x) / max(0.01, BALL_STEP_RATIO)
        self.ball_vy = (self.ball_y - prev_y) / max(0.01, BALL_STEP_RATIO)
        if t >= 1.0:
            # 도착 — 수신자가 carrier (속도는 부드럽게 감쇠)
            if (self.pass_target_idx is not None
                    and self.players[self.pass_target_idx].on_pitch):
                self.carrier_idx = self.pass_target_idx
                self.carrier_ttl = self.visual_rng.randint(*CARRIER_TTL_RANGE)
            self.pass_in_flight = False
            self.pass_target_idx = None
            # 도착 후 잔여 속도 감쇠 → 다음 carrier follow가 부드럽게 시작
            self.ball_vx *= 0.25
            self.ball_vy *= 0.25
        return True

    def _weighted_pick(self, weights: list) -> int:
        total = sum(weights)
        if total <= 0:
            return self.visual_rng.randrange(len(weights))
        r = self.visual_rng.random() * total
        acc = 0.0
        for i, w in enumerate(weights):
            acc += w
            if r <= acc:
                return i
        return len(weights) - 1

    # ── 교체 + 스태미너 ──────────────────────────────────

    def _team_freshness(self, team_idx: int) -> float:
        """현재 분 기준 팀 평균 신선도. 1.0=기본, <1=피로, >1=신선.
        starting_stamina < 100 인 팀(이전 매치 ET 승리)은 전체 곱셈."""
        on_pitch = [p for p in self.players
                     if p.team_idx == team_idx and p.on_pitch]
        if not on_pitch:
            return 1.0
        total = 0.0
        for p in on_pitch:
            if p.is_starter:
                if self.minute <= FATIGUE_START_MIN:
                    f = 1.0
                else:
                    f = max(0.85, 1.0 - (self.minute - FATIGUE_START_MIN) * FATIGUE_PER_MIN)
            else:
                on_for = self.minute - p.sub_in_minute
                f = SUB_FRESH_BOOST if 0 <= on_for <= SUB_FRESH_DURATION_MIN else 1.0
            total += f
        avg = total / len(on_pitch)
        # ET carry-over — 90→1.0, 80→0.95, etc.
        base_stamina = (self.home_starting_stamina if team_idx == 0
                         else self.away_starting_stamina)
        return avg * (base_stamina / 100.0)

    def _check_substitution(self):
        """매 게임분 진입 시 호출 — 양 팀에 대해 교체 트리거 체크 (self.rng 사용)."""
        if not (SUB_WINDOW_START_MIN <= self.minute <= SUB_WINDOW_END_MIN):
            return
        if self.minute == self._last_min_subbed:
            return
        for team_idx in (0, 1):
            if self.subs_made[team_idx] >= self.max_subs_for_team:
                continue
            if self.rng.random() > SUB_PROB_PER_GAME_MIN:
                continue
            self._do_substitution(team_idx)
        self._last_min_subbed = self.minute

    def _do_substitution(self, team_idx: int):
        """한 명 교체 — 점수 차에 따라 공/수비 형 결정."""
        # OUT 후보: 출전 중인 선발 (스타 제외, 들어온 지 얼마 안 된 후보 제외)
        starters_on = [(i, p) for i, p in enumerate(self.players)
                        if p.team_idx == team_idx and p.on_pitch
                        and not p.is_star
                        and p.sub_in_minute == 0]   # 들어온 후보는 빠지지 않음
        bench = [(i, p) for i, p in enumerate(self.players)
                  if p.team_idx == team_idx and not p.on_pitch and not p.subbed_off]
        if not starters_on or not bench:
            return

        # 점수 차로 sub 타입
        if team_idx == 0:
            score_diff = self.result.home_goals - self.result.away_goals
        else:
            score_diff = self.result.away_goals - self.result.home_goals
        if score_diff < 0:
            out_role, in_role = 'DEF', 'FWD'  # 공격적
        elif score_diff > 0:
            out_role, in_role = 'FWD', 'DEF'  # 수비적
        else:
            out_role, in_role = None, None

        def filt(lst, role):
            return [(i, p) for i, p in lst
                     if (role is None or p.role == role)]
        out_cands = filt(starters_on, out_role) or starters_on
        in_cands = filt(bench, in_role) or bench
        if not out_cands or not in_cands:
            return

        # 가장 평점 낮은 선발 → 가장 높은 후보
        out_idx, out_p = min(out_cands, key=lambda x: x[1].rating)
        _, in_p = max(in_cands, key=lambda x: x[1].rating)

        # 위치/속성 인계
        in_p.x, in_p.y = out_p.x, out_p.y
        in_p.home_x, in_p.home_y = out_p.home_x, out_p.home_y
        in_p.is_fullback = out_p.is_fullback
        in_p.drift_phase_x = out_p.drift_phase_x
        in_p.drift_phase_y = out_p.drift_phase_y
        in_p.on_pitch = True
        in_p.sub_in_minute = self.minute
        out_p.on_pitch = False
        out_p.subbed_off = True       # 영구 마킹 — 다시 못 들어옴
        out_p.x = -100
        out_p.y = -100

        self.subs_made[team_idx] += 1
        team_code = self.home.code if team_idx == 0 else self.away.code
        self.events.append(MatchEvent(
            self.minute, 'sub', team_idx,
            f'SUB ({team_code}) {out_p.name} → {in_p.name}'
        ))

        # carrier 였던 선수가 빠지면 reset
        if self.carrier_idx == out_idx:
            self.carrier_idx = None
            self.carrier_ttl = 0

    def _maybe_swap_positions(self):
        """전술 변경 시뮬 — 한 팀 MID/FWD 두 명의 home 위치 맞바꿈.
        ON-PITCH 선수만 대상 (벤치 home은 -100,-100 이라 섞이면 안 됨)."""
        team = self.visual_rng.randint(0, 1)
        cands = [p for p in self.players
                 if p.team_idx == team and p.on_pitch
                 and p.role in ('MID', 'FWD')]
        if len(cands) >= 2:
            a, b = self.visual_rng.sample(cands, 2)
            a.home_x, b.home_x = b.home_x, a.home_x
            a.home_y, b.home_y = b.home_y, a.home_y

    # ── 매치 통계 헬퍼 (UI/저장용) ─────────────────────────

    def possession_pct(self) -> tuple:
        """(home%, away%) 점유율 정수 백분율."""
        total = self.possession_ticks[0] + self.possession_ticks[1]
        if total == 0:
            return (50, 50)
        h = round(100 * self.possession_ticks[0] / total)
        return (h, 100 - h)

    def shot_counts(self) -> tuple:
        """((home_total, home_on_target), (away_total, away_on_target))"""
        return (
            (self.team_shots[0], self.team_on_target[0]),
            (self.team_shots[1], self.team_on_target[1]),
        )

    def xg(self) -> tuple:
        """(home_xG, away_xG) — 누적 기대 득점."""
        return (round(self.team_xG[0], 2), round(self.team_xG[1], 2))

    def setpiece_counts(self) -> tuple:
        """발생한 골 중 세트피스 카운트.
           ((h_corner, h_fk, h_pen), (a_corner, a_fk, a_pen))"""
        h = {'corner': 0, 'free_kick': 0, 'penalty': 0}
        a = {'corner': 0, 'free_kick': 0, 'penalty': 0}
        for g in self.goals_log:
            gtype = g.get('type', 'open_play')
            if gtype not in h:
                continue
            if g['team_idx'] == 0:
                h[gtype] += 1
            else:
                a[gtype] += 1
        return ((h['corner'], h['free_kick'], h['penalty']),
                (a['corner'], a['free_kick'], a['penalty']))

    def _late_match_def_mult(self, team: 'Team') -> float:
        """후반에 적응 안 된 팀에게 적용되는 수비력 배율."""
        if self.minute < HOT_LATE_MIN:
            return 1.0
        mult = 1.0
        if self.altitude == 'high' and not team.altitude_native:
            mult *= ALT_LATE_PENALTY
        if self.is_hot and not team.heat_native:
            mult *= HOT_LATE_PENALTY
        return mult

    def _score(self, team_idx: int, goal_type: str = 'open_play',
                scorer_name: str = '', assist_name: str = ''):
        team = self.home if team_idx == 0 else self.away
        opp = self.away if team_idx == 0 else self.home
        if team_idx == 0:
            self.result.home_goals += 1
        else:
            self.result.away_goals += 1

        # 자책골 — 오픈플레이에서만 가능 (세트피스는 자책골 없음)
        is_own_goal = (goal_type == 'open_play'
                        and self.visual_rng.random() < OWN_GOAL_CHANCE)
        if is_own_goal:
            scorer_name = self._pick_own_goal_scorer(1 - team_idx)
            self.own_goals += 1
            self.result.own_goals += 1
            opp_code = opp.code
            text = (f'OWN GOAL — {scorer_name} ({opp_code}, vs {team.code})'
                    if scorer_name else f'OWN GOAL — {team.code}')
            self.events.append(MatchEvent(self.minute, 'own_goal', team_idx, text))
            # 자책골은 player_goals에 누적 안 함
            recorded_scorer = ''
            recorded_assist = ''
        else:
            # 호출자가 scorer/assist를 안 줬으면 fallback (직접 호출 케이스)
            if not scorer_name:
                scorer_name = self._pick_scorer(team_idx, goal_type)
            if not assist_name and scorer_name:
                assist_name = self._pick_assister(team_idx, scorer_name)
            if scorer_name:
                key = (team_idx, scorer_name)
                self.player_goals[key] = self.player_goals.get(key, 0) + 1
                if assist_name:
                    akey = (team_idx, assist_name)
                    self.player_assists[akey] = self.player_assists.get(akey, 0) + 1
                    text = f'GOAL — {scorer_name} (assist: {assist_name}, {team.code})'
                else:
                    text = f'GOAL — {scorer_name} ({team.code})'
            else:
                text = f'GOAL — {team.code}'
            self.events.append(MatchEvent(self.minute, 'goal', team_idx, text))
            recorded_scorer = scorer_name
            recorded_assist = assist_name

        # goals_log 에 기록 (setpiece_counts / kickoff 페이즈 참조용)
        self.goals_log.append({
            'minute': self.minute,
            'tick': self.tick_count,
            'team_idx': team_idx,
            'type': goal_type,
            'scorer': recorded_scorer,
            'assist': recorded_assist,
        })
        self.last_goal_type = goal_type
        self.last_goal_team = team_idx

        self.goal_flash = 24
        self.goal_flash_team = team_idx
        self.possession_idx = 1 - team_idx

        # 골 후 kickoff phase 진입
        self.phase = 'kickoff'
        self.phase_start_tick = self.tick_count
        self.shot_state = None
        self.setpiece_state = None
        self.carrier_idx = None
        self.carrier_ttl = 0
        self.pass_in_flight = False
        self.pass_target_idx = None

        # 골 직후 즉시 전술 반응 (지고 있는 팀이 공격적으로, 이긴 팀은 잠그기)
        self._check_post_goal_tactical(team_idx)

    def _pick_own_goal_scorer(self, defending_team_idx: int) -> str:
        """자책골 — 수비팀에서 픽 (DEF 우선, GK 제외)."""
        cands = [p for p in self.players
                 if p.team_idx == defending_team_idx and p.on_pitch
                 and p.role != 'GK' and p.name]
        if not cands:
            return ''
        # DEF 70%, MID 20%, FWD 10% (자책골은 DEF가 가장 많이)
        weights = [{'DEF': 0.70, 'MID': 0.20, 'FWD': 0.10}.get(p.role, 0.10)
                   for p in cands]
        total = sum(weights)
        r = self.visual_rng.random() * total
        acc = 0.0
        for p, w in zip(cands, weights):
            acc += w
            if r <= acc:
                return p.name
        return cands[-1].name

    def _pick_assister(self, team_idx: int, scorer_name: str) -> str:
        """70% 확률 어시스트 부여. 미드 우선, 같은 팀의 다른 선수."""
        if self.visual_rng.random() > ASSIST_CHANCE:
            return ''
        if not scorer_name:
            return ''
        cands = [p for p in self.players
                 if p.team_idx == team_idx and p.on_pitch
                 and p.name and p.name != scorer_name]
        if not cands:
            return ''
        weights = []
        for p in cands:
            w = ASSIST_ROLE_W.get(p.role, 0.1)
            if p.is_star:
                w *= 1.9
            w *= max(0.6, (p.rating / 75.0) ** 1.5)
            weights.append(w)
        total = sum(weights)
        if total <= 0:
            return ''
        r = self.visual_rng.random() * total
        acc = 0.0
        for p, w in zip(cands, weights):
            acc += w
            if r <= acc:
                return p.name
        return cands[-1].name

    def _pick_scorer(self, team_idx: int, gtype: str = 'open_play') -> str:
        """골 종류별로 적절한 스코러 선택 (visual_rng 사용)."""
        # 팀 선수 풀 (visualization 모드에서만 player.name이 채워져 있음)
        team_players = [p for p in self.players
                         if p.team_idx == team_idx and p.on_pitch and p.name]
        if not team_players:
            return ''

        # 골 종류별 role 가중치
        if gtype == 'penalty':
            # PK 전담 키커 우선
            takers = [p for p in team_players if p.pk_taker]
            if takers:
                return takers[0].name
            role_w = {'FWD': 0.6, 'MID': 0.4, 'DEF': 0.0, 'GK': 0.0}
        elif gtype == 'corner':
            role_w = {'DEF': 0.45, 'FWD': 0.40, 'MID': 0.15, 'GK': 0.0}
        elif gtype == 'free_kick':
            role_w = {'MID': 0.50, 'FWD': 0.45, 'DEF': 0.05, 'GK': 0.0}
        else:  # open_play
            role_w = {'FWD': 0.62, 'MID': 0.30, 'DEF': 0.07, 'GK': 0.01}

        weights = []
        for p in team_players:
            w = role_w.get(p.role, 0.05)
            # 스타 가중치 ×2.6, rating 비례 (별 → 별 패턴 강화)
            if p.is_star:
                w *= 2.6
            w *= max(0.5, (p.rating / 75.0) ** 1.5)
            weights.append(w)
        total = sum(weights)
        if total <= 0:
            return self.visual_rng.choice(team_players).name
        r = self.visual_rng.random() * total
        acc = 0.0
        for p, w in zip(team_players, weights):
            acc += w
            if r <= acc:
                return p.name
        return team_players[-1].name

    # ── 종료 처리 ───────────────────────────────────────────

    def _finish_regulation(self):
        """정규시간(90+추가) 종료. 무승부+토너먼트면 ET 진입, 아니면 종료."""
        if self.result.home_goals == self.result.away_goals and self.knockout:
            # 연장전 30분 진입
            self.in_extra_time = True
            self.result.went_to_et = True
            self.extra_time_end_minute = self.minute + EXTRA_TIME_DURATION
            self.max_subs_for_team += EXTRA_TIME_BONUS_SUB    # +1 추가 교체
            self.events.append(MatchEvent(self.minute, 'half', -1,
                                            'EXTRA TIME'))
        else:
            self.finished = True
            self.events.append(MatchEvent(self.minute, 'fulltime', -1,
                                            'FULL TIME'))

    def _finish_extra_time(self):
        """연장전 종료. 무승부면 PK, 아니면 ET로 결판."""
        if self.result.home_goals == self.result.away_goals:
            self.in_pk = True
            self.in_extra_time = False
            self.home_pk_kickers = self._select_pk_kickers(0)
            self.away_pk_kickers = self._select_pk_kickers(1)
            self.events.append(MatchEvent(self.minute, 'half', -1,
                                            'TO PENALTIES'))
        else:
            self.finished = True
            self.in_extra_time = False
            self.result.decided_in_et = True
            self.events.append(MatchEvent(self.minute, 'fulltime', -1,
                                            'ET FULL TIME'))

    def _select_pk_kickers(self, team_idx: int) -> list:
        """PK 5인 키커 명단 — pk_taker 우선 → FWD/MID/DEF rating 순."""
        on_pitch = [p for p in self.players
                     if p.team_idx == team_idx and p.on_pitch
                     and p.role != 'GK' and p.name]
        if not on_pitch:
            return []
        designated = [p for p in on_pitch if p.pk_taker]
        rest = [p for p in on_pitch if not p.pk_taker]
        rest.sort(key=lambda p: (
            {'FWD': 0, 'MID': 1, 'DEF': 2}.get(p.role, 3),
            -p.rating,
        ))
        ordered = designated + rest
        return ordered[:5]

    def _tick_pk(self):
        """PK 5라운드 + sudden death. 키커는 미리 정해진 5인 명단 순서."""
        pk_period = max(4, int(SIM_TICKS_PER_MIN * 1.5))
        if self.tick_count % pk_period != 0:
            self.tick_count += 1
            return
        self.tick_count += 1
        self.pk_round += 1

        round_idx = (self.pk_round - 1) % 5   # 5명 순환 (sudden death도)
        for shooter_idx, kickers, taker_team, gk_team in [
            (0, self.home_pk_kickers, self.home, self.away),
            (1, self.away_pk_kickers, self.away, self.home),
        ]:
            kicker = kickers[round_idx] if kickers else None
            kicker_attack = kicker.rating if kicker else taker_team.attack
            kicker_name = kicker.name if kicker else taker_team.code
            edge = (kicker_attack - gk_team.keeper) / 25.0
            score_p = 0.55 + 0.5 / (1.0 + math.exp(-edge))
            score_p = min(0.97, score_p * (1.0 + taker_team.wc_titles * 0.015))
            if self.rng.random() < score_p:
                if shooter_idx == 0:
                    self.result.home_pk += 1
                else:
                    self.result.away_pk += 1
                self.events.append(MatchEvent(
                    90, 'pk', shooter_idx,
                    f'PK GOAL ({taker_team.code}) {kicker_name}'))
            else:
                self.events.append(MatchEvent(
                    90, 'pk', shooter_idx,
                    f'PK MISS ({taker_team.code}) {kicker_name}'))

        if self.pk_round >= 5:
            if self.result.home_pk != self.result.away_pk:
                self.result.went_to_pk = True
                self.finished = True
                self.events.append(MatchEvent(90, 'fulltime', -1, 'PK DECIDED'))


def _populate_result_player_stats(m: 'Match'):
    """매치 결과에 player_goals/assists/own_goals + appearances 복사."""
    m.result.player_goals = dict(m.player_goals)
    m.result.player_assists = dict(m.player_assists)
    m.result.own_goals = m.own_goals
    seen = set()
    appearances = []
    for p in m.players:
        if not p.name or p.name in seen:
            continue
        # 출전한 적 있는 선수 (선발은 항상, 후보는 sub_in_minute>0)
        played = p.is_starter or p.sub_in_minute > 0
        if not played:
            continue
        seen.add(p.name)
        appearances.append((p.team_idx, p.name, p.role, p.rating, p.is_star))
    m.result.appearances = appearances


def play_full(home: Team, away: Team, knockout: bool,
              rng: random.Random,
              altitude: str = 'low', hot: bool = False,
              last_round_push: bool = False,
              home_familiar: bool = False,
              away_familiar: bool = False,
              home_starting_stamina: int = 100,
              away_starting_stamina: int = 100) -> MatchResult:
    """경기를 끝까지 즉시 진행 (시각화 없이)."""
    m = Match(home, away, knockout, rng,
              altitude=altitude, hot=hot, last_round_push=last_round_push,
              home_familiar=home_familiar, away_familiar=away_familiar,
              home_starting_stamina=home_starting_stamina,
              away_starting_stamina=away_starting_stamina)
    while not m.finished:
        m.tick()
    _populate_result_player_stats(m)
    return m.result


def play_full_fast(home: Team, away: Team, knockout: bool,
                   rng: random.Random,
                   altitude: str = 'low', hot: bool = False,
                   last_round_push: bool = False,
                   home_familiar: bool = False,
                   away_familiar: bool = False,
                   home_starting_stamina: int = 100,
                   away_starting_stamina: int = 100) -> MatchResult:
    """결과만 빠르게 — 매 tick 시각 업데이트 + 일정 precompute 스킵.

    RNG 경로가 시각화 tick과 동일 (visual_rng만 분리). 헤드리스 MC용.
    """
    m = Match(home, away, knockout, rng, precompute=False,
              altitude=altitude, hot=hot, last_round_push=last_round_push,
              home_familiar=home_familiar, away_familiar=away_familiar,
              home_starting_stamina=home_starting_stamina,
              away_starting_stamina=away_starting_stamina)
    while not m.finished:
        m.tick_fast()
    _populate_result_player_stats(m)
    return m.result
