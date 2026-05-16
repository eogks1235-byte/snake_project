"""대회 진행 — 모든 경기(예선 72 + 토너먼트 32)가 시각화 큐로 흘러간다.

흐름:
  1) 초기화: 모든 팀 능력치 생성, 예선 72경기를 큐에 넣음
  2) main이 큐에서 한 경기씩 꺼내 시각화
  3) 예선 마지막 경기 결과 등록 시 → 조 정렬 + 3위 선발 + R32 시드
  4) R32 결과 등록 시 → R16 큐에 추가, 등등
"""
import random
from dataclasses import dataclass
from typing import Optional

from .data import (
    GROUPS, BRACKET_MATCHES, KO_FOLLOWUPS, THIRD_PLACE_SLOTS, STAGE_LABELS,
)
from .teams import Team, build_team, palette_for
from .match_engine import MatchResult, MatchEvent


GROUP_MATCHES_TOTAL = 72  # 12조 × 6경기


# ──────────────────────────────────────────────────────────────
# 2026 북중미 월드컵 16개 경기장
# (altitude, hot, country)
# ──────────────────────────────────────────────────────────────
VENUES_2026 = {
    # Mexico (3) — Mexico City 고도 ~2240m
    'Mexico City':   ('high', False, 'MEX'),
    'Guadalajara':   ('mid',  False, 'MEX'),  # ~1566m
    'Monterrey':     ('low',  True,  'MEX'),  # 더운 사막
    # Canada (2) — 시원/온화
    'Toronto':       ('low',  False, 'CAN'),
    'Vancouver':     ('low',  False, 'CAN'),
    # USA (11) — 북부 시원, 남부/중부 더움
    'Atlanta':       ('low',  True,  'USA'),
    'Boston':        ('low',  False, 'USA'),
    'Dallas':        ('low',  True,  'USA'),
    'Houston':       ('low',  True,  'USA'),
    'Kansas City':   ('low',  True,  'USA'),
    'Los Angeles':   ('low',  False, 'USA'),
    'Miami':         ('low',  True,  'USA'),
    'New York/NJ':   ('low',  False, 'USA'),
    'Philadelphia':  ('low',  False, 'USA'),
    'San Francisco': ('low',  False, 'USA'),
    'Seattle':       ('low',  False, 'USA'),
}

# 그룹별 매치 6개의 venue 풀 (실제 host 국가 위주 + 인접 분산)
GROUP_VENUES = {
    'A': ['Mexico City', 'Guadalajara', 'Monterrey',
          'Mexico City', 'Guadalajara', 'Monterrey'],          # MEX host
    'B': ['Toronto', 'Vancouver', 'Toronto',
          'Vancouver', 'Boston', 'Philadelphia'],              # CAN host (+미국 일부)
    'C': ['Mexico City', 'Atlanta', 'Houston',
          'Toronto', 'Boston', 'Guadalajara'],
    'D': ['Dallas', 'Houston', 'Kansas City',
          'Atlanta', 'Miami', 'Los Angeles'],                  # USA host
    'E': ['Philadelphia', 'New York/NJ', 'Boston',
          'Toronto', 'Miami', 'Atlanta'],
    'F': ['Los Angeles', 'San Francisco', 'Seattle',
          'Vancouver', 'Kansas City', 'Houston'],
    'G': ['Atlanta', 'Miami', 'Dallas',
          'Houston', 'Philadelphia', 'New York/NJ'],
    'H': ['New York/NJ', 'Boston', 'Philadelphia',
          'Atlanta', 'Miami', 'Houston'],
    'I': ['Mexico City', 'Guadalajara', 'Dallas',
          'Houston', 'Atlanta', 'Kansas City'],
    'J': ['Los Angeles', 'San Francisco', 'Seattle',
          'Vancouver', 'Boston', 'New York/NJ'],
    'K': ['Monterrey', 'Mexico City', 'Houston',
          'Dallas', 'Kansas City', 'Atlanta'],
    'L': ['Toronto', 'Vancouver', 'Boston',
          'New York/NJ', 'Philadelphia', 'Miami'],
}

# 토너먼트 매치 → venue (FIFA 공식 발표 기반)
KNOCKOUT_VENUES = {
    # 결승: 뉴욕/뉴저지
    'M104': 'New York/NJ',
    # 3·4위전: 마이애미
    'M103': 'Miami',
    # 4강
    'M101': 'Atlanta',
    'M102': 'Dallas',
    # 8강 (공식 발표)
    'M97':  'Boston',
    'M98':  'Kansas City',
    'M99':  'Los Angeles',
    'M100': 'Miami',
    # 16강
    'M89': 'Atlanta',     'M90': 'Mexico City',
    'M91': 'Dallas',      'M92': 'New York/NJ',
    'M93': 'Philadelphia','M94': 'Houston',
    'M95': 'San Francisco', 'M96': 'Toronto',
    # 32강
    'M73': 'Mexico City', 'M74': 'Toronto',
    'M75': 'Vancouver',   'M76': 'Atlanta',
    'M77': 'Houston',     'M78': 'Dallas',
    'M79': 'Boston',      'M80': 'New York/NJ',
    'M81': 'Kansas City', 'M82': 'Philadelphia',
    'M83': 'Los Angeles', 'M84': 'Miami',
    'M85': 'San Francisco','M86': 'Seattle',
    'M87': 'Guadalajara', 'M88': 'Monterrey',
}


# 매치 → 라운드 깊이 (그 매치에 등장 = 그 라운드까지 도달)
def _round_for_match(mid: str) -> int:
    if mid.startswith('G'):
        return 0
    n = int(mid[1:])
    if n <= 88: return 1   # R32 진출
    if n <= 96: return 2   # R16 진출
    if n <= 100: return 3  # QF 진출
    if n <= 103: return 4  # SF/3·4위전 진출
    return 5               # Final 진출


@dataclass
class PlayerStats:
    name: str
    country: str
    role: str
    rating: int
    is_star: bool
    goals: int = 0
    assists: int = 0
    matches: int = 0
    team_round: int = 0     # 0=조별 탈락, 1=R32, 2=R16, 3=QF, 4=SF, 5=결승, 6=우승

    def mvp_score(self) -> float:
        s = (self.goals * 1.5
             + self.assists * 0.7
             + self.team_round * 0.5
             + self.rating * 0.02
             + (0.4 if self.is_star else 0))
        if self.team_round == 6:
            s += 1.5
        s *= (0.7 + min(1.0, self.matches / 7.0) * 0.3)
        return s

    def goals_plus_assists(self) -> float:
        return self.goals + self.assists * 0.5


def match_sort_key(mid: str) -> tuple:
    """예선 → 토너먼트 순서. GA1<GA2<...<GL6<M73<M74<...<M104"""
    if mid.startswith('G'):
        return (0, mid[1], int(mid[2:]))
    return (1, '', int(mid[1:]))


class Tournament:
    def __init__(self, seed: int, real_results: dict = None):
        self.seed = seed
        self.rng = random.Random(seed)
        # 사용자가 입력한 실제 경기 결과 (재시뮬용)
        self.real_results_data: dict = real_results or {}

        # 1) 팀 생성
        self.teams: dict = {}
        self.group_teams: dict = {letter: [] for letter in GROUPS}
        for letter, td_list in GROUPS.items():
            for td in td_list:
                primary, secondary = palette_for(td, self.rng)
                team = build_team(td, primary, secondary, self.rng)
                self.teams[td.code] = team
                self.group_teams[letter].append(team)
                team.reset_stats()

        # 2) 예선 큐 — 'GA1' ~ 'GL6' (조별 6경기 × 12조)
        self.match_queue: list = []
        self._enqueue_group_matches()

        # 진행 상태
        self.results: dict = {}              # match_id -> MatchResult
        self.group_results: dict = {}        # letter -> [Team] sorted (계산은 예선 끝 후)
        self.group_third_place: list = []
        self.qualifying_thirds: list = []
        self.third_assignment: dict = {}
        self.groups_finalized = False
        self.completed = False

        # 2026 매치 조건 — match_id별 (altitude, hot, last_round_push, venue)
        self.match_conditions: dict = self._assign_match_conditions()
        # 팀별 방문 venue 추적 (재방문 친숙도 보너스용)
        self.team_venues: dict = {}
        # 선수별 누적 통계 (Awards 계산용)
        self.player_stats: dict = {}    # name → PlayerStats
        # 매치별 통계 스냅샷 — 득점왕/어시왕 라인차트 애니메이션용
        # [(match_id, {name: {goals, assists, country, role}}), ...]
        self.stat_history: list = []
        # 다음 경기 시작 stamina (ET로 이긴 팀은 90)
        self.team_stamina_carry: dict = {}     # team_code → 다음 매치 stamina

        # bootstrap: 큐에 들어 있는 실제 결과들을 미리 주입
        self._try_inject_real()

    def _assign_match_conditions(self) -> dict:
        """매치별 조건 — 2026 실제 16개 venue 기반.

        예선: 그룹별 venue 풀을 match_id 순서대로 배정 (시드 무관 고정)
        토너: KNOCKOUT_VENUES (FIFA 공식 발표) 그대로
        고도/더위: venue에서 자동 도출
        """
        cond: dict = {}

        # 예선 — match_id별 venue 고정 (셔플 없음 → 같은 GA1은 항상 같은 경기장)
        for letter, venues in GROUP_VENUES.items():
            for r in range(1, 7):
                mid = f'G{letter}{r}'
                venue = venues[r - 1]
                alt, hot, country = VENUES_2026[venue]
                cond[mid] = {
                    'altitude': alt, 'hot': hot,
                    'last_round_push': (r == 6),
                    'venue': venue, 'host_country': country,
                }

        # 토너먼트 — FIFA 공식 venue 매핑
        for n in range(73, 105):
            mid = f'M{n}'
            venue = KNOCKOUT_VENUES.get(mid, 'New York/NJ')
            alt, hot, country = VENUES_2026[venue]
            cond[mid] = {
                'altitude': alt, 'hot': hot,
                'last_round_push': False,
                'venue': venue, 'host_country': country,
            }
        return cond

    def conditions_of(self, match_id: str) -> dict:
        return self.match_conditions.get(match_id, {
            'altitude': 'low', 'hot': False, 'last_round_push': False,
            'venue': '?', 'host_country': '?',
        })

    # ── 선수 통계 + Awards ─────────────────────────────────

    def _update_player_stats(self, match_id: str, result: MatchResult):
        rnd = _round_for_match(match_id)
        is_final = (match_id == 'M104')
        winner_code = result.winner.code if result.winner else None

        for (team_idx, name, role, rating, is_star) in result.appearances:
            country = result.home.code if team_idx == 0 else result.away.code
            stats = self.player_stats.get(name)
            if stats is None:
                stats = PlayerStats(
                    name=name, country=country, role=role,
                    rating=rating, is_star=is_star,
                )
                self.player_stats[name] = stats
            stats.matches += 1
            if rnd > stats.team_round:
                stats.team_round = rnd
            if is_final and country == winner_code:
                stats.team_round = 6

        for (team_idx, name), goals in result.player_goals.items():
            if name in self.player_stats:
                self.player_stats[name].goals += goals
        # 어시스트 누적
        for (team_idx, name), assists in result.player_assists.items():
            if name in self.player_stats:
                self.player_stats[name].assists += assists

        # 매치별 스냅샷 — 라인차트 애니메이션용
        snap = {
            name: {
                'goals': s.goals,
                'assists': s.assists,
                'country': s.country,
                'role': s.role,
            }
            for name, s in self.player_stats.items()
        }
        self.stat_history.append((match_id, snap))

    def compute_awards(self) -> dict:
        """대회 종료 시 어워드 계산. completed=False여도 현재 시점 통계로 반환."""
        if not self.player_stats:
            return None
        all_stats = list(self.player_stats.values())

        # Golden Boot — 골 → 어시스트 → 적은 출전 시간 → rating
        golden_boot = max(all_stats,
                           key=lambda p: (p.goals, p.assists, -p.matches, p.rating))

        # MVP — composite score
        mvp = max(all_stats, key=lambda p: p.mvp_score())

        # 포지션별 최고
        best_per_role = {}
        for role in ('GK', 'DEF', 'MID', 'FWD'):
            cands = [p for p in all_stats if p.role == role]
            if cands:
                best_per_role[role] = max(cands, key=lambda p: p.mvp_score())

        # Best XI (4-3-3): 1 GK + 4 DEF + 3 MID + 3 FWD
        target = {'GK': 1, 'DEF': 4, 'MID': 3, 'FWD': 3}
        best_xi: list = []
        for role, n in target.items():
            cands = sorted([p for p in all_stats if p.role == role],
                            key=lambda p: -p.mvp_score())
            best_xi.extend(cands[:n])

        return {
            'golden_boot': golden_boot,
            'mvp': mvp,
            'best_per_role': best_per_role,
            'best_xi': best_xi,
        }

    def familiar(self, team_code: str, venue: str) -> bool:
        """대회 중 이 팀이 이 경기장을 한 번이라도 사용한 적 있는가."""
        return venue in self.team_venues.get(team_code, set())

    def starting_stamina_for(self, team_code: str) -> int:
        """다음 매치 시작 stamina. ET로 이긴 팀은 90, 일반은 100. 1회용."""
        if team_code in self.team_stamina_carry:
            val = self.team_stamina_carry.pop(team_code)
            return val
        return 100

    def _record_venue_visit(self, team_code: str, venue: str):
        if team_code not in self.team_venues:
            self.team_venues[team_code] = set()
        self.team_venues[team_code].add(venue)

    # ── 예선 큐 시드 ─────────────────────────────────────

    def _enqueue_group_matches(self):
        """각 조 4팀의 라운드로빈 6경기를 큐에 추가.

        매치 ID: G{letter}{n} (n=1..6)
        예선이 너무 한쪽 조에 몰리지 않도록 라운드 단위로 인터리브.
        """
        per_group: dict = {}    # letter -> [(home, away)] 6개
        for letter, teams in self.group_teams.items():
            pairs = []
            for i in range(len(teams)):
                for j in range(i + 1, len(teams)):
                    pairs.append((teams[i], teams[j]))
            per_group[letter] = pairs

        # 라운드 단위로 인터리브: round 0의 모든 조 → round 1 → ...
        for r in range(6):
            for letter in GROUPS:
                home, away = per_group[letter][r]
                match_id = f'G{letter}{r + 1}'
                self.match_queue.append((match_id, home, away))

    # ── 외부 인터페이스 ──────────────────────────────────

    def has_next_match(self) -> bool:
        return bool(self.match_queue) and not self.completed

    def peek_next(self) -> Optional[tuple]:
        return self.match_queue[0] if self.match_queue else None

    def is_knockout(self, match_id: str) -> bool:
        """예선은 무승부 허용, 토너먼트는 PK."""
        return not match_id.startswith('G')

    def record_result(self, match_id: str, result: MatchResult):
        self.results[match_id] = result
        # 큐에서 해당 매치 제거 (위치 무관 — real_results 주입 시 순서가 다를 수 있음)
        self.match_queue = [m for m in self.match_queue if m[0] != match_id]
        # 양 팀에 venue 방문 기록 (다음 매치에 친숙도 ↑)
        venue = self.match_conditions.get(match_id, {}).get('venue')
        if venue and venue != '?':
            self._record_venue_visit(result.home.code, venue)
            self._record_venue_visit(result.away.code, venue)
        # 선수별 통계 누적
        self._update_player_stats(match_id, result)
        # ET 결정으로 이긴 팀: 다음 매치 stamina 90으로 시작
        if result.decided_in_et and result.winner is not None:
            from .match_engine import ET_WINNER_STAMINA
            self.team_stamina_carry[result.winner.code] = ET_WINNER_STAMINA
        # 정상 종료 후엔 carry 사용 후 reset
        for code in (result.home.code, result.away.code):
            if code in self.team_stamina_carry:
                # 이번 매치에 적용 후 다음엔 100으로 복귀 (carry는 1회용)
                # → 매치 시작 시 read 했고 끝나면 제거
                pass    # 시작 시 처리 — 아래 starting_stamina_for 참조

        if match_id.startswith('G'):
            self._apply_group_result(result)
            # 모든 예선 끝나면 조 정리 + R32 시드
            group_done = sum(1 for k in self.results if k.startswith('G'))
            if group_done >= GROUP_MATCHES_TOTAL and not self.groups_finalized:
                self._finalize_groups()
                self._seed_r32()
        else:
            # 토너먼트 — 다음 라운드 매치를 큐에 추가
            for next_id, src_h, src_a in KO_FOLLOWUPS:
                if next_id in self.results:
                    continue
                home = self._resolve_token(src_h)
                away = self._resolve_token(src_a)
                if home is None or away is None:
                    continue
                if any(mid == next_id for mid, _, _ in self.match_queue):
                    continue
                self.match_queue.append((next_id, home, away))

            if 'M104' in self.results:
                self.completed = True

        # 새로 큐에 추가된 매치 중 실제 결과가 있으면 그것도 주입
        self._try_inject_real()

    # ── 실제 결과 주입 ─────────────────────────────────

    def _try_inject_real(self):
        """real_results_data 중 큐에 있는 매치를 자동으로 record."""
        for match_id in list(self.real_results_data.keys()):
            if match_id in self.results:
                continue
            target = next((m for m in self.match_queue if m[0] == match_id), None)
            if target is None:
                continue
            _, home, away = target
            data = self.real_results_data[match_id]
            # 선택적 home/away 코드 검증
            if 'home' in data and home.code != data['home']:
                print(f'[REAL WARN] {match_id} home mismatch: '
                      f'expected {home.code}, got {data["home"]}')
            if 'away' in data and away.code != data['away']:
                print(f'[REAL WARN] {match_id} away mismatch: '
                      f'expected {away.code}, got {data["away"]}')
            res = MatchResult(
                home=home, away=away,
                home_goals=int(data['home_goals']),
                away_goals=int(data['away_goals']),
            )
            if data.get('pk_home') is not None:
                res.went_to_pk = True
                res.home_pk = int(data['pk_home'])
                res.away_pk = int(data['pk_away'])
            res.events.append(MatchEvent(
                minute=90, kind='real', team_idx=-1,
                text=f"REAL {home.code} {res.score_str()} {away.code}"
            ))
            # record_result가 후속 처리 + 재귀적으로 다음 real 주입
            self.record_result(match_id, res)
            return  # 재귀가 나머지 처리

    # ── 예선 결과 누적 ──────────────────────────────────

    def _apply_group_result(self, res: MatchResult):
        h, a = res.home, res.away
        h.goals_for += res.home_goals
        h.goals_against += res.away_goals
        a.goals_for += res.away_goals
        a.goals_against += res.home_goals
        if res.home_goals > res.away_goals:
            h.wins += 1; a.losses += 1; h.points += 3
        elif res.away_goals > res.home_goals:
            a.wins += 1; h.losses += 1; a.points += 3
        else:
            h.draws += 1; a.draws += 1; h.points += 1; a.points += 1

    def _finalize_groups(self):
        for letter, teams in self.group_teams.items():
            ranked = sorted(
                teams,
                key=lambda t: (t.points, t.goal_diff(), t.goals_for),
                reverse=True,
            )
            self.group_results[letter] = ranked
            self.group_third_place.append((letter, ranked[2]))
        self._select_third_places()
        self.groups_finalized = True

    # ── 3위 8팀 선발 + 슬롯 매칭 ────────────────────────

    def _select_third_places(self):
        ranked = sorted(
            self.group_third_place,
            key=lambda lt: (lt[1].points, lt[1].goal_diff(), lt[1].goals_for),
            reverse=True,
        )
        self.qualifying_thirds = ranked[:8]
        qualifying_letters = {letter for letter, _ in self.qualifying_thirds}

        slots = list(THIRD_PLACE_SLOTS.items())
        slots.sort(key=lambda s: len(s[1] & qualifying_letters))

        third_team_by_letter = dict(self.qualifying_thirds)
        used = set()
        for slot_code, candidate_letters in slots:
            for letter, _ in self.qualifying_thirds:
                if letter in used:
                    continue
                if letter not in candidate_letters:
                    continue
                self.third_assignment[slot_code] = third_team_by_letter[letter]
                used.add(letter)
                break

        for slot_code, _ in slots:
            if slot_code not in self.third_assignment:
                for letter, _ in self.qualifying_thirds:
                    if letter not in used:
                        self.third_assignment[slot_code] = third_team_by_letter[letter]
                        used.add(letter)
                        break

    # ── R32 시드 ────────────────────────────────────────

    def _seed_r32(self):
        for match_id, (slot_h, slot_a) in BRACKET_MATCHES.items():
            home = self._resolve_slot(slot_h)
            away = self._resolve_slot(slot_a)
            self.match_queue.append((match_id, home, away))
        # M73 ~ M88 순서대로
        self.match_queue.sort(key=lambda m: int(m[0][1:]) if m[0].startswith('M') else 9999)

    def _resolve_slot(self, slot: str) -> Optional[Team]:
        if slot.startswith('3'):
            return self.third_assignment.get(slot)
        rank = int(slot[0])
        letter = slot[1]
        return self.group_results[letter][rank - 1]

    def _resolve_token(self, token: str) -> Optional[Team]:
        if token.startswith('W') or token.startswith('L'):
            mid = 'M' + token[1:]
            r = self.results.get(mid)
            if r is None:
                return None
            return r.winner if token.startswith('W') else r.loser
        return None

    # ── UI 보조 ─────────────────────────────────────────

    def stage_of(self, match_id: str) -> str:
        if match_id.startswith('G'):
            return f'GROUP {match_id[1]}'
        return STAGE_LABELS.get(match_id, '?')

    def champion(self) -> Optional[Team]:
        r = self.results.get('M104')
        return r.winner if r else None

    def runner_up(self) -> Optional[Team]:
        r = self.results.get('M104')
        return r.loser if r else None

    def third(self) -> Optional[Team]:
        r = self.results.get('M103')
        return r.winner if r else None
