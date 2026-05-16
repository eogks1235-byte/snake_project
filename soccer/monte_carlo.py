"""몬테카를로 시뮬레이션 — N회 헤드리스 토너먼트 후 가장 극적인 한 판 선정.

극적 점수 (drama score) 산출:
  - 골: stage_weight × 골 수
  - PK 승부차기: stage_weight × 8
  - 토너먼트 업셋(낮은 OVR 팀이 이김): stage_weight × OVR_diff × 0.5
  - 언더독 챔피언: (1700 - FIFA pts) × 0.04 (1700 미만일수록 큰 보너스)

stage_weight:
  Group=1.0 / R32=1.6 / R16=2.2 / QF=3.2 / SF=4.2 / 3rd=2.0 / Final=6.0

저장:
  recordings/mc_<timestamp>/
    matches/GA1.json ... GL6.json  (예선 72개)
    matches/M73.json ... M104.json (토너 32개)
    summary.json                    (집계 통계 + 선정 정보)
"""
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .tournament import Tournament, match_sort_key
from .match_engine import play_full_fast, MatchResult
from .teams import Team


# ── 점수 계산 ────────────────────────────────────────────

def stage_weight(match_id: str) -> float:
    if match_id.startswith('G'):
        return 1.0
    n = int(match_id[1:])
    if n <= 88: return 1.6
    if n <= 96: return 2.2
    if n <= 100: return 3.2
    if n <= 102: return 4.2
    if n == 103: return 2.0
    return 6.0


def drama_score(t: Tournament) -> float:
    score = 0.0
    for mid, res in t.results.items():
        sw = stage_weight(mid)
        # 골 수
        score += (res.home_goals + res.away_goals) * sw
        # PK
        if res.went_to_pk:
            score += 8 * sw
        # 업셋 — 토너먼트 한정
        if (not mid.startswith('G')) and res.winner is not None and res.loser is not None:
            ovr_diff = res.loser.overall - res.winner.overall
            if ovr_diff > 0:
                score += ovr_diff * sw * 0.5

    champ = t.champion()
    if champ is not None:
        underdog = max(0.0, 1700.0 - champ.data.fifa_points)
        score += underdog * 0.04
    return score


def thriller_score(t: Tournament) -> float:
    """tight + goals + high_final 합성:
       박빙(작은 골 차) + 다득점 + 결승 격렬, 셋 다 강해야 점수 ↑."""
    score = 0.0
    for mid, res in t.results.items():
        sw = stage_weight(mid)
        total = res.home_goals + res.away_goals
        margin = abs(res.home_goals - res.away_goals)
        # 골 보상
        score += total * sw * 1.5
        # 2골 이상 차 페널티 (시소 ↓)
        score -= max(0, margin - 1) * sw * 2.5
        # PK = tight + tense → 보너스
        if res.went_to_pk:
            score += sw * 5

    # 결승 특별 가중 (high_final)
    final = t.results.get('M104')
    if final is not None:
        ftotal = final.home_goals + final.away_goals
        score += ftotal * 8
        if final.went_to_pk:
            score += 25
    return score


def realistic_score(t: Tournament) -> float:
    """현실적/평균적인 토너먼트:
       골/PK/업셋 수가 실제 WC 평균에 가까울수록 점수 ↑.
       + 강팀(FIFA pts ↑)이 적당히 멀리 가는 결과 선호."""
    # FIFA 실제 통계 기준 타겟 (평균값)
    TARGET_GOALS = 280       # ~2.7 goals × 104 matches
    TARGET_PK = 3            # 한 대회당 PK 결판 ~3-4회
    TARGET_UPSETS = 5        # 토너먼트 단계 큰 업셋 ~5회

    total_goals = 0
    pk_count = 0
    upset_count = 0
    for mid, res in t.results.items():
        total_goals += res.home_goals + res.away_goals
        if res.went_to_pk:
            pk_count += 1
        if (not mid.startswith('G')) and res.winner is not None and res.loser is not None:
            ovr_diff = res.loser.overall - res.winner.overall
            if ovr_diff > 5:
                upset_count += 1

    # 각 항목별 거리 (작을수록 평균에 가까움)
    d_goals = abs(total_goals - TARGET_GOALS) / 40.0
    d_pk = abs(pk_count - TARGET_PK) / 3.0
    d_upset = abs(upset_count - TARGET_UPSETS) / 5.0

    # 강팀이 챔피언이 될수록 ↑ (현실성)
    champ = t.champion()
    realism = 0.0
    if champ is not None:
        realism = (champ.data.fifa_points - 1500) / 50.0  # 1700 → 4, 1500 → 0

    # 음수 거리 + 양수 realism → 점수 ↑일수록 평균적
    return realism - (d_goals + d_pk + d_upset) * 3.0


# 토너 매치 ID → round 깊이 (팀 응원 점수 계산용)
_ROUND_OF = {}
for _n in range(73, 89): _ROUND_OF[_n] = 1   # R32
for _n in range(89, 97): _ROUND_OF[_n] = 2   # R16
for _n in range(97, 101): _ROUND_OF[_n] = 3  # QF
for _n in (101, 102, 103): _ROUND_OF[_n] = 4 # SF/3·4위전
_ROUND_OF[104] = 5                            # 결승


def team_furthest_round(t: Tournament, code: str) -> int:
    """0=조별 탈락, 1=R32 out, 2=R16 out, 3=QF out, 4=4강 진출, 5=결승 진출, 6=우승."""
    max_r = 0
    for mid, res in t.results.items():
        if not mid.startswith('M'):
            continue
        if res.home.code != code and res.away.code != code:
            continue
        n = int(mid[1:])
        max_r = max(max_r, _ROUND_OF.get(n, 0))
        if n == 104 and res.winner is not None and res.winner.code == code:
            return 6
    return max_r


def team_score(t: Tournament, code: str) -> float:
    """팀이 가장 멀리 간 정도 (큰 가중) + 그 팀 득점 + drama 보조 tiebreak."""
    depth = team_furthest_round(t, code)
    team_goals = 0
    team_conceded = 0
    for res in t.results.values():
        if res.home.code == code:
            team_goals += res.home_goals
            team_conceded += res.away_goals
        elif res.away.code == code:
            team_goals += res.away_goals
            team_conceded += res.home_goals
    # depth 차이가 압도적, 같은 depth끼리는 득점 ↑ 실점 ↓ 우선
    return (depth * 100_000
            + team_goals * 200
            - team_conceded * 80
            + drama_score(t) * 0.05)


def get_score_fn(strategy: str):
    """문자열 → 점수 함수. 'drama' / 'thriller' / 'realistic' / 'team:CODE'"""
    if not strategy or strategy == 'drama':
        return drama_score, 'drama'
    if strategy == 'thriller':
        return thriller_score, 'thriller'
    if strategy == 'realistic':
        return realistic_score, 'realistic'
    if strategy.startswith('team:'):
        code = strategy.split(':', 1)[1].strip().upper()
        return (lambda t: team_score(t, code)), f'team:{code}'
    raise ValueError(f"unknown selection strategy: {strategy}")


# ── MC 러너 ─────────────────────────────────────────────

@dataclass
class RunSummary:
    seed: int
    drama: float
    champion: str
    runner_up: str
    third: str
    pk_count: int
    upset_count: int
    total_goals: int


def run_montecarlo(base_seed: int, n_runs: int,
                   real_results: dict = None,
                   verbose: bool = True,
                   strategy: str = 'drama') -> dict:
    """N회 토너먼트 헤드리스 시뮬. real_results(매치ID→실제 결과)는 모든 런에서 고정.

    strategy:
      'drama'      — 골 + PK + 업셋 + 언더독 챔피언 (default)
      'thriller'   — tight + goals + high_final 합성
      'team:CODE'  — 그 팀이 가장 멀리 간 한 판 (예: team:KOR)
    """
    score_fn, strategy_label = get_score_fn(strategy)
    summaries: list = []
    best_seed: Optional[int] = None
    best_score = float('-inf')
    best_tournament: Optional[Tournament] = None

    for i in range(n_runs):
        seed = base_seed + i
        t = _play_full_tournament(seed, real_results)
        score = score_fn(t)

        # 빠른 통계
        pk_count = sum(1 for r in t.results.values() if r.went_to_pk)
        upsets = 0
        for mid, res in t.results.items():
            if mid.startswith('G') or res.winner is None or res.loser is None:
                continue
            if res.loser.overall > res.winner.overall:
                upsets += 1
        total_goals = sum(r.home_goals + r.away_goals for r in t.results.values())

        summaries.append(RunSummary(
            seed=seed,
            drama=score,
            champion=t.champion().code if t.champion() else '',
            runner_up=t.runner_up().code if t.runner_up() else '',
            third=t.third().code if t.third() else '',
            pk_count=pk_count,
            upset_count=upsets,
            total_goals=total_goals,
        ))

        if score > best_score:
            best_score = score
            best_seed = seed
            best_tournament = t

        if verbose and (i + 1) % 100 == 0:
            print(f'  MC {i+1}/{n_runs}  best[{strategy_label}]={best_score:.1f}  '
                  f'(champion={best_tournament.champion().code if best_tournament else "?"})')

    return {
        'summaries': summaries,
        'selected_seed': best_seed,
        'selected_tournament': best_tournament,
        'selected_score': best_score,
        'base_seed': base_seed,
        'n_runs': n_runs,
        'real_results_count': len(real_results) if real_results else 0,
        'strategy': strategy_label,
    }


def match_seed_offset(match_id: str) -> int:
    """match_id → 결정적 정수 (Python hash 랜덤화 회피).

    GA1=1, GA2=2, ..., GL6=72  /  M73=73 ~ M104=104
    """
    if match_id.startswith('G'):
        letter_idx = ord(match_id[1]) - ord('A')
        n = int(match_id[2:])
        return letter_idx * 6 + n
    return int(match_id[1:])


def _play_full_tournament(seed: int, real_results: dict = None) -> Tournament:
    t = Tournament(seed, real_results=real_results)
    while t.has_next_match():
        mid, h, a = t.peek_next()
        rng = random.Random(seed * 7919 + match_seed_offset(mid))
        cond = t.conditions_of(mid)
        venue = cond.get('venue', '')
        res = play_full_fast(h, a, knockout=t.is_knockout(mid), rng=rng,
                              altitude=cond['altitude'],
                              hot=cond['hot'],
                              last_round_push=cond['last_round_push'],
                              home_familiar=t.familiar(h.code, venue),
                              away_familiar=t.familiar(a.code, venue),
                              home_starting_stamina=t.starting_stamina_for(h.code),
                              away_starting_stamina=t.starting_stamina_for(a.code))
        t.record_result(mid, res)
    return t


# ── 직렬화 ──────────────────────────────────────────────

def _team_dict(team: Team, goals: int) -> dict:
    return {
        'code': team.code,
        'iso2': team.iso2,
        'name': team.name,
        'formation': team.formation,
        'style_tag': team.style_tag,
        'wc_titles': team.wc_titles,
        'overall': team.overall,
        'attack': team.attack,
        'defense': team.defense,
        'midfield': team.midfield,
        'keeper': team.keeper,
        'goals': goals,
    }


def _match_dict(mid: str, stage: str, res: MatchResult) -> dict:
    return {
        'match_id': mid,
        'stage': stage,
        'home': _team_dict(res.home, res.home_goals),
        'away': _team_dict(res.away, res.away_goals),
        'score': {
            'home': res.home_goals,
            'away': res.away_goals,
            'went_to_pk': res.went_to_pk,
            'pk_home': res.home_pk if res.went_to_pk else None,
            'pk_away': res.away_pk if res.went_to_pk else None,
            'winner': res.winner.code if res.winner else None,
            'loser': res.loser.code if res.loser else None,
        },
        'events': [
            {
                'minute': e.minute,
                'kind': e.kind,
                'team': (res.home.code if e.team_idx == 0
                         else (res.away.code if e.team_idx == 1 else None)),
                'text': e.text,
            }
            for e in res.events
        ],
    }


def save_selected_tournament(t: Tournament, out_dir: Path) -> int:
    """선정된 토너먼트의 104경기를 matches/<id>.json 으로 저장."""
    matches_dir = out_dir / 'matches'
    matches_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for mid in sorted(t.results.keys(), key=match_sort_key):
        res = t.results[mid]
        data = _match_dict(mid, t.stage_of(mid), res)
        path = matches_dir / f'{mid}.json'
        with path.open('w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        count += 1
    return count


def save_summary(mc_result: dict, out_dir: Path) -> Path:
    """집계 통계 + 선정 메타정보를 summary.json으로 저장."""
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries: list = mc_result['summaries']
    n = len(summaries)

    champ_counter = Counter(s.champion for s in summaries if s.champion)
    finalist_counter = Counter()
    semi_counter = Counter()
    qf_counter = Counter()
    for s in summaries:
        if s.champion:
            finalist_counter[s.champion] += 1
            semi_counter[s.champion] += 1
            qf_counter[s.champion] += 1
        if s.runner_up:
            finalist_counter[s.runner_up] += 1
            semi_counter[s.runner_up] += 1
            qf_counter[s.runner_up] += 1
        if s.third:
            semi_counter[s.third] += 1
            qf_counter[s.third] += 1

    avg_pk = sum(s.pk_count for s in summaries) / max(1, n)
    avg_upsets = sum(s.upset_count for s in summaries) / max(1, n)
    avg_goals = sum(s.total_goals for s in summaries) / max(1, n)

    sel: RunSummary = next(s for s in summaries if s.seed == mc_result['selected_seed'])

    summary = {
        'meta': {
            'n_runs': n,
            'base_seed': mc_result['base_seed'],
            'selected_seed': mc_result['selected_seed'],
            'selected_score': round(mc_result['selected_score'], 2),
            'strategy': mc_result.get('strategy', 'drama'),
        },
        'selected_run': {
            'champion': sel.champion,
            'runner_up': sel.runner_up,
            'third': sel.third,
            'pk_count': sel.pk_count,
            'upset_count': sel.upset_count,
            'total_goals': sel.total_goals,
        },
        'aggregates': {
            'avg_pk_per_tournament': round(avg_pk, 2),
            'avg_upsets_per_tournament': round(avg_upsets, 2),
            'avg_goals_per_tournament': round(avg_goals, 1),
        },
        'champion_pct':       _pct_dict(champ_counter, n),
        'finalist_pct':       _pct_dict(finalist_counter, n),
        'semifinalist_pct':   _pct_dict(semi_counter, n),
    }

    path = out_dir / 'summary.json'
    with path.open('w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return path


def _pct_dict(counter: Counter, n: int) -> dict:
    return {code: round(c / n * 100, 2) for code, c in counter.most_common()}
