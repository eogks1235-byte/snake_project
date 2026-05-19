"""라운드별 결과 jsonl 로그 + 알고리즘별 누적 집계.

매 라운드 종료 시 한 줄 append (recordings/territory/stats.jsonl).
시작 시 같은 파일을 읽어 등장한 알고리즘들의 누적 승률을 사이드바에 표시.
영상(mp4)을 지워도 이 파일은 별개라 통계가 보존된다.
"""
import json
from datetime import datetime
from pathlib import Path


def stats_path() -> Path:
    return (Path(__file__).resolve().parent.parent
            / 'recordings' / 'territory' / 'stats.jsonl')


def load_all() -> list:
    path = stats_path()
    if not path.exists():
        return []
    out = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def aggregate_by_name(rounds: list) -> dict:
    """알고리즘 이름 -> {games, wins, top3, sum_pct, deaths} 누적."""
    agg = {}
    for r in rounds:
        for entry in r.get('results', []):
            name = entry.get('name')
            if not name:
                continue
            d = agg.setdefault(name, {
                'games': 0, 'wins': 0, 'top3': 0, 'sum_pct': 0.0, 'deaths': 0
            })
            d['games'] += 1
            rk = entry.get('rank', 99)
            if rk == 1:
                d['wins'] += 1
            if rk <= 3:
                d['top3'] += 1
            if entry.get('dead'):
                d['deaths'] += 1
            d['sum_pct'] += float(entry.get('pct', 0))
    return agg


def append_round(round_data: dict):
    path = stats_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(round_data, ensure_ascii=False) + '\n')


def make_round_data(sim, agents, seed: int) -> dict:
    """현재 sim 상태에서 라운드 결과 dict 생성."""
    areas = sim.get_areas()
    total = sim.grid.total_cells()
    sorted_agents = sorted(agents, key=lambda a: -areas.get(a.id, 0))
    results = []
    for rank, agent in enumerate(sorted_agents, 1):
        pct = (100.0 * areas.get(agent.id, 0) / total) if total else 0
        results.append({
            'name': agent.name,
            'pct': round(pct, 1),
            'rank': rank,
            'team': sim.teams.get(agent.id) if hasattr(sim, 'teams') else None,
            'dead': sim.dead.get(agent.id, False) if hasattr(sim, 'dead') else False,
        })
    return {
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'seed': seed,
        'map': getattr(sim, 'map_name', 'open'),
        'mode': 'team' if getattr(sim, 'team_mode', False) else 'free-for-all',
        'n_players': len(agents),
        'duration_ticks': sim.tick,
        'results': results,
    }
