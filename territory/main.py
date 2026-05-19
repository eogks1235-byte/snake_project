"""실행 진입점 — 모든 게 매 판 랜덤.

매 라운드 (시작 시 + R 키):
  - 맵 프리셋 9종 중 1개 (open/cross/pillars/arena/diagonal/stripes/islands/donut/corridors)
  - 플레이어 수 (2~8 랜덤)
  - 알고리즘 풀 15개에서 N개 랜덤 픽
  - 각 플레이어 색상 HSV 기반 랜덤 (서로 충분히 다른 색)
  - 효과 풀 15개에서 4개 활성화
  - 시작 위치 랜덤
  - 아이템 4개 무작위 위치 (1회용)

조작:
  SPACE  : Fast forward
  R      : 새 라운드 (모든 게 다시 랜덤)
  V      : 녹화 토글
  ESC    : 종료

CLI:
  python territory/main.py                       # 모두 랜덤
  python territory/main.py --players 6           # 플레이어 수만 고정
  python territory/main.py --map arena           # 맵 고정
  python territory/main.py --record              # 자동 녹화
  python territory/main.py --seed 1234           # 시드 고정 (재현용)
"""
import sys
import random
import argparse
import colorsys
from pathlib import Path
from datetime import datetime

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from territory.agents import (ALGORITHM_POOL, BetrayerAgent,
                                    SaboteurAgent, RevenantAgent,
                                    TeamAttackerAgent, TeamSupporterAgent,
                                    TeamFlexAgent, TeamRaiderAgent)
    from territory.simulation import Simulation, BETRAYER_REVEAL_TICK
    from territory.render import Renderer
    from territory.recorder import VideoRecorder
    from territory.maps import MAP_PRESETS
    from territory.stats import (load_all, aggregate_by_name,
                                  append_round, make_round_data)
else:
    from .agents import (ALGORITHM_POOL, BetrayerAgent, SaboteurAgent,
                          RevenantAgent, TeamAttackerAgent,
                          TeamSupporterAgent, TeamFlexAgent, TeamRaiderAgent)
    from .simulation import Simulation, BETRAYER_REVEAL_TICK
    from .render import Renderer
    from .recorder import VideoRecorder
    from .maps import MAP_PRESETS
    from .stats import (load_all, aggregate_by_name,
                         append_round, make_round_data)

import pygame


GRID_SIZE = 60
CELL_SIZE = 16   # 16 → 960×960 grid → 1080×1920 window (YouTube Shorts 9:16 1080p)
FPS_NORMAL = 30
FPS_FAST = 240
RECORD_FPS = 30          # 영상 속도 = 시뮬 속도. 60→30으로 2배 천천히 (시청 편함)
STABLE_LIMIT = 180
HOLD_AFTER_DONE = 90
MIN_PLAYERS = 2
MAX_PLAYERS = 8
BLITZ_PROBABILITY = 0.25       # 25% 확률로 시간 제한 모드 (Blitz)
BLITZ_TIME_LIMIT = 900         # 30초 @ 30 FPS


def random_palette(n: int, rng: random.Random) -> list:
    """HSV 균등 분할 + 약간의 jitter — 서로 잘 구분되는 N색.

    파스텔 배경에서 가독성 좋은 mid-tone (너무 밝지도 어둡지도 않게).
    글리프 오버레이가 -95씩 빠진 톤으로 그려지므로 베이스가 너무 어두우면
    오버레이가 검게 뭉친다 → v 하한을 0.62로 유지.
    """
    base_hue = rng.random()
    out = []
    for i in range(n):
        h = (base_hue + i / n + rng.uniform(-0.025, 0.025)) % 1.0
        s = rng.uniform(0.55, 0.80)
        v = rng.uniform(0.62, 0.82)
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        out.append((int(r * 255), int(g * 255), int(b * 255)))
    rng.shuffle(out)
    return out


def team_palette(teams: list, rng: random.Random) -> list:
    """팀 단위로 색을 묶어서 같은 팀원은 비슷한 hue, 명도만 다르게.

    teams[i] = 에이전트 i 가 속한 팀 id (0 부터). 팀별로 hue 슬롯 배정.
    """
    unique_teams = sorted(set(teams))
    base_hue = rng.random()
    team_hue = {}
    for i, t in enumerate(unique_teams):
        team_hue[t] = (base_hue + i / max(1, len(unique_teams))
                       + rng.uniform(-0.02, 0.02)) % 1.0

    # 팀별로 멤버에게 명도/채도 jitter 부여
    team_count = {t: 0 for t in unique_teams}
    out = []
    for t in teams:
        h = team_hue[t]
        idx = team_count[t]
        team_count[t] += 1
        s = 0.65 + 0.08 * idx + rng.uniform(-0.05, 0.05)
        v = 0.72 - 0.10 * idx + rng.uniform(-0.04, 0.04)
        s = max(0.45, min(0.92, s))
        v = max(0.55, min(0.88, v))
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        out.append((int(r * 255), int(g * 255), int(b * 255)))
    return out


def decide_teams(n_players: int, rng: random.Random):
    """3명 이상이면 40% 확률로 팀전 모드. 2~3팀 비대칭 분할 허용.

    예: 6명 → 1v2v3, 2v2v2, 3v3, 1v5 등 무작위. 각 팀에 최소 1명 보장.
    반환: teams_list[i] = i번 에이전트의 team id (0 ..). None 이면 free-for-all.
    """
    if n_players < 3:
        return None
    if rng.random() > 0.40:
        return None

    max_teams = min(3, n_players)
    n_teams = rng.randint(2, max_teams)
    # 각 팀에 최소 1명 + 남은 인원은 무작위 팀에 배분 → 비대칭 분할
    teams = list(range(n_teams))
    teams += [rng.randint(0, n_teams - 1) for _ in range(n_players - n_teams)]
    rng.shuffle(teams)
    return teams


def build_simulation(seed: int, n_players: int = None,
                     map_preset: str = None) -> tuple[Simulation, list, int]:
    rng = random.Random(seed)
    if n_players is None:
        n_players = rng.randint(MIN_PLAYERS, MAX_PLAYERS)
    n_players = max(MIN_PLAYERS, min(MAX_PLAYERS, n_players))

    pool = list(ALGORITHM_POOL)
    if n_players <= len(pool):
        chosen = rng.sample(pool, k=n_players)
    else:
        chosen = rng.sample(pool, k=len(pool))
        chosen += [rng.choice(pool) for _ in range(n_players - len(pool))]

    # 팀전 모드 결정 (4/6명일 때 일부 시드만)
    team_assignment = decide_teams(n_players, rng)
    if team_assignment is not None:
        colors = team_palette(team_assignment, rng)
        teams = {i + 1: team_assignment[i] for i in range(n_players)}

        # 팀 모드 — 기존 캐릭터 그대로 유지하고 team_role 만 부여 (modifier 방식).
        # 30% 확률로 한 자리에 BetrayerAgent 가 끼어들어감.

        # Betrayer 투입 (역할 부여 전에 자리 고정)
        if n_players >= 4 and rng.random() < 0.30:
            traitor_idx = rng.randint(0, n_players - 1)
            chosen[traitor_idx] = BetrayerAgent
    else:
        colors = random_palette(n_players, rng)
        teams = None

    agents = [Cls(i + 1, color=c) for i, (Cls, c) in enumerate(zip(chosen, colors))]

    # 팀 역할 modifier — 비-Betrayer 멤버에 team_role 부여 (캐릭터 행동은 그대로)
    if team_assignment is not None:
        team_to_indices: dict = {}
        for i in range(n_players):
            team_to_indices.setdefault(team_assignment[i], []).append(i)
        for tid, idxs in team_to_indices.items():
            non_betrayer = [i for i in idxs
                            if not isinstance(agents[i], BetrayerAgent)]
            n_t = len(non_betrayer)
            if n_t == 0:
                continue
            if n_t == 1:
                roles = ['attacker']
            elif n_t == 2:
                roles = ['attacker', 'raider']
            elif n_t == 3:
                roles = ['attacker', 'supporter', 'raider']
            else:
                half = n_t // 2
                roles = (['attacker'] * half +
                         ['supporter'] * (n_t - half - 1) +
                         ['raider'])
            rng.shuffle(roles)
            for i, role in zip(non_betrayer, roles):
                agents[i].team_role = role

    # 배신자: 시작 시점엔 팀 hue + 팀 매핑 모두 원래 팀 그대로.
    # 20초 (= 600틱) 후 sim 측이 teams[id] 를 솔로 팀으로 변경 → 인클로저/사망
    # 룰이 그 시점부터 적대적으로 작동.
    solo_assignments = {}
    if teams is not None:
        solo_id = max(teams.values()) + 100
        for a in agents:
            if isinstance(a, BetrayerAgent):
                solo_assignments[a.id] = solo_id
                solo_id += 1

    sim = Simulation(agents, width=GRID_SIZE, height=GRID_SIZE,
                     seed=seed, randomize_start=True, map_preset=map_preset,
                     teams=teams,
                     team_mode=(team_assignment is not None))

    # Betrayer reveal 등록
    for aid, sid in solo_assignments.items():
        sim.betrayer_solo_team[aid] = sid
        sim.betrayer_reveal_at[aid] = BETRAYER_REVEAL_TICK

    # Blitz 모드 — 30초 시간 제한 (시드 25% 확률)
    if rng.random() < BLITZ_PROBABILITY:
        sim.time_limit = BLITZ_TIME_LIMIT

    # 누적 통계 attach — 등장 알고리즘들의 과거 승률을 사이드바에 표시
    stats_table = aggregate_by_name(load_all())
    sim.agent_stats = {a.id: stats_table.get(a.name, {}) for a in agents}

    return sim, agents, n_players


def make_recording_path() -> Path:
    out_dir = Path(__file__).resolve().parent.parent / 'recordings' / 'territory'
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return out_dir / f'territory_{stamp}.mp4'


def announce_round(seed, agents, sim, n_players):
    print(f'[ROUND] seed={seed}  players={n_players}  map={sim.map_name}')
    print(f'        algorithms: {", ".join(a.name for a in agents)}')
    print(f'        effects:    {", ".join(sim.active_effects)}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--players', type=int, default=None,
                        help=f'플레이어 수 고정 ({MIN_PLAYERS}~{MAX_PLAYERS}). 생략 시 매 판 랜덤')
    parser.add_argument('--record', action='store_true')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--map', dest='map_preset', type=str, default=None,
                        choices=[name for name, _ in MAP_PRESETS],
                        help='맵 고정. 생략 시 매 판 랜덤.')
    parser.add_argument('--layout', choices=['portrait', 'square'], default='portrait',
                        help='portrait(9:16, Shorts용, 기본) 또는 square(1:1)')
    args = parser.parse_args()

    seed = args.seed if args.seed is not None else random.randint(0, 1_000_000)
    sim, agents, n_players = build_simulation(seed, args.players, args.map_preset)
    renderer = Renderer(sim, agents, cell_size=CELL_SIZE, layout=args.layout)

    announce_round(seed, agents, sim, n_players)

    recorder: VideoRecorder | None = None
    if args.record:
        recorder = VideoRecorder(str(make_recording_path()), fps=RECORD_FPS)
        print(f'[REC] {recorder.output_path}')

    fast_forward = False
    running = True
    last_total = -1
    stable_count = 0
    done = False
    hold_frames = 0

    def reset_round():
        nonlocal sim, agents, last_total, stable_count, done, hold_frames, seed, n_players
        seed = random.randint(0, 1_000_000)
        sim, agents, n_players = build_simulation(seed, args.players, args.map_preset)
        # 플레이어 수가 매 라운드 바뀌므로 윈도우 크기 다시 계산
        renderer.__init__(sim, agents, cell_size=CELL_SIZE, layout=args.layout)
        last_total = -1
        stable_count = 0
        done = False
        hold_frames = 0
        announce_round(seed, agents, sim, n_players)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    fast_forward = not fast_forward
                elif event.key == pygame.K_r:
                    if recorder is not None:
                        recorder.close()
                        print(f'[REC] saved ({recorder.frame_count} frames, '
                              f'{recorder.duration_sec:.1f}s)')
                        recorder = VideoRecorder(str(make_recording_path()), fps=RECORD_FPS)
                        print(f'[REC] {recorder.output_path}')
                    reset_round()
                elif event.key == pygame.K_v:
                    if recorder is None:
                        recorder = VideoRecorder(str(make_recording_path()), fps=RECORD_FPS)
                        print(f'[REC] {recorder.output_path}')
                    else:
                        recorder.close()
                        print(f'[REC] saved ({recorder.frame_count} frames, '
                              f'{recorder.duration_sec:.1f}s)')
                        recorder = None
                elif event.key == pygame.K_ESCAPE:
                    running = False

        if not done:
            sim.step()
            cur_total = sum(sim.get_areas().values())
            if cur_total == last_total:
                stable_count += 1
            else:
                stable_count = 0
                last_total = cur_total
            time_up = (sim.time_limit is not None
                       and sim.tick >= sim.time_limit)
            if (cur_total >= sim.grid.total_cells()
                    or stable_count >= STABLE_LIMIT
                    or time_up):
                done = True
                hold_frames = HOLD_AFTER_DONE
                # winner 정보 sim에 attach (영토 1위) → render highlight 용
                areas_final = sim.get_areas()
                winner = max(agents, key=lambda a: areas_final.get(a.id, 0))
                sim.winner_id = winner.id
                sim.winner_team = sim.teams.get(winner.id) if sim.team_mode else None
                # 라운드 결과 jsonl 한 줄 append (한 번만)
                append_round(make_round_data(sim, agents, seed))

        renderer.draw(fast_forward=fast_forward)

        if recorder is not None:
            recorder.capture(renderer.screen)

        if done and recorder is not None:
            hold_frames -= 1
            if hold_frames <= 0:
                running = False

        renderer.tick_clock(FPS_FAST if fast_forward else FPS_NORMAL)

    if recorder is not None:
        recorder.close()
        print(f'[REC] saved {recorder.output_path} '
              f'({recorder.frame_count} frames, {recorder.duration_sec:.1f}s)')

    renderer.quit()
    sys.exit(0)


if __name__ == '__main__':
    main()
