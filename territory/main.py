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
    from territory.agents import ALGORITHM_POOL
    from territory.simulation import Simulation
    from territory.render import Renderer
    from territory.recorder import VideoRecorder
    from territory.maps import MAP_PRESETS
else:
    from .agents import ALGORITHM_POOL
    from .simulation import Simulation
    from .render import Renderer
    from .recorder import VideoRecorder
    from .maps import MAP_PRESETS

import pygame


GRID_SIZE = 60
CELL_SIZE = 9
FPS_NORMAL = 30
FPS_FAST = 240
RECORD_FPS = 60
STABLE_LIMIT = 180
HOLD_AFTER_DONE = 90
MIN_PLAYERS = 2
MAX_PLAYERS = 8


def random_palette(n: int, rng: random.Random) -> list:
    """HSV 균등 분할 + 약간의 jitter — 서로 잘 구분되는 N색."""
    base_hue = rng.random()
    out = []
    for i in range(n):
        h = (base_hue + i / n + rng.uniform(-0.025, 0.025)) % 1.0
        s = rng.uniform(0.65, 0.95)
        v = rng.uniform(0.78, 0.98)
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        out.append((int(r * 255), int(g * 255), int(b * 255)))
    rng.shuffle(out)
    return out


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

    colors = random_palette(n_players, rng)
    agents = [Cls(i + 1, color=c) for i, (Cls, c) in enumerate(zip(chosen, colors))]
    sim = Simulation(agents, width=GRID_SIZE, height=GRID_SIZE,
                     seed=seed, randomize_start=True, map_preset=map_preset)
    return sim, agents, n_players


def make_recording_path() -> Path:
    out_dir = Path(__file__).resolve().parent.parent / 'recordings'
    out_dir.mkdir(exist_ok=True)
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
    args = parser.parse_args()

    seed = args.seed if args.seed is not None else random.randint(0, 1_000_000)
    sim, agents, n_players = build_simulation(seed, args.players, args.map_preset)
    renderer = Renderer(sim, agents, cell_size=CELL_SIZE)

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
        renderer.__init__(sim, agents, cell_size=CELL_SIZE)
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
            if cur_total >= sim.grid.total_cells() or stable_count >= STABLE_LIMIT:
                done = True
                hold_frames = HOLD_AFTER_DONE

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
