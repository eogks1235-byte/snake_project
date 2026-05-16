"""실행 진입점 — 식물/초식/육식 3계층 생태계 시뮬.

조작:
  SPACE  : Fast forward
  R      : 새 시드로 재시작
  V      : 녹화 토글
  ESC    : 종료
"""
import sys
import random
import argparse
from pathlib import Path
from datetime import datetime

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from ecosystem.world import World
    from ecosystem.render import Renderer
    from ecosystem.recorder import VideoRecorder
else:
    from .world import World
    from .render import Renderer
    from .recorder import VideoRecorder

import pygame


FPS_NORMAL = 30
FPS_FAST = 240
RECORD_FPS = 60


def make_recording_path() -> Path:
    out_dir = Path(__file__).resolve().parent.parent / 'recordings'
    out_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return out_dir / f'ecosystem_{stamp}.mp4'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--record', action='store_true')
    args = parser.parse_args()

    seed = args.seed if args.seed is not None else random.randint(0, 1_000_000)
    world = World(seed)
    renderer = Renderer()
    pops = world.populations()
    print(f'[ECOSYSTEM] seed={seed}  '
          f'plants={pops["plants"]} herb={pops["herb"]} carn={pops["carn"]}')

    recorder = None
    if args.record:
        recorder = VideoRecorder(str(make_recording_path()), fps=RECORD_FPS)
        print(f'[REC] {recorder.output_path}')

    fast_forward = False
    running = True

    def reset():
        nonlocal seed, world
        seed = random.randint(0, 1_000_000)
        world = World(seed)
        pops = world.populations()
        print(f'[ECOSYSTEM] seed={seed}  '
              f'plants={pops["plants"]} herb={pops["herb"]} carn={pops["carn"]}')

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
                        recorder = VideoRecorder(str(make_recording_path()),
                                                 fps=RECORD_FPS)
                        print(f'[REC] {recorder.output_path}')
                    reset()
                elif event.key == pygame.K_v:
                    if recorder is None:
                        recorder = VideoRecorder(str(make_recording_path()),
                                                 fps=RECORD_FPS)
                        print(f'[REC] {recorder.output_path}')
                    else:
                        recorder.close()
                        print(f'[REC] saved ({recorder.frame_count} frames, '
                              f'{recorder.duration_sec:.1f}s)')
                        recorder = None
                elif event.key == pygame.K_ESCAPE:
                    running = False

        # 동물이 모두 멸종 시 잠시 대기 후 자동 리셋
        if world.is_extinct():
            renderer.draw(world, fast_forward)
            pygame.time.wait(1500)
            reset()
            continue

        world.step()
        renderer.draw(world, fast_forward)

        if recorder is not None:
            recorder.capture(renderer.screen)

        renderer.tick_clock(FPS_FAST if fast_forward else FPS_NORMAL)

    if recorder is not None:
        recorder.close()
        print(f'[REC] saved {recorder.output_path} '
              f'({recorder.frame_count} frames, {recorder.duration_sec:.1f}s)')

    renderer.quit()
    sys.exit(0)


if __name__ == '__main__':
    main()
