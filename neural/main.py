"""실행 진입점 — 뉴럴넷 기반 진화 시뮬 (영상 녹화 중심, lean version).

조작:
  SPACE  : Fast forward
  R      : 새 시드로 재시작
  ESC    : 종료

영상 녹화:
  python -m neural.main --record                          # 진화, 자동 mp4
  python -m neural.main --ppo-policy <pt> --record        # PPO 정책

자동저장 (학습 곡선 기록용): 50초마다 HOF 슬롯 갱신.

평가 모드:
  python -m neural.main --evaluate path/to/brain.npz
"""
import sys
import time
import random
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Tuple

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from neural.world import World, INITIAL_POPULATION
    from neural.render import Renderer
    from neural.recorder import VideoRecorder
    from neural.brain import Brain
else:
    from .world import World, INITIAL_POPULATION
    from .render import Renderer
    from .recorder import VideoRecorder
    from .brain import Brain

import pygame


FPS_NORMAL = 60
FPS_FAST = 1000
FAST_RENDER_EVERY = 4   # SPACE 중에는 N틱마다 1번만 그림
RECORD_FPS = 60
AUTOSAVE_SEC = 50
HOF_SIZE = 5
EVAL_TICKS = 1500


def out_dir() -> Path:
    d = Path(__file__).resolve().parent.parent / 'recordings' / 'neural'
    d.mkdir(parents=True, exist_ok=True)
    return d


def make_recording_path() -> Path:
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return out_dir() / f'neural_{stamp}.mp4'


def champion_path() -> Path:
    return out_dir() / 'neural_champion.npz'


def hof_path(i: int) -> Path:
    return out_dir() / f'neural_hof_{i}.npz'


def snapshot_path() -> Path:
    return out_dir() / 'neural_snapshot.pkl'


def csv_path() -> Path:
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return out_dir() / f'neural_history_{stamp}.csv'


def load_hof_slots() -> List[Tuple[int, Optional[dict]]]:
    slots = []
    for i in range(HOF_SIZE):
        p = hof_path(i)
        if not p.exists():
            slots.append((i, None))
            continue
        try:
            _, meta = Brain.load(p)
            slots.append((i, meta))
        except Exception:
            slots.append((i, None))
    return slots


def slot_score(meta: Optional[dict]) -> Tuple[int, int]:
    if meta is None:
        return (-1, -1)
    return (int(meta.get('generation', 0)),
            int(meta.get('lifetime_eaten', 0)))


def hof_consider(brain: Brain, meta: dict) -> Optional[int]:
    slots = load_hof_slots()
    cur = slot_score(meta)
    weakest_idx = min(slots, key=lambda s: slot_score(s[1]))[0]
    weakest_score = slot_score(slots[weakest_idx][1])
    if cur > weakest_score:
        brain.save(hof_path(weakest_idx), meta)
        return weakest_idx
    return None


def hof_pick_random(rng: random.Random) -> Optional[Tuple[Brain, dict, int]]:
    available = [(i, m) for i, m in load_hof_slots() if m is not None]
    if not available:
        return None
    i, _ = rng.choice(available)
    brain, meta = Brain.load(hof_path(i))
    return brain, meta, i


def evaluate(brain_path: Path, ticks: int = EVAL_TICKS, seed: int = 0) -> dict:
    print(f'[EVAL] loading {brain_path}')
    brain, meta = Brain.load(brain_path)
    print(f'[EVAL] meta={meta}')
    rng = random.Random(seed)
    clones = [brain.mutate(rng) for _ in range(INITIAL_POPULATION)]
    world = World(seed, initial_brains=clones)

    for _ in range(ticks):
        if world.population() == 0:
            break
        world.step()

    pop = world.population()
    survival_rate = pop / INITIAL_POPULATION
    if pop > 0:
        avg_eaten = sum(c.lifetime_eaten for c in world.creatures) / pop
        avg_age = sum(c.age for c in world.creatures) / pop
        max_gen = max(c.generation for c in world.creatures)
    else:
        avg_eaten = avg_age = max_gen = 0

    result = {
        'ticks_run': world.tick,
        'final_pop': pop,
        'survival_rate': survival_rate,
        'avg_eaten': avg_eaten,
        'avg_age': avg_age,
        'max_gen': max_gen,
        'kills': world.kills,
        'births': world.births,
        'sexual_births': world.sexual_births,
        'diversity_at_end': world._last_diversity,
    }
    print('\n[EVAL] ====== result ======')
    for k, v in result.items():
        print(f'  {k:18}: {v}')
    return result


def interactive(args):
    if args.headless:
        import os
        os.environ['SDL_VIDEODRIVER'] = 'dummy'
    seed = args.seed if args.seed is not None else random.randint(0, 1_000_000)
    world = World(seed,
                   prey_hidden=64 if args.big_brain else 24,
                   pred_hidden=48 if args.big_brain else 16,
                   hebbian=args.learn)

    ppo_policy = None
    if args.ppo_policy:
        import torch
        from neural.torch_brain import TorchBrain
        ckpt = torch.load(args.ppo_policy, map_location='cpu', weights_only=False)
        cfg = ckpt['config']
        ppo_policy = TorchBrain(cfg['obs_dim'], cfg['hidden'], cfg['act_dim'],
                                 arch='mlp')
        ppo_policy.load_state_dict(ckpt['state_dict'])
        ppo_policy.eval()
        world.shared_policy = ppo_policy
        world.disable_reproduction = True
        print(f'[PPO] loaded {args.ppo_policy}  (all prey controlled by policy)')

    renderer = Renderer()
    print(f'[NEURAL] seed={seed}  pop={world.population()}  '
          f'learn={args.learn}  big_brain={args.big_brain}  '
          f'ppo={"on" if ppo_policy else "off"}')

    if args.load_champion and champion_path().exists():
        try:
            brain, meta = Brain.load(champion_path())
            world.inject_brain(brain)
            print(f'[CHAMP] loaded → injected  meta={meta}')
        except ValueError as e:
            print(f'[CHAMP] skip: {e}')

    recorder = None
    if args.record:
        recorder = VideoRecorder(str(make_recording_path()), fps=RECORD_FPS)
        print(f'[REC] {recorder.output_path}')

    # auto-exit + headless 모드: fast_forward 끔 (1 tick = 1 frame, 1분 = 3600틱)
    fast_forward = False
    running = True
    last_autosave = time.time()

    def reset():
        nonlocal seed, world
        seed = random.randint(0, 1_000_000)
        world = World(seed,
                       prey_hidden=64 if args.big_brain else 24,
                       pred_hidden=48 if args.big_brain else 16,
                       hebbian=args.learn)
        if ppo_policy is not None:
            world.shared_policy = ppo_policy
            world.disable_reproduction = True
        renderer.selected_id = None
        renderer.selected_id_b = None
        print(f'[NEURAL] seed={seed}  pop={world.population()}')

    def autosave_hof():
        """자동 HOF 갱신 (학습 곡선 기록용, 사용자 키는 없음)."""
        champ = world.champion()
        if champ is None:
            return
        meta = {
            'generation': champ.generation,
            'lifetime_eaten': champ.lifetime_eaten,
            'seed': world.seed,
        }
        slot = hof_consider(champ.brain, meta)
        if slot is not None:
            print(f'[AUTOSAVE] HOF slot{slot} ← gen={champ.generation} '
                  f'eaten={champ.lifetime_eaten}')

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
                        recorder = VideoRecorder(str(make_recording_path()),
                                                 fps=RECORD_FPS)
                    reset()
                elif event.key == pygame.K_ESCAPE:
                    running = False

        if world.population() == 0:
            renderer.draw(world, fast_forward)
            pygame.time.wait(1500)
            reset()
            continue

        world.step()

        if (args.auto_exit_ticks is not None
                and world.tick >= args.auto_exit_ticks):
            running = False

        if time.time() - last_autosave >= AUTOSAVE_SEC:
            autosave_hof()
            last_autosave = time.time()

        do_draw = (not fast_forward) or (world.tick % FAST_RENDER_EVERY == 0)
        if do_draw:
            renderer.draw(world, fast_forward)
            if recorder is not None:
                recorder.capture(renderer.screen)

        renderer.tick_clock(FPS_FAST if fast_forward else FPS_NORMAL)

    if recorder is not None:
        recorder.close()
        print(f'[REC] saved {recorder.output_path} '
              f'({recorder.frame_count} frames, {recorder.duration_sec:.1f}s)')

    renderer.quit()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--record', action='store_true')
    parser.add_argument('--load-champion', action='store_true')
    parser.add_argument('--evaluate', type=str, default=None)
    parser.add_argument('--ticks', type=int, default=EVAL_TICKS)
    parser.add_argument('--learn', action='store_true',
                        help='개체별 in-lifetime Hebbian 학습 활성화')
    parser.add_argument('--big-brain', action='store_true',
                        help='prey hidden 24→64, predator 16→48')
    parser.add_argument('--ppo-policy', type=str, default=None,
                        help='학습된 PPO 정책(.pt) 로드 — 모든 prey가 이 정책으로 행동')
    parser.add_argument('--auto-exit-ticks', type=int, default=None,
                        help='이 틱 수 도달 시 자동 종료 (자동 영상 만들기용, 60fps 1분 = 3600)')
    parser.add_argument('--headless', action='store_true',
                        help='창 없이 실행 (자동 영상 생성 시 권장)')
    args = parser.parse_args()

    if args.evaluate is not None:
        evaluate(Path(args.evaluate),
                 ticks=args.ticks,
                 seed=args.seed if args.seed is not None else 0)
        sys.exit(0)

    interactive(args)
    sys.exit(0)


if __name__ == '__main__':
    main()
