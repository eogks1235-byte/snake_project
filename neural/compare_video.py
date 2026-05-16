"""진화 vs PPO 나란히 비교 영상 — mp4 출력.

실행:
  # 빠른 확인 (창 띄움, 종료 시 mp4 저장)
  python -m neural.compare_video --seed 0 --ticks 2000

  # 헤드리스 mp4만 (창 없음, 빠름)
  python -m neural.compare_video --seed 0 --ticks 2000 --headless

같은 seed의 동일한 World 두 개를 띄움:
  왼쪽: 진화 (numpy brain, 다양한 색)
  오른쪽: PPO 학습 정책 (모두 같은 정책, 동일 색)

상단에 실시간 통계 (pop, kills, avg_eaten).
"""
import sys
import argparse
import math
from pathlib import Path
from datetime import datetime
from typing import Optional

import numpy as np
import pygame

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from neural.world import (World, WORLD_W, WORLD_H, MAX_ENERGY, SIZE, VISION,
                              PREDATOR_SIZE, COMMON_FOOD_VALUE, PHEROMONE_LIFE)
    from neural.recorder import VideoRecorder
    from neural.torch_brain import TorchBrain
    from neural.ppo_world import policy_path, load_policy, make_world
else:
    from .world import (World, WORLD_W, WORLD_H, MAX_ENERGY, SIZE, VISION,
                        PREDATOR_SIZE, COMMON_FOOD_VALUE, PHEROMONE_LIFE)
    from .recorder import VideoRecorder
    from .torch_brain import TorchBrain
    from .ppo_world import policy_path, load_policy, make_world


# 색 (main.py와 동일 톤)
BG = (15, 17, 21)
PANEL = (22, 25, 31)
WORLD_BG = (20, 24, 30)
FOOD_COLOR = (140, 200, 90)
RARE_FOOD = (240, 170, 80)
WALL_COLOR = (60, 64, 75)
WALL_EDGE = (100, 105, 120)
PRED_COLOR = (220, 70, 80)
PHEROMONE = (200, 140, 240)
TEXT = (235, 235, 240)
TEXT_DIM = (140, 145, 160)
ACCENT_EVO = (130, 200, 240)
ACCENT_PPO = (240, 170, 100)


def draw_world(screen, world: World, ox: int, oy: int):
    """주어진 좌표 (ox, oy)에 world 그림. 1000×600 영역."""
    pygame.draw.rect(screen, WORLD_BG, (ox, oy, WORLD_W, WORLD_H))

    for w in world.walls:
        rect = pygame.Rect(int(ox + w.x), int(oy + w.y),
                            int(w.w), int(w.h))
        pygame.draw.rect(screen, WALL_COLOR, rect, border_radius=3)
        pygame.draw.rect(screen, WALL_EDGE, rect, width=1, border_radius=3)

    for p in world.pheromones:
        alpha = max(40, int(160 * p.life / PHEROMONE_LIFE))
        surf = pygame.Surface((6, 6), pygame.SRCALPHA)
        pygame.draw.circle(surf, (*PHEROMONE, alpha), (3, 3), 3)
        screen.blit(surf, (int(ox + p.x - 3), int(oy + p.y - 3)))

    for f in world.foods:
        col = RARE_FOOD if f.value > COMMON_FOOD_VALUE + 0.1 else FOOD_COLOR
        r = 3 if col == RARE_FOOD else 2
        pygame.draw.circle(screen, col,
                           (int(ox + f.x), int(oy + f.y)), r)

    for pr in world.predators:
        px, py = int(ox + pr.x), int(oy + pr.y)
        ang = math.atan2(pr.vy, pr.vx)
        r = PREDATOR_SIZE + 2
        pts = [
            (px + math.cos(ang) * r, py + math.sin(ang) * r),
            (px + math.cos(ang + 2.4) * r * 0.8,
             py + math.sin(ang + 2.4) * r * 0.8),
            (px + math.cos(ang - 2.4) * r * 0.8,
             py + math.sin(ang - 2.4) * r * 0.8),
        ]
        body = pr.color()
        pygame.draw.polygon(screen, body, pts)
        pygame.draw.polygon(screen, (60, 20, 25), pts, 1)

    for c in world.creatures:
        cx, cy = int(ox + c.x), int(oy + c.y)
        color = c.color()
        tail_x = cx - int(c.vx * 4)
        tail_y = cy - int(c.vy * 4)
        pygame.draw.line(screen,
                          (color[0] // 2, color[1] // 2, color[2] // 2),
                          (cx, cy), (tail_x, tail_y), 2)
        pygame.draw.circle(screen, color, (cx, cy), SIZE)
        bar_w = SIZE * 2
        ratio = max(0, min(1, c.energy / MAX_ENERGY))
        pygame.draw.rect(screen, (40, 44, 54),
                          (cx - bar_w // 2, cy - SIZE - 5, bar_w, 2))
        pygame.draw.rect(screen, (220, 220, 220),
                          (cx - bar_w // 2, cy - SIZE - 5,
                           int(bar_w * ratio), 2))


def record_compare_video(seed: int, ticks: int, policy,
                          headless: bool = True,
                          fps: int = 60,
                          out_path: Optional[Path] = None) -> Path:
    """진화 vs PPO 나란히 mp4 저장. 호출 가능 API."""
    world_evo = World(seed)
    world_ppo = make_world(seed, policy)
    print(f'[VIDEO] seed={seed}  evo_pop={world_evo.population()}  '
          f'ppo_pop={world_ppo.population()}')

    if headless:
        import os
        os.environ['SDL_VIDEODRIVER'] = 'dummy'

    MARGIN = 20
    HEADER = 90
    LABEL_H = 30
    win_w = WORLD_W * 2 + MARGIN * 3
    win_h = WORLD_H + HEADER + LABEL_H + MARGIN * 2
    pygame.init()
    screen = pygame.display.set_mode((win_w, win_h))
    pygame.display.set_caption('Compare: Evolution vs PPO')
    clock = pygame.time.Clock()
    font_title = pygame.font.SysFont('arial,malgungothic', 26, bold=True)
    font_label = pygame.font.SysFont('arial,malgungothic', 18, bold=True)
    font_stat = pygame.font.SysFont('arial,malgungothic', 14)

    if out_path is None:
        out_dir = Path(__file__).resolve().parent.parent / 'recordings' / 'neural'
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_path = out_dir / f'compare_{stamp}_seed{seed}.mp4'
    recorder = VideoRecorder(str(out_path), fps=fps)
    print(f'[REC] {out_path}')

    evo_x = MARGIN
    ppo_x = MARGIN * 2 + WORLD_W
    world_y = HEADER + LABEL_H

    running = True
    ticks_done = 0
    try:
        while running and ticks_done < ticks:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False

            if world_evo.population() > 0:
                world_evo.step()
            if world_ppo.population() > 0:
                world_ppo.step()
            ticks_done += 1

            screen.fill(BG)
            title = font_title.render(
                f'Evolution  vs  PPO   ·   seed {seed}   ·   tick {ticks_done}',
                True, TEXT)
            screen.blit(title, (MARGIN, 14))

            evo_eaten = (sum(c.lifetime_eaten for c in world_evo.creatures)
                         / max(1, world_evo.population()))
            ppo_eaten = (sum(c.lifetime_eaten for c in world_ppo.creatures)
                         / max(1, world_ppo.population()))
            screen.blit(font_label.render('EVOLUTION', True, ACCENT_EVO),
                         (evo_x, HEADER - 24))
            screen.blit(font_label.render('PPO', True, ACCENT_PPO),
                         (ppo_x, HEADER - 24))
            screen.blit(font_stat.render(
                f'pop {world_evo.population()}  ·  kills {world_evo.kills}  ·  '
                f'avg_eaten {evo_eaten:.1f}',
                True, TEXT_DIM), (evo_x + 130, HEADER - 20))
            screen.blit(font_stat.render(
                f'pop {world_ppo.population()}  ·  kills {world_ppo.kills}  ·  '
                f'avg_eaten {ppo_eaten:.1f}',
                True, TEXT_DIM), (ppo_x + 70, HEADER - 20))

            pad = 4
            pygame.draw.rect(screen, PANEL,
                              (evo_x - pad, world_y - pad,
                               WORLD_W + pad * 2, WORLD_H + pad * 2),
                              border_radius=8)
            pygame.draw.rect(screen, PANEL,
                              (ppo_x - pad, world_y - pad,
                               WORLD_W + pad * 2, WORLD_H + pad * 2),
                              border_radius=8)
            draw_world(screen, world_evo, evo_x, world_y)
            draw_world(screen, world_ppo, ppo_x, world_y)

            pygame.display.flip()
            recorder.capture(screen)

            if not headless:
                clock.tick(fps)

            if ticks_done % 200 == 0:
                print(f'  tick {ticks_done}/{ticks}  '
                      f'evo pop={world_evo.population()} '
                      f'ppo pop={world_ppo.population()}')
    finally:
        recorder.close()
        print(f'[REC] saved {recorder.frame_count} frames, '
              f'{recorder.duration_sec:.1f}s → {out_path}')
        pygame.quit()
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--ticks', type=int, default=2000)
    parser.add_argument('--fps', type=int, default=60)
    parser.add_argument('--headless', action='store_true',
                        help='창 없이 mp4만 생성 (훨씬 빠름)')
    parser.add_argument('--policy', type=str, default=None)
    args = parser.parse_args()

    p = Path(args.policy) if args.policy else policy_path()
    if not p.exists():
        print(f'no policy at {p} — 먼저 학습 필요')
        sys.exit(1)

    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    policy = load_policy(p, device)
    print(f'[POLICY] loaded {p}')

    record_compare_video(args.seed, args.ticks, policy,
                          headless=args.headless, fps=args.fps)


if __name__ == '__main__':
    main()
