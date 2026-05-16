"""L: Battle mode — 진화 brain과 PPO 정책을 한 World에 섞어 누가 살아남는지.

지금 ppo_world는 모든 prey가 같은 정책. 이건 절반 진화 / 절반 PPO 섞음.
색으로 구분 (진화: 다양한 색 / PPO: 흰 테두리 강조).

실행 (자동, mp4):
  python -m neural.battle --ticks 3000

  python -m neural.battle --ticks 3000 \\
      --evo-brain recordings/neural/neural_hof_0.npz \\
      --ppo-policy recordings/neural/ppo_world_policy.pt
"""
import sys
import math
import random
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional

import numpy as np
import pygame

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from neural.world import (World, INITIAL_POPULATION, WORLD_W, WORLD_H,
                              MAX_ENERGY, SIZE, PHEROMONE_LIFE,
                              COMMON_FOOD_VALUE, MAX_SPEED, METABOLISM, N_IN,
                              N_MEMORY, PPO_OBS_DIM, PREDATOR_SIZE,
                              Creature, Brain as NumpyBrain)
    from neural.brain import Brain, OUT_MEM_START
    from neural.recorder import VideoRecorder
    from neural.ppo_world import load_policy, policy_path
else:
    from .world import (World, INITIAL_POPULATION, WORLD_W, WORLD_H,
                        MAX_ENERGY, SIZE, PHEROMONE_LIFE, COMMON_FOOD_VALUE,
                        MAX_SPEED, METABOLISM, N_IN, N_MEMORY, PPO_OBS_DIM,
                        PREDATOR_SIZE, Creature)
    from .brain import Brain, OUT_MEM_START
    from .recorder import VideoRecorder
    from .ppo_world import load_policy, policy_path


# 색 (compare_video와 동일 톤)
BG = (15, 17, 21)
PANEL = (22, 25, 31)
WORLD_BG = (20, 24, 30)
FOOD = (140, 200, 90)
RARE_FOOD = (240, 170, 80)
WALL = (60, 64, 75)
WALL_EDGE = (100, 105, 120)
PRED = (220, 70, 80)
PHEROMONE_C = (200, 140, 240)
TEXT = (235, 235, 240)
DIM = (140, 145, 160)
EVO_RING = (130, 200, 240)
PPO_RING = (240, 170, 100)


class BattleWorld(World):
    """절반은 진화(개체별 numpy brain), 절반은 PPO(공유 정책)인 World."""
    def __init__(self, seed: int, ppo_policy, evo_brain=None):
        super().__init__(seed)
        self.ppo_policy = ppo_policy
        self.evo_brain_template = evo_brain  # None이면 random
        # 절반에 PPO 태그
        self.ppo_ids = set()
        n = len(self.creatures)
        for i, c in enumerate(self.creatures[:n // 2]):
            self.ppo_ids.add(c.creature_id)
            # ppo 개체는 별도 색 시그니처 흉내 (밝은 주황 톤)
        for i, c in enumerate(self.creatures[n // 2:]):
            if self.evo_brain_template is not None:
                c.brain = self.evo_brain_template.mutate(self.rng)

        self.disable_reproduction = True  # 단순화 — 죽으면 retired, 새 spawn은 같은 비율 유지
        self.ppo_buffer_max = None

    def _think_and_move_creatures(self):
        """Mixed: PPO 개체는 공유 정책, 진화 개체는 자기 brain."""
        import torch
        if not self.creatures:
            return
        # PPO 그룹과 진화 그룹 분리
        ppo_creatures = [c for c in self.creatures if c.creature_id in self.ppo_ids]
        evo_creatures = [c for c in self.creatures if c.creature_id not in self.ppo_ids]

        # 시야 한 번에 계산
        all_xy = np.array([(c.x, c.y) for c in self.creatures], dtype=np.float32)
        food_xy = np.array([(f.x, f.y) for f in self.foods],
                            dtype=np.float32) if self.foods \
                  else np.empty((0, 2), dtype=np.float32)
        food_value = np.array([f.value for f in self.foods],
                                dtype=np.float32) / 2.5 if self.foods \
                     else np.empty(0, dtype=np.float32)
        pred_xy = np.array([(p.x, p.y) for p in self.predators],
                            dtype=np.float32) if self.predators \
                  else np.empty((0, 2), dtype=np.float32)
        if self._wall_sense_xy is None:
            self._refresh_wall_cache()
        danger_xy = np.concatenate([pred_xy, self._wall_sense_xy], axis=0) \
                    if pred_xy.shape[0] + self._wall_sense_xy.shape[0] > 0 \
                    else np.empty((0, 2), dtype=np.float32)
        ph_xy = np.array([(p.x, p.y) for p in self.pheromones],
                          dtype=np.float32) if self.pheromones \
                else np.empty((0, 2), dtype=np.float32)
        ph_int = np.array([p.life / PHEROMONE_LIFE for p in self.pheromones],
                            dtype=np.float32) if self.pheromones \
                 else np.empty(0, dtype=np.float32)

        from .world import VISION
        food_all = self._all_sectors(all_xy, food_xy, VISION, intensity=food_value)
        pred_all = self._all_sectors(all_xy, danger_xy, VISION)
        kin_all = self._all_sectors(all_xy, all_xy, VISION, exclude_self=True)
        phero_all = self._all_sectors(all_xy, ph_xy, VISION, intensity=ph_int)

        # PPO 일괄 forward
        if ppo_creatures:
            ppo_idxs = [i for i, c in enumerate(self.creatures)
                        if c.creature_id in self.ppo_ids]
            ppo_obs = np.zeros((len(ppo_idxs), PPO_OBS_DIM), dtype=np.float32)
            ppo_obs[:, 0:8] = food_all[ppo_idxs]
            ppo_obs[:, 8:16] = pred_all[ppo_idxs]
            ppo_obs[:, 16:24] = kin_all[ppo_idxs]
            ppo_obs[:, 24:32] = phero_all[ppo_idxs]
            for k, idx in enumerate(ppo_idxs):
                c = self.creatures[idx]
                ppo_obs[k, 32] = c.energy / MAX_ENERGY
                ppo_obs[k, 33] = c.vx / MAX_SPEED
                ppo_obs[k, 34] = c.vy / MAX_SPEED
            ppo_obs[:, -1] = 1.0
            with torch.no_grad():
                actions, _, _, _ = self.ppo_policy.act(torch.from_numpy(ppo_obs))
            ppo_act = actions.cpu().numpy()
            for k, idx in enumerate(ppo_idxs):
                c = self.creatures[idx]
                c.vx = float(ppo_act[k, 0]) * MAX_SPEED
                c.vy = float(ppo_act[k, 1]) * MAX_SPEED

        # 진화 brain은 개체별
        for i, c in enumerate(self.creatures):
            if c.creature_id in self.ppo_ids:
                continue
            inp = np.empty(N_IN, dtype=np.float32)
            inp[0:8] = food_all[i]
            inp[8:16] = pred_all[i]
            inp[16:24] = kin_all[i]
            inp[24:32] = phero_all[i]
            inp[32] = c.energy / MAX_ENERGY
            inp[33] = c.vx / MAX_SPEED
            inp[34] = c.vy / MAX_SPEED
            inp[35:35 + N_MEMORY] = c.memory
            inp[-1] = 1.0
            out = c.brain.forward(inp)
            c.vx = float(out[0]) * MAX_SPEED
            c.vy = float(out[1]) * MAX_SPEED
            c.memory = out[OUT_MEM_START:OUT_MEM_START + N_MEMORY].astype(np.float32).copy()

        # 이동 + 에너지
        for c in self.creatures:
            if not c.alive:
                continue
            self._apply_move_with_wall_slide(c, c.vx, c.vy)
            c.energy -= METABOLISM
            c.age += 1
        self._consume_food_batched()

    def _maintain_population(self):
        """대체 spawn 시 같은 진영(PPO/진화)으로 유지."""
        target = INITIAL_POPULATION
        # 진영별 부족분 계산
        ppo_alive = sum(1 for c in self.creatures if c.creature_id in self.ppo_ids)
        evo_alive = len(self.creatures) - ppo_alive
        ppo_target = target // 2
        evo_target = target - ppo_target
        while ppo_alive < ppo_target:
            self._spawn_battle_creature(is_ppo=True)
            ppo_alive += 1
        while evo_alive < evo_target:
            self._spawn_battle_creature(is_ppo=False)
            evo_alive += 1

    def _spawn_battle_creature(self, is_ppo: bool):
        for _ in range(10):
            x = self.rng.uniform(0, WORLD_W)
            y = self.rng.uniform(0, WORLD_H)
            if not self._in_any_wall(x, y):
                break
        cid = self._new_id()
        if is_ppo:
            dummy = Brain.random(self.rng)
        elif self.evo_brain_template is not None:
            dummy = self.evo_brain_template.mutate(self.rng)
        else:
            dummy = Brain.random(self.rng)
        from .world import INITIAL_ENERGY
        c = Creature(
            creature_id=cid,
            x=x, y=y,
            vx=self.rng.uniform(-1, 1),
            vy=self.rng.uniform(-1, 1),
            energy=INITIAL_ENERGY,
            age=0,
            brain=dummy,
            generation=1,
            parent_ids=(),
            ancestor_id=cid,
        )
        if is_ppo:
            self.ppo_ids.add(cid)
        self.creatures.append(c)
        self._record_birth(c)


def draw_world(screen, world: BattleWorld, ox: int, oy: int, fonts):
    pygame.draw.rect(screen, WORLD_BG, (ox, oy, WORLD_W, WORLD_H))
    for w in world.walls:
        rect = pygame.Rect(int(ox + w.x), int(oy + w.y), int(w.w), int(w.h))
        pygame.draw.rect(screen, WALL, rect, border_radius=3)
        pygame.draw.rect(screen, WALL_EDGE, rect, width=1, border_radius=3)
    for p in world.pheromones:
        a = max(40, int(160 * p.life / PHEROMONE_LIFE))
        s = pygame.Surface((6, 6), pygame.SRCALPHA)
        pygame.draw.circle(s, (*PHEROMONE_C, a), (3, 3), 3)
        screen.blit(s, (int(ox + p.x - 3), int(oy + p.y - 3)))
    for f in world.foods:
        col = RARE_FOOD if f.value > COMMON_FOOD_VALUE + 0.1 else FOOD
        pygame.draw.circle(screen, col,
                            (int(ox + f.x), int(oy + f.y)),
                            3 if col == RARE_FOOD else 2)
    for pr in world.predators:
        px, py = int(ox + pr.x), int(oy + pr.y)
        ang = math.atan2(pr.vy, pr.vx)
        r = PREDATOR_SIZE + 2
        pts = [
            (px + math.cos(ang) * r, py + math.sin(ang) * r),
            (px + math.cos(ang + 2.4) * r * 0.8, py + math.sin(ang + 2.4) * r * 0.8),
            (px + math.cos(ang - 2.4) * r * 0.8, py + math.sin(ang - 2.4) * r * 0.8),
        ]
        pygame.draw.polygon(screen, pr.color(), pts)
        pygame.draw.polygon(screen, (60, 20, 25), pts, 1)
    for c in world.creatures:
        cx, cy = int(ox + c.x), int(oy + c.y)
        color = c.color()
        ring = PPO_RING if c.creature_id in world.ppo_ids else EVO_RING
        tail_x = cx - int(c.vx * 4)
        tail_y = cy - int(c.vy * 4)
        pygame.draw.line(screen, (color[0]//2, color[1]//2, color[2]//2),
                          (cx, cy), (tail_x, tail_y), 2)
        pygame.draw.circle(screen, color, (cx, cy), SIZE)
        pygame.draw.circle(screen, ring, (cx, cy), SIZE + 2, 1)
        bar_w = SIZE * 2
        ratio = max(0, min(1, c.energy / MAX_ENERGY))
        pygame.draw.rect(screen, (40, 44, 54),
                          (cx - bar_w // 2, cy - SIZE - 5, bar_w, 2))
        pygame.draw.rect(screen, (220, 220, 220),
                          (cx - bar_w // 2, cy - SIZE - 5,
                           int(bar_w * ratio), 2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--ticks', type=int, default=3000)
    parser.add_argument('--fps', type=int, default=60)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--ppo-policy', type=str, default=None)
    parser.add_argument('--evo-brain', type=str, default=None,
                        help='진화 brain .npz (없으면 random)')
    args = parser.parse_args()

    pol_p = Path(args.ppo_policy) if args.ppo_policy else policy_path()
    if not pol_p.exists():
        print(f'no policy at {pol_p}'); sys.exit(1)
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    policy = load_policy(pol_p, device)

    evo_brain = None
    if args.evo_brain:
        evo_brain, meta = Brain.load(Path(args.evo_brain))
        print(f'[EVO] loaded gen={meta.get("generation","?")}')

    world = BattleWorld(args.seed, policy, evo_brain=evo_brain)
    if args.headless:
        import os
        os.environ['SDL_VIDEODRIVER'] = 'dummy'

    MARGIN = 20
    HEADER = 60
    win_w = WORLD_W + MARGIN * 2
    win_h = WORLD_H + HEADER + MARGIN * 2
    pygame.init()
    screen = pygame.display.set_mode((win_w, win_h))
    pygame.display.set_caption('Battle: Evolution vs PPO')
    clock = pygame.time.Clock()
    font_title = pygame.font.SysFont('arial,malgungothic', 22, bold=True)
    font_stat = pygame.font.SysFont('arial,malgungothic', 14)
    fonts = (font_title, font_stat)

    out_dir = Path(__file__).resolve().parent.parent / 'recordings' / 'neural'
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_mp4 = out_dir / f'battle_{stamp}_seed{args.seed}.mp4'
    recorder = VideoRecorder(str(out_mp4), fps=args.fps)
    print(f'[REC] {out_mp4}')

    try:
        for t in range(args.ticks):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise SystemExit
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    raise SystemExit
            world.step()
            world._maintain_population()

            ppo_alive = sum(1 for c in world.creatures if c.creature_id in world.ppo_ids)
            evo_alive = len(world.creatures) - ppo_alive
            ppo_eaten = sum(c.lifetime_eaten for c in world.creatures
                            if c.creature_id in world.ppo_ids)
            evo_eaten = sum(c.lifetime_eaten for c in world.creatures
                            if c.creature_id not in world.ppo_ids)

            screen.fill(BG)
            title = font_title.render(
                f'BATTLE   seed {args.seed}   tick {t+1}', True, TEXT)
            screen.blit(title, (MARGIN, 10))
            evo_st = font_stat.render(
                f'EVO  alive={evo_alive:3d}  eaten={evo_eaten:4d}',
                True, EVO_RING)
            ppo_st = font_stat.render(
                f'PPO  alive={ppo_alive:3d}  eaten={ppo_eaten:4d}',
                True, PPO_RING)
            screen.blit(evo_st, (MARGIN, 38))
            screen.blit(ppo_st, (MARGIN + 280, 38))

            pad = 4
            pygame.draw.rect(screen, PANEL,
                              (MARGIN - pad, HEADER + MARGIN - pad,
                               WORLD_W + pad * 2, WORLD_H + pad * 2),
                              border_radius=8)
            draw_world(screen, world, MARGIN, HEADER + MARGIN, fonts)
            pygame.display.flip()
            recorder.capture(screen)
            if not args.headless:
                clock.tick(args.fps)
            if (t + 1) % 300 == 0:
                print(f'  tick {t+1}/{args.ticks}  evo={evo_alive} '
                      f'ppo={ppo_alive}  evo_eaten={evo_eaten} ppo_eaten={ppo_eaten}')
    except SystemExit:
        pass
    finally:
        recorder.close()
        ppo_alive = sum(1 for c in world.creatures if c.creature_id in world.ppo_ids)
        evo_alive = len(world.creatures) - ppo_alive
        ppo_eaten = sum(c.lifetime_eaten for c in world.creatures
                        if c.creature_id in world.ppo_ids)
        evo_eaten = sum(c.lifetime_eaten for c in world.creatures
                        if c.creature_id not in world.ppo_ids)
        print(f'\n[FINAL] EVO alive={evo_alive} eaten={evo_eaten}  '
              f'PPO alive={ppo_alive} eaten={ppo_eaten}')
        winner = 'PPO' if ppo_eaten > evo_eaten else 'EVO'
        print(f'[WINNER] {winner} (먹이 누적 기준)')
        print(f'[REC] saved {recorder.frame_count} frames → {out_mp4}')
        pygame.quit()


if __name__ == '__main__':
    main()
