"""뉴럴 진화 시각화 — 4채널 시야 · 페로몬 · 다양성 그래프 · 비교 · 벽 · 음식종 · 스택차트."""
import math
from typing import Optional, List

import numpy as np
import pygame

from .world import (World, Creature, WORLD_W, WORLD_H, MAX_ENERGY, MAX_SPEED,
                    SIZE, VISION, PREDATOR_SIZE, PREDATOR_VISION,
                    PHEROMONE_LIFE, COMMON_FOOD_VALUE, RARE_FOOD_VALUE)
from .brain import (Brain, N_IN, N_HIDDEN, N_OUT, N_SECTORS, N_MEMORY)


BG_COLOR = (15, 17, 21)
PANEL_COLOR = (22, 25, 31)
WORLD_BG = (20, 24, 30)
FOOD_COLOR = (140, 200, 90)
RARE_FOOD_COLOR = (240, 170, 80)
WALL_COLOR = (60, 64, 75)
WALL_EDGE = (100, 105, 120)
PREDATOR_COLOR = (220, 70, 80)
PHEROMONE_COLOR = (200, 140, 240)
KIN_COLOR = (110, 200, 240)
TEXT_PRIMARY = (235, 235, 240)
TEXT_SECONDARY = (140, 145, 160)
TEXT_DIM = (90, 95, 110)
ACCENT_GOLD = (255, 215, 90)
ACCENT_BLUE = (120, 180, 250)
ACCENT_PINK = (240, 130, 200)
ACCENT_RED = (240, 100, 110)
SELECT_COLOR = (255, 255, 255)
SELECT_B_COLOR = (255, 180, 220)

ANCESTOR_PALETTE = [
    (250, 180, 90),   # 주황
    (110, 200, 240),  # 하늘
    (180, 140, 240),  # 보라
    (140, 220, 130),  # 연두
    (240, 130, 200),  # 분홍
    (240, 220, 100),  # 노랑
]


class Renderer:
    HEADER_H = 96
    SIDEBAR_W = 380
    MARGIN = 24

    def __init__(self):
        pygame.init()
        self.window_w = WORLD_W + self.SIDEBAR_W + self.MARGIN * 3
        self.window_h = WORLD_H + self.HEADER_H + self.MARGIN * 2
        self.screen = pygame.display.set_mode((self.window_w, self.window_h))
        pygame.display.set_caption('Evolution — Neural Network')
        self.clock = pygame.time.Clock()

        self.font_title = pygame.font.SysFont('arial,malgungothic', 28, bold=True)
        self.font_sub = pygame.font.SysFont('arial,malgungothic', 13)
        self.font_stat = pygame.font.SysFont('arial,malgungothic', 13, bold=True)
        self.font_pin = pygame.font.SysFont('arial,malgungothic', 12, bold=True)
        self.font_label = pygame.font.SysFont('arial,malgungothic', 11)
        self.font_tiny = pygame.font.SysFont('arial,malgungothic', 10)
        self.font_big = pygame.font.SysFont('arial,malgungothic', 36, bold=True)

        self.world_x = self.MARGIN
        self.world_y = self.HEADER_H + self.MARGIN

        self.selected_id: Optional[int] = None
        self.selected_id_b: Optional[int] = None
        self.toast: Optional[tuple] = None

    def screen_to_world(self, sx, sy):
        return (sx - self.world_x, sy - self.world_y)

    def in_world(self, sx, sy) -> bool:
        return (self.world_x <= sx < self.world_x + WORLD_W
                and self.world_y <= sy < self.world_y + WORLD_H)

    def click(self, world: World, sx: int, sy: int, shift: bool = False):
        if not self.in_world(sx, sy):
            if shift:
                self.selected_id_b = None
            else:
                self.selected_id = None
                self.selected_id_b = None
            return
        wx, wy = self.screen_to_world(sx, sy)
        c = world.find_at(wx, wy, 12)
        cid = c.creature_id if c is not None else None
        if shift:
            self.selected_id_b = cid if cid != self.selected_id else None
        else:
            self.selected_id = cid
            self.selected_id_b = None

    def show_toast(self, msg: str, world_tick: int, duration: int = 90):
        self.toast = (msg, world_tick + duration)

    def draw(self, world: World, fast_forward: bool, paused: bool = False):
        self.screen.fill(BG_COLOR)
        self._draw_header(world, fast_forward, paused)
        self._draw_world(world)
        self._draw_sidebar(world)
        pygame.display.flip()

    def _draw_header(self, world: World, fast: bool, paused: bool):
        title = self.font_title.render('NEURAL EVOLUTION', True, TEXT_PRIMARY)
        self.screen.blit(title, (self.MARGIN, 14))
        sub = (f'prey gen {world.max_generation()} · pop {world.population()} · '
               f'pred {len(world.predators)} (g{max((p.generation for p in world.predators), default=0)}) · '
               f'walls {len(world.walls)} · tick {world.tick} · seed {world.seed}')
        self.screen.blit(self.font_sub.render(sub, True, TEXT_SECONDARY),
                          (self.MARGIN, 46))

        hint = ('SPACE fast · P pause · N step · R reset · V rec · S save · L load · '
                'X csv · F5/F9 snap · 좌클릭 select · shift+좌 compare · 우클릭 food · 중클릭 pred')
        self.screen.blit(self.font_tiny.render(hint, True, TEXT_DIM),
                          (self.MARGIN, 72))

        # 우측 칩
        chip_x = self.window_w - self.MARGIN - 130
        chip_y = 22
        if paused:
            self._draw_chip(chip_x, chip_y, 'PAUSED', ACCENT_RED)
            chip_y += 30
        if fast:
            self._draw_chip(chip_x, chip_y, 'FAST FORWARD', ACCENT_GOLD)

        if self.toast is not None:
            msg, until = self.toast
            if world.tick < until:
                self._draw_chip(self.MARGIN + 360, 22, msg, ACCENT_BLUE)
            else:
                self.toast = None

    def _draw_chip(self, x, y, text, bg):
        label = self.font_pin.render(text, True, BG_COLOR)
        pad_x, pad_y = 10, 6
        rect = pygame.Rect(x, y, label.get_width() + pad_x * 2,
                           label.get_height() + pad_y * 2)
        pygame.draw.rect(self.screen, bg, rect, border_radius=10)
        self.screen.blit(label, (rect.x + pad_x, rect.y + pad_y))

    def _draw_world(self, world: World):
        pad = 6
        panel = pygame.Rect(self.world_x - pad, self.world_y - pad,
                            WORLD_W + pad * 2, WORLD_H + pad * 2)
        pygame.draw.rect(self.screen, PANEL_COLOR, panel, border_radius=10)
        pygame.draw.rect(self.screen, WORLD_BG,
                         (self.world_x, self.world_y, WORLD_W, WORLD_H))

        # 벽
        for w in world.walls:
            rect = pygame.Rect(int(self.world_x + w.x), int(self.world_y + w.y),
                               int(w.w), int(w.h))
            pygame.draw.rect(self.screen, WALL_COLOR, rect, border_radius=3)
            pygame.draw.rect(self.screen, WALL_EDGE, rect, width=1, border_radius=3)

        # 페로몬
        for p in world.pheromones:
            alpha = max(40, int(180 * p.life / PHEROMONE_LIFE))
            surf = pygame.Surface((6, 6), pygame.SRCALPHA)
            pygame.draw.circle(surf, (*PHEROMONE_COLOR, alpha), (3, 3), 3)
            self.screen.blit(surf, (int(self.world_x + p.x - 3),
                                    int(self.world_y + p.y - 3)))

        # 음식 — 흔한/희귀 색 구분
        for f in world.foods:
            color = RARE_FOOD_COLOR if f.value > COMMON_FOOD_VALUE + 0.1 else FOOD_COLOR
            r = 3 if color == RARE_FOOD_COLOR else 2
            pygame.draw.circle(self.screen, color,
                               (int(self.world_x + f.x),
                                int(self.world_y + f.y)), r)

        # 포식자
        for pr in world.predators:
            self._draw_predator(pr)

        # prey
        top = sorted(world.creatures, key=lambda c: -c.generation)[:3]
        top_ids = {c.creature_id for c in top}

        selected_a: Optional[Creature] = None
        selected_b: Optional[Creature] = None
        for c in world.creatures:
            cx, cy = int(self.world_x + c.x), int(self.world_y + c.y)
            color = c.color()

            tail_x = cx - int(c.vx * 4)
            tail_y = cy - int(c.vy * 4)
            pygame.draw.line(self.screen,
                             (color[0] // 2, color[1] // 2, color[2] // 2),
                             (cx, cy), (tail_x, tail_y), 2)
            pygame.draw.circle(self.screen, color, (cx, cy), SIZE)
            if c.creature_id in top_ids:
                pygame.draw.circle(self.screen, ACCENT_GOLD,
                                   (cx, cy), SIZE + 2, 1)
            if c.creature_id == self.selected_id:
                selected_a = c
            if c.creature_id == self.selected_id_b:
                selected_b = c

            bar_w = SIZE * 2
            ratio = max(0, min(1, c.energy / MAX_ENERGY))
            pygame.draw.rect(self.screen, (40, 44, 54),
                             (cx - bar_w // 2, cy - SIZE - 5, bar_w, 2))
            pygame.draw.rect(self.screen, (220, 220, 220),
                             (cx - bar_w // 2, cy - SIZE - 5,
                              int(bar_w * ratio), 2))

        if selected_a is not None:
            self._draw_selected_overlay(selected_a, primary=True)
        if selected_b is not None:
            self._draw_selected_overlay(selected_b, primary=False)

    def _draw_predator(self, pr):
        px, py = int(self.world_x + pr.x), int(self.world_y + pr.y)
        halo = pygame.Surface(
            (int(PREDATOR_VISION * 2), int(PREDATOR_VISION * 2)),
            pygame.SRCALPHA)
        pygame.draw.circle(halo, (220, 70, 80, 14),
                           (int(PREDATOR_VISION), int(PREDATOR_VISION)),
                           int(PREDATOR_VISION))
        self.screen.blit(halo, (px - int(PREDATOR_VISION),
                                py - int(PREDATOR_VISION)))
        ang = math.atan2(pr.vy, pr.vx)
        r = PREDATOR_SIZE + 2
        pts = [
            (px + math.cos(ang) * r, py + math.sin(ang) * r),
            (px + math.cos(ang + 2.4) * r * 0.8,
             py + math.sin(ang + 2.4) * r * 0.8),
            (px + math.cos(ang - 2.4) * r * 0.8,
             py + math.sin(ang - 2.4) * r * 0.8),
        ]
        body_color = pr.color()
        pygame.draw.polygon(self.screen, body_color, pts)
        pygame.draw.polygon(self.screen, (60, 20, 25), pts, 1)
        # 세대 표시 작은 점
        if pr.generation >= 2:
            pygame.draw.circle(self.screen, ACCENT_GOLD, (px, py), 1)

    def _draw_selected_overlay(self, c: Creature, primary: bool):
        cx, cy = int(self.world_x + c.x), int(self.world_y + c.y)
        r = int(VISION)
        wedge = pygame.Surface((r * 2 + 2, r * 2 + 2), pygame.SRCALPHA)
        wcx = r + 1
        wcy = r + 1
        pygame.draw.circle(wedge, (255, 255, 255, 22), (wcx, wcy), r, 1)

        if c.last_inputs is not None and primary:
            food_sect = c.last_inputs[0:8]
            pred_sect = c.last_inputs[8:16]
            kin_sect = c.last_inputs[16:24]
            phero_sect = c.last_inputs[24:32]
            sa = math.tau / N_SECTORS
            for s in range(N_SECTORS):
                a0 = s * sa
                a1 = a0 + sa
                f = float(food_sect[s])
                p = float(pred_sect[s])
                k = float(kin_sect[s])
                ph = float(phero_sect[s])
                if f > 0.05:
                    self._wedge(wedge, wcx, wcy, r, a0, a1,
                                (90, 200, 110, int(25 + 100 * f)))
                if p > 0.05:
                    self._wedge(wedge, wcx, wcy, r, a0, a1,
                                (230, 80, 90, int(40 + 130 * p)))
                if k > 0.05:
                    self._wedge(wedge, wcx, wcy, r, a0, a1,
                                (110, 200, 240, int(20 + 80 * k)))
                if ph > 0.05:
                    self._wedge(wedge, wcx, wcy, r, a0, a1,
                                (200, 140, 240, int(15 + 80 * ph)))
        self.screen.blit(wedge, (cx - wcx, cy - wcy))

        col = SELECT_COLOR if primary else SELECT_B_COLOR
        pygame.draw.circle(self.screen, col, (cx, cy), SIZE + 4, 2)

    def _wedge(self, surf, cx, cy, r, a0, a1, color):
        pts = [(cx, cy)]
        steps = max(2, int((a1 - a0) * 10))
        for i in range(steps + 1):
            a = a0 + (a1 - a0) * i / steps
            pts.append((cx + math.cos(a) * r, cy + math.sin(a) * r))
        pygame.draw.polygon(surf, color, pts)

    def _draw_sidebar(self, world: World):
        x = self.world_x + WORLD_W + self.MARGIN
        y = self.world_y - 6
        w = self.SIDEBAR_W
        h = WORLD_H + 12
        pygame.draw.rect(self.screen, PANEL_COLOR, (x, y, w, h),
                         border_radius=10)

        inner_x = x + 16
        inner_w = w - 32
        cur_y = y + 14

        self.screen.blit(self.font_stat.render('POPULATION', True, TEXT_SECONDARY),
                         (inner_x, cur_y)); cur_y += 20
        pop_surf = self.font_big.render(str(world.population()), True, ACCENT_GOLD)
        self.screen.blit(pop_surf, (inner_x, cur_y))
        kc = self.font_stat.render(f'kills {world.kills}', True, PREDATOR_COLOR)
        self.screen.blit(kc, (inner_x + inner_w - kc.get_width(), cur_y + 20))
        cur_y += 46

        if world.history:
            last = world.history[-1]
            self._draw_stat(inner_x, cur_y, inner_w, 'max gen / pred gen',
                            f'{last["max_gen"]} / {last["pred_max_gen"]}'); cur_y += 18
            self._draw_stat(inner_x, cur_y, inner_w, 'avg eaten',
                            f'{last["avg_eaten"]:.1f}'); cur_y += 18
            self._draw_stat(inner_x, cur_y, inner_w, 'sexual / total births',
                            f'{world.sexual_births} / {world.births}'); cur_y += 18
            self._draw_stat(inner_x, cur_y, inner_w, 'diversity',
                            f'{last["diversity"]:.2f}'); cur_y += 18
            self._draw_stat(inner_x, cur_y, inner_w, 'predators / pheromones',
                            f'{last["predators"]} / {last["pheromones"]}'); cur_y += 36

        graph_h = 40
        self._draw_titled_graph(inner_x, cur_y, inner_w, graph_h, 'POPULATION',
                                'count', world.history, (160, 200, 240))
        cur_y += graph_h + 26
        self._draw_titled_graph(inner_x, cur_y, inner_w, graph_h, 'AVG EATEN',
                                'avg_eaten', world.history, (220, 200, 120))
        cur_y += graph_h + 26
        self._draw_titled_graph(inner_x, cur_y, inner_w, graph_h, 'DIVERSITY',
                                'diversity', world.history, ACCENT_PINK)
        cur_y += graph_h + 26
        # H: 조상별 인구 스택 차트
        self._draw_stacked_ancestor_chart(inner_x, cur_y, inner_w, graph_h, world)
        cur_y += graph_h + 22

        sel_a = self._find(world, self.selected_id)
        sel_b = self._find(world, self.selected_id_b)
        if self.selected_id_b is not None and sel_b is None:
            self.selected_id_b = None
        if self.selected_id is not None and sel_a is None:
            self.selected_id = None

        remaining = h - (cur_y - y) - 14
        if sel_a is not None and sel_b is not None:
            self._draw_compare(inner_x, cur_y, inner_w, remaining, sel_a, sel_b)
        elif sel_a is not None:
            self._draw_inspector(inner_x, cur_y, inner_w, remaining, world, sel_a)
        else:
            self._draw_top_brain(inner_x, cur_y, inner_w, remaining, world)

    def _find(self, world: World, cid: Optional[int]) -> Optional[Creature]:
        if cid is None:
            return None
        for c in world.creatures:
            if c.creature_id == cid:
                return c
        return None

    def _draw_stat(self, x, y, w, label, value):
        lab = self.font_label.render(label, True, TEXT_SECONDARY)
        val = self.font_stat.render(value, True, TEXT_PRIMARY)
        self.screen.blit(lab, (x, y + 2))
        self.screen.blit(val, (x + w - val.get_width(), y))

    def _draw_titled_graph(self, x, y, w, h, title, key, history, color):
        ts = self.font_stat.render(title, True, TEXT_SECONDARY)
        self.screen.blit(ts, (x, y - 16))
        self._draw_graph(x, y, w, h, key, history, color)

    def _draw_graph(self, x, y, w, h, key, history, color):
        pygame.draw.rect(self.screen, (28, 32, 40), (x, y, w, h),
                         border_radius=4)
        if not history:
            return
        data = history[-200:]
        values = [d.get(key, 0) for d in data]
        if not values:
            return
        hi = max(values) * 1.1 + 1
        if hi <= 0:
            return
        pts = []
        for i, v in enumerate(values):
            px = x + i * w / max(1, len(values) - 1)
            py = y + h - v / hi * h
            pts.append((px, py))
        if len(pts) >= 2:
            pygame.draw.lines(self.screen, color, False, pts, 2)

    def _draw_stacked_ancestor_chart(self, x, y, w, h, world: World):
        """조상별 인구를 색깔 스택 영역 차트로."""
        self.screen.blit(self.font_stat.render('ANCESTOR LINEAGES', True,
                                                TEXT_SECONDARY),
                         (x, y - 16))
        pygame.draw.rect(self.screen, (28, 32, 40), (x, y, w, h),
                         border_radius=4)
        if not world.history:
            return
        # 현재 시점의 top 조상 (인원 많은 순)
        cur_counts = world.history[-1].get('ancestor_counts', {})
        if not cur_counts:
            return
        top_ancestors = sorted(cur_counts.items(), key=lambda kv: -kv[1])[:6]
        anc_ids = [aid for aid, _ in top_ancestors]

        data = world.history[-200:]
        if len(data) < 2:
            return
        n = len(data)
        # max total height
        max_total = max(d.get('count', 0) for d in data) + 1

        # 각 시점의 stack: bottom→top 누적
        for i in range(n):
            counts = data[i].get('ancestor_counts', {})
            cum = 0
            px = x + i * w / max(1, n - 1)
            for k, aid in enumerate(anc_ids):
                c = counts.get(aid, 0)
                if c == 0:
                    continue
                y_bot = y + h - cum / max_total * h
                y_top = y + h - (cum + c) / max_total * h
                color = ANCESTOR_PALETTE[k % len(ANCESTOR_PALETTE)]
                pygame.draw.line(self.screen, color,
                                  (px, y_bot), (px, y_top), 1)
                cum += c
        # 범례 (작게)
        lg_x = x + 4
        lg_y = y + 4
        for k, (aid, cnt) in enumerate(top_ancestors):
            color = ANCESTOR_PALETTE[k % len(ANCESTOR_PALETTE)]
            pygame.draw.rect(self.screen, color, (lg_x, lg_y, 8, 8))
            self.screen.blit(self.font_tiny.render(
                f'#{aid}:{cnt}', True, TEXT_PRIMARY),
                (lg_x + 11, lg_y - 2))
            lg_y += 11
            if lg_y + 11 > y + h - 2:
                break

    def _draw_top_brain(self, x, y, w, h, world: World):
        self.screen.blit(self.font_stat.render('TOP BRAIN', True, TEXT_SECONDARY),
                         (x, y))
        if not world.creatures:
            return
        top = max(world.creatures, key=lambda c: c.generation)
        self._draw_brain(x, y + 16, w, h - 16, top.brain,
                          inputs=top.last_inputs,
                          outputs=top.last_outputs)

    def _draw_inspector(self, x, y, w, h, world: World, c: Creature):
        self.screen.blit(self.font_stat.render('INSPECTOR', True, ACCENT_BLUE),
                         (x, y))
        cur_y = y + 18
        col = c.color()
        pygame.draw.rect(self.screen, col, (x, cur_y, 12, 12), border_radius=2)
        info = f'id {c.creature_id}  gen {c.generation}  age {c.age}'
        self.screen.blit(self.font_label.render(info, True, TEXT_PRIMARY),
                         (x + 18, cur_y + 1))
        cur_y += 16
        e = (f'energy {c.energy:.1f}  eaten {c.lifetime_eaten}  '
             f'parents {len(c.parent_ids)}  anc {c.ancestor_id}')
        self.screen.blit(self.font_label.render(e, True, TEXT_SECONDARY),
                         (x, cur_y))
        cur_y += 16

        brain_h = max(90, h - 80)
        self._draw_brain(x, cur_y, w, brain_h, c.brain,
                          inputs=c.last_inputs, outputs=c.last_outputs)
        cur_y += brain_h + 6

        self.screen.blit(self.font_label.render('LINEAGE', True, TEXT_SECONDARY),
                         (x, cur_y))
        cur_y += 12
        chain = self._ancestors(world, c.creature_id, max_depth=8)
        chain_h = max(28, h - (cur_y - y))
        self._draw_lineage_chain(x, cur_y, w, chain_h, chain)

    def _draw_compare(self, x, y, w, h, a: Creature, b: Creature):
        self.screen.blit(self.font_stat.render('COMPARE', True, ACCENT_PINK),
                         (x, y))
        cur_y = y + 18
        dist = Brain.distance(a.brain, b.brain)
        info = (f'A id{a.creature_id} g{a.generation}  '
                f'B id{b.creature_id} g{b.generation}  '
                f'Δw={dist:.2f}')
        self.screen.blit(self.font_label.render(info, True, TEXT_PRIMARY),
                         (x, cur_y))
        cur_y += 16

        pygame.draw.rect(self.screen, a.color(), (x, cur_y, 14, 14),
                         border_radius=2)
        pygame.draw.rect(self.screen, (255, 255, 255), (x, cur_y, 14, 14),
                         width=1, border_radius=2)
        pygame.draw.rect(self.screen, b.color(), (x + 22, cur_y, 14, 14),
                         border_radius=2)
        pygame.draw.rect(self.screen, SELECT_B_COLOR, (x + 22, cur_y, 14, 14),
                         width=1, border_radius=2)
        cur_y += 20

        each_h = max(50, (h - (cur_y - y) - 8) // 2)
        self._draw_brain(x, cur_y, w, each_h, a.brain,
                          inputs=a.last_inputs, outputs=a.last_outputs)
        cur_y += each_h + 4
        self._draw_brain(x, cur_y, w, each_h, b.brain,
                          inputs=b.last_inputs, outputs=b.last_outputs)

    def _ancestors(self, world: World, cid: int, max_depth: int):
        chain = []
        cur = cid
        for _ in range(max_depth):
            rec = world.lineage_records.get(cur)
            if rec is None:
                break
            chain.append(rec)
            if not rec.parent_ids:
                break
            cur = rec.parent_ids[0]
        return list(reversed(chain))

    def _draw_lineage_chain(self, x, y, w, h, chain):
        pygame.draw.rect(self.screen, (28, 32, 40), (x, y, w, h),
                         border_radius=4)
        if not chain:
            return
        n = len(chain)
        step = w / max(1, n)
        for i, rec in enumerate(chain):
            cx = int(x + i * step + step / 2)
            cy = int(y + h / 2)
            r = 6 if len(rec.parent_ids) <= 1 else 7
            pygame.draw.circle(self.screen, rec.color, (cx, cy), r)
            if len(rec.parent_ids) == 2:
                pygame.draw.circle(self.screen, (255, 255, 255),
                                   (cx, cy), r + 1, 1)
            lbl = self.font_tiny.render(f'g{rec.generation}', True, TEXT_DIM)
            self.screen.blit(lbl, (cx - lbl.get_width() // 2, cy + 8))
            if i > 0:
                px = int(x + (i - 1) * step + step / 2)
                pygame.draw.line(self.screen, TEXT_DIM, (px, cy), (cx, cy), 1)

    def _draw_brain(self, x, y, w, h, brain: Brain,
                     inputs=None, outputs=None):
        pygame.draw.rect(self.screen, (28, 32, 40), (x, y, w, h),
                         border_radius=4)
        pad = 12
        n_in = brain.W1.shape[0]
        n_hidden = brain.W1.shape[1]
        n_out = brain.W2.shape[1]
        col_xs = [x + pad, x + w // 2, x + w - pad]
        nodes = [n_in, n_hidden, n_out]
        node_ys = []
        for col, nn in zip(col_xs, nodes):
            if nn == 1:
                ys = [y + h // 2]
            else:
                step = (h - pad * 2) / (nn - 1)
                ys = [y + pad + i * step for i in range(nn)]
            node_ys.append(ys)

        def line_color(wv):
            if wv > 0:
                a = min(255, int(60 + abs(wv) * 100))
                return (60, a, 220)
            else:
                a = min(255, int(60 + abs(wv) * 100))
                return (220, 80, 60)

        thresh = 0.30
        for i in range(n_in):
            for j in range(n_hidden):
                wv = float(brain.W1[i, j])
                if abs(wv) < thresh:
                    continue
                pygame.draw.line(self.screen, line_color(wv),
                                 (col_xs[0], node_ys[0][i]),
                                 (col_xs[1], node_ys[1][j]),
                                 max(1, int(abs(wv))))
        for i in range(n_hidden):
            for j in range(n_out):
                wv = float(brain.W2[i, j])
                if abs(wv) < thresh:
                    continue
                pygame.draw.line(self.screen, line_color(wv),
                                 (col_xs[1], node_ys[1][i]),
                                 (col_xs[2], node_ys[2][j]),
                                 max(1, int(abs(wv))))

        def act_color(act):
            a = (math.tanh(act) * 0.5 + 0.5)
            v = int(40 + a * 215)
            return (v, v, v)

        for col_idx, (col, ys) in enumerate(zip(col_xs, node_ys)):
            for i, ny in enumerate(ys):
                if col_idx == 0 and inputs is not None and i < len(inputs):
                    color = act_color(float(inputs[i]))
                elif col_idx == 2 and outputs is not None and i < len(outputs):
                    color = act_color(float(outputs[i]))
                else:
                    color = (180, 180, 190)
                pygame.draw.circle(self.screen, color,
                                   (int(col), int(ny)), 3)
                pygame.draw.circle(self.screen, (60, 64, 72),
                                   (int(col), int(ny)), 3, 1)

    def tick_clock(self, fps: int):
        self.clock.tick(fps)

    def quit(self):
        pygame.quit()
