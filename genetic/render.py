"""유전 진화 시뮬 시각화.

neural/ 과의 차별점을 강조해서 보여준다:
- 월드 배경에 biome 3분할 색조 (cold/temperate/warm)
- 개체에 sex 마커(♂/♀) + pattern 줄무늬 + hawk 표식
- 사이드바: trait scatter(speed×vision), hawk 비율, biome 분포
"""
import math
import pygame
from typing import List

from .world import (World, WORLD_W, WORLD_H, Sex,
                    BIOME_COLD, BIOME_TEMPERATE, BIOME_WARM,
                    AGGR_HAWK_CUTOFF, MAX_ENERGY)


BG_COLOR = (15, 17, 21)
PANEL_COLOR = (22, 25, 31)
WORLD_BG = (20, 26, 22)
GRID_LINE = (28, 36, 30)
FOOD_COLOR = (140, 200, 90)
TEXT_PRIMARY = (235, 235, 240)
TEXT_SECONDARY = (140, 145, 160)
ACCENT_GOLD = (255, 215, 90)
ACCENT_BLUE = (130, 180, 240)
ACCENT_RED = (240, 110, 100)

# biome 색조 (월드 배경 위 얇은 오버레이)
BIOME_TINT = {
    BIOME_COLD:     (40, 70, 110, 55),     # 푸르스름
    BIOME_TEMPERATE: (40, 60, 50, 25),     # 거의 중립
    BIOME_WARM:     (110, 70, 40, 55),     # 오렌지빛
}
BIOME_LABELS = {BIOME_COLD: 'COLD', BIOME_TEMPERATE: 'TEMP', BIOME_WARM: 'WARM'}
BIOME_BAR_COLORS = {
    BIOME_COLD:      (130, 170, 230),
    BIOME_TEMPERATE: (180, 200, 170),
    BIOME_WARM:      (240, 170, 110),
}

SEX_M_OUTLINE = (160, 200, 240)
SEX_F_OUTLINE = (240, 180, 130)


class Renderer:
    HEADER_H = 80
    SIDEBAR_W = 340
    MARGIN = 24

    def __init__(self):
        pygame.init()
        self.window_w = WORLD_W + self.SIDEBAR_W + self.MARGIN * 3
        self.window_h = WORLD_H + self.HEADER_H + self.MARGIN * 2
        self.screen = pygame.display.set_mode((self.window_w, self.window_h))
        pygame.display.set_caption('Evolution — Genetic Algorithm')
        self.clock = pygame.time.Clock()

        self.font_title = pygame.font.SysFont('arial,malgungothic', 28, bold=True)
        self.font_sub = pygame.font.SysFont('arial,malgungothic', 13)
        self.font_stat = pygame.font.SysFont('arial,malgungothic', 13, bold=True)
        self.font_pin = pygame.font.SysFont('arial,malgungothic', 12, bold=True)
        self.font_label = pygame.font.SysFont('arial,malgungothic', 11)
        self.font_big = pygame.font.SysFont('arial,malgungothic', 36, bold=True)
        self.font_biome = pygame.font.SysFont('arial,malgungothic', 10, bold=True)

        self.world_x = self.MARGIN
        self.world_y = self.HEADER_H + self.MARGIN

        # biome 오버레이 surface — 매번 새로 만들지 않도록 캐시
        self._biome_overlay = self._make_biome_overlay()

        self.graph_max_points = 200

    def _make_biome_overlay(self) -> pygame.Surface:
        surf = pygame.Surface((WORLD_W, WORLD_H), pygame.SRCALPHA)
        third = WORLD_W // 3
        rects = [
            (BIOME_COLD,      pygame.Rect(0, 0, third, WORLD_H)),
            (BIOME_TEMPERATE, pygame.Rect(third, 0, third, WORLD_H)),
            (BIOME_WARM,      pygame.Rect(third * 2, 0, WORLD_W - third * 2, WORLD_H)),
        ]
        for biome, rect in rects:
            pygame.draw.rect(surf, BIOME_TINT[biome], rect)
        # 경계선
        for i in (1, 2):
            x = third * i
            pygame.draw.line(surf, (10, 12, 16, 70), (x, 0), (x, WORLD_H), 1)
        return surf

    # ── 외부 진입점 ──

    def draw(self, world: World, fast_forward: bool):
        self.screen.fill(BG_COLOR)
        self._draw_header(world, fast_forward)
        self._draw_world(world)
        self._draw_sidebar(world)
        pygame.display.flip()

    # ── 헤더 ──

    def _draw_header(self, world: World, fast: bool):
        title = self.font_title.render('EVOLUTION', True, TEXT_PRIMARY)
        self.screen.blit(title, (self.MARGIN, 22))

        season = world.season_factor()
        season_label = 'BLOOM' if season > 1.15 else ('FAMINE' if season < 0.85 else 'STABLE')
        sub = (f'gen {world.max_generation()}  ·  pop {world.population()}  ·  '
               f'tick {world.tick}  ·  seed {world.seed}  ·  '
               f'season {season_label} ({season:.2f}×)')
        sub_surf = self.font_sub.render(sub, True, TEXT_SECONDARY)
        self.screen.blit(sub_surf, (self.MARGIN, 56))

        if fast:
            label = self.font_pin.render('FAST FORWARD', True, BG_COLOR)
            pad_x, pad_y = 10, 6
            rect = pygame.Rect(
                self.window_w - self.MARGIN - label.get_width() - pad_x * 2, 32,
                label.get_width() + pad_x * 2, label.get_height() + pad_y * 2,
            )
            pygame.draw.rect(self.screen, ACCENT_GOLD, rect, border_radius=10)
            self.screen.blit(label, (rect.x + pad_x, rect.y + pad_y))

    # ── 월드 ──

    def _draw_world(self, world: World):
        pad = 6
        panel = pygame.Rect(self.world_x - pad, self.world_y - pad,
                            WORLD_W + pad * 2, WORLD_H + pad * 2)
        pygame.draw.rect(self.screen, PANEL_COLOR, panel, border_radius=10)
        pygame.draw.rect(self.screen, WORLD_BG,
                         (self.world_x, self.world_y, WORLD_W, WORLD_H))

        # biome 오버레이
        self.screen.blit(self._biome_overlay, (self.world_x, self.world_y))

        # biome 라벨 (좌상단에 작게)
        third = WORLD_W // 3
        for biome in (BIOME_COLD, BIOME_TEMPERATE, BIOME_WARM):
            lab = self.font_biome.render(BIOME_LABELS[biome], True, (220, 220, 230))
            self.screen.blit(lab, (self.world_x + biome * third + 8, self.world_y + 6))

        # 음식
        for f in world.foods:
            pygame.draw.circle(self.screen, FOOD_COLOR,
                               (int(self.world_x + f.x), int(self.world_y + f.y)), 2)

        # 개체
        for c in world.creatures:
            cx = int(self.world_x + c.x)
            cy = int(self.world_y + c.y)
            r = max(2, int(c.genome.size_px()))
            color = c.color()

            # 시야 (반투명)
            if c.energy > 0:
                vis_r = int(c.genome.vision_px())
                vision_surf = pygame.Surface((vis_r * 2, vis_r * 2), pygame.SRCALPHA)
                pygame.draw.circle(vision_surf, (color[0], color[1], color[2], 14),
                                   (vis_r, vis_r), vis_r)
                self.screen.blit(vision_surf, (cx - vis_r, cy - vis_r))

            # 본체
            pygame.draw.circle(self.screen, color, (cx, cy), r)

            # pattern — 줄무늬 (성선택 ornament)
            n_stripes = c.genome.pattern_count()
            if n_stripes > 0 and r >= 3:
                stripe_color = (max(0, color[0] - 50),
                                max(0, color[1] - 50),
                                max(0, color[2] - 50))
                for s in range(n_stripes):
                    yy = cy - r + int((s + 1) * 2 * r / (n_stripes + 1))
                    dx_w = int(math.sqrt(max(0, r * r - (yy - cy) ** 2)))
                    pygame.draw.line(self.screen, stripe_color,
                                     (cx - dx_w, yy), (cx + dx_w, yy), 1)

            # 성별 외곽선
            outline = SEX_F_OUTLINE if c.sex == Sex.F else SEX_M_OUTLINE
            pygame.draw.circle(self.screen, outline, (cx, cy), r + 1, 1)

            # hawk 표시 — 작은 빨간 점
            if c.genome.aggression > AGGR_HAWK_CUTOFF:
                pygame.draw.circle(self.screen, ACCENT_RED, (cx + r, cy - r), 2)

            # 에너지바
            bar_w = max(8, r * 2)
            ratio = max(0.0, min(1.0, c.energy / MAX_ENERGY))
            pygame.draw.rect(self.screen, (40, 44, 54),
                             (cx - bar_w // 2, cy - r - 6, bar_w, 2))
            pygame.draw.rect(self.screen, (220, 220, 220),
                             (cx - bar_w // 2, cy - r - 6, int(bar_w * ratio), 2))

    # ── 사이드바 ──

    def _draw_sidebar(self, world: World):
        x = self.world_x + WORLD_W + self.MARGIN
        y = self.world_y - 6
        w = self.SIDEBAR_W
        h = WORLD_H + 12
        pygame.draw.rect(self.screen, PANEL_COLOR, (x, y, w, h), border_radius=10)

        inner_x = x + 16
        inner_w = w - 32
        cur_y = y + 14

        # 인구 + 성비
        title = self.font_stat.render('POPULATION', True, TEXT_SECONDARY)
        self.screen.blit(title, (inner_x, cur_y))
        cur_y += 18

        pop_surf = self.font_big.render(str(world.population()), True, ACCENT_GOLD)
        self.screen.blit(pop_surf, (inner_x, cur_y))
        females = sum(1 for c in world.creatures if c.sex == Sex.F)
        males = world.population() - females
        sex_surf = self.font_sub.render(f'♀ {females}   ♂ {males}', True, TEXT_SECONDARY)
        self.screen.blit(sex_surf, (inner_x + pop_surf.get_width() + 14,
                                    cur_y + pop_surf.get_height() - 18))
        cur_y += pop_surf.get_height() + 8

        # 2열 통계 그리드
        last = world.history[-1] if world.history else {}
        rows = [
            ('avg speed',  f'{last.get("avg_speed", 0):.2f}',
             'avg vision', f'{last.get("avg_vision", 0):.0f}'),
            ('avg size',   f'{last.get("avg_size", 0):.2f}',
             'avg metab',  f'{last.get("avg_metab", 0):.3f}'),
            ('avg life',   f'{last.get("avg_lifespan", 0):.0f}',
             'avg pattern', f'{last.get("avg_pattern", 0):.2f}'),
            ('max gen',    str(world.max_generation()),
             'hawk %',     f'{last.get("hawk_ratio", 0) * 100:.0f}%'),
            ('births',     str(world.births),
             'deaths',     str(world.deaths)),
        ]
        col_w = inner_w // 2
        for label_a, val_a, label_b, val_b in rows:
            self._draw_stat(inner_x, cur_y, col_w - 4, label_a, val_a)
            self._draw_stat(inner_x + col_w + 4, cur_y, col_w - 4, label_b, val_b)
            cur_y += 19
        cur_y += 6

        # biome 분포 — 3 막대
        title_b = self.font_stat.render('BIOME DISTRIBUTION', True, TEXT_SECONDARY)
        self.screen.blit(title_b, (inner_x, cur_y))
        cur_y += 16
        cold, temp, warm = world.biome_populations()
        total = max(1, cold + temp + warm)
        bar_h = 14
        for biome, count, name in [(BIOME_COLD, cold, 'cold'),
                                   (BIOME_TEMPERATE, temp, 'temp'),
                                   (BIOME_WARM, warm, 'warm')]:
            lab = self.font_label.render(name, True, TEXT_SECONDARY)
            self.screen.blit(lab, (inner_x, cur_y + 1))
            bar_x = inner_x + 36
            bar_w_max = inner_w - 36 - 30
            pygame.draw.rect(self.screen, (28, 32, 40),
                             (bar_x, cur_y, bar_w_max, bar_h), border_radius=3)
            fill = int(bar_w_max * count / total)
            pygame.draw.rect(self.screen, BIOME_BAR_COLORS[biome],
                             (bar_x, cur_y, fill, bar_h), border_radius=3)
            val = self.font_label.render(str(count), True, TEXT_PRIMARY)
            self.screen.blit(val, (bar_x + bar_w_max + 6, cur_y + 1))
            cur_y += bar_h + 3
        cur_y += 6

        # 인구 그래프
        title2 = self.font_stat.render('POPULATION', True, TEXT_SECONDARY)
        self.screen.blit(title2, (inner_x, cur_y))
        cur_y += 16
        self._draw_graph(inner_x, cur_y, inner_w, 56, 'count', world.history,
                         color=(160, 200, 240))
        cur_y += 56 + 8

        # hawk 비율 — 0~1, 50% 점선 가이드
        title_h = self.font_stat.render('HAWK RATIO (game-theoretic)', True, TEXT_SECONDARY)
        self.screen.blit(title_h, (inner_x, cur_y))
        cur_y += 16
        self._draw_graph(inner_x, cur_y, inner_w, 50, 'hawk_ratio', world.history,
                         color=ACCENT_RED, normalize=(0.0, 1.0),
                         guide_y=0.5)
        cur_y += 50 + 8

        # trait scatter — speed × vision
        title_s = self.font_stat.render('TRAITS — speed × vision', True, TEXT_SECONDARY)
        self.screen.blit(title_s, (inner_x, cur_y))
        cur_y += 16
        self._draw_scatter(inner_x, cur_y, inner_w, 120, world)

    def _draw_stat(self, x: int, y: int, w: int, label: str, value: str):
        lab = self.font_label.render(label, True, TEXT_SECONDARY)
        val = self.font_stat.render(value, True, TEXT_PRIMARY)
        self.screen.blit(lab, (x, y + 2))
        self.screen.blit(val, (x + w - val.get_width(), y))

    def _draw_graph(self, x: int, y: int, w: int, h: int, key: str,
                    history: List[dict], color: tuple,
                    normalize: tuple = None, guide_y: float = None):
        pygame.draw.rect(self.screen, (28, 32, 40), (x, y, w, h), border_radius=4)
        if not history:
            return

        data = history[-self.graph_max_points:]
        values = [d.get(key, 0) for d in data]
        if not values:
            return

        if normalize:
            lo, hi = normalize
        else:
            lo = 0
            hi = max(values) * 1.1 + 1

        if hi <= lo:
            return

        # 가이드 라인
        if guide_y is not None and lo <= guide_y <= hi:
            gy = y + h - (guide_y - lo) / (hi - lo) * h
            pygame.draw.line(self.screen, (60, 65, 75),
                             (x + 2, gy), (x + w - 2, gy), 1)

        pts = []
        for i, v in enumerate(values):
            px = x + i * w / max(1, len(values) - 1)
            py = y + h - (v - lo) / (hi - lo) * h
            pts.append((px, py))
        if len(pts) >= 2:
            pygame.draw.lines(self.screen, color, False, pts, 2)

    def _draw_scatter(self, x: int, y: int, w: int, h: int, world: World):
        """speed × vision 산점도. 각 점 = 한 개체, 색 = 유전체 색.
        축: x = speed(0~1), y = vision(0~1, 위로 갈수록 큼)."""
        pygame.draw.rect(self.screen, (28, 32, 40), (x, y, w, h), border_radius=4)

        # 축 가이드
        pygame.draw.line(self.screen, (45, 50, 60),
                         (x + w // 2, y + 2), (x + w // 2, y + h - 2), 1)
        pygame.draw.line(self.screen, (45, 50, 60),
                         (x + 2, y + h // 2), (x + w - 2, y + h // 2), 1)

        # 점
        for c in world.creatures:
            px = int(x + 3 + c.genome.speed * (w - 6))
            py = int(y + h - 3 - c.genome.vision * (h - 6))
            color = c.color()
            pygame.draw.circle(self.screen, color, (px, py), 2)

        # 축 라벨
        lab_x = self.font_label.render('speed →', True, TEXT_SECONDARY)
        lab_y = self.font_label.render('↑ vision', True, TEXT_SECONDARY)
        self.screen.blit(lab_x, (x + w - lab_x.get_width() - 4, y + h - 14))
        self.screen.blit(lab_y, (x + 4, y + 2))

    def tick_clock(self, fps: int):
        self.clock.tick(fps)

    def quit(self):
        pygame.quit()
